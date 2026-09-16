"""Song requests: an approval embed with buttons, then the paid generation."""

import asyncio
import logging
import time

import discord

from .llm_chat import ChatError
from .tools.files import post_file

log = logging.getLogger("red.agenteliza")

# The vote window of a song request: idle this long without a click and the
# request expires, unbilled. Only a vote resets the clock.
MUSIC_REQUEST_IDLE = 600
# The value cap of one embed field (a Discord limit): the prompt and the
# lyrics truncate to it.
MUSIC_EMBED_FIELD_MAX = 1024


def _clip(text: str) -> str:
    """One embed field value: the text, truncated at the field cap with a note."""
    if len(text) <= MUSIC_EMBED_FIELD_MAX:
        return text
    dropped = len(text) - MUSIC_EMBED_FIELD_MAX
    return text[:MUSIC_EMBED_FIELD_MAX] + f"\n[{dropped} characters dropped]"


class SongView(discord.ui.View):
    """The buttons of an open song request: Approve and Reject."""

    def __init__(self, manager: "SongManager", session_id: int):
        super().__init__(timeout=None)
        self.manager = manager
        self.session_id = session_id
        for label, suffix, style in (
            ("Approve", "a", discord.ButtonStyle.success)
            , ("Reject", "r", discord.ButtonStyle.danger)
        ):
            button = discord.ui.Button(style=style, label=label, custom_id=f"eliza-music:{session_id}:{suffix}")
            button.callback = self._make_callback(suffix)
            self.add_item(button)

    def _make_callback(self, suffix: str):
        async def callback(interaction: discord.Interaction):
            await self.manager.vote(self.session_id, interaction, "approve" if suffix == "a" else "reject")
        return callback


class SongManager:
    """The song request of each session: the approval vote, then the generation.

    A request spends money, so the vote comes first: an embed with the
    parameters and the price, Approve and Reject below it. A direct message
    closes on the click of its single reader, a guild at 60 percent of the
    active users. The approval runs the generation as a background task: the
    queue call, the retrieve polls, the audio post, the complete call. The
    state persists in the Config global `music`: a reload re-registers the
    live views and resumes a queued generation."""

    def __init__(self, channel_getter, discord_call, config=None):
        # Async callable returning the channel of an id, cache first then the API.
        self.channel_getter = channel_getter
        # The retry wrapper of the cog for Discord API calls.
        self.discord_call = discord_call
        # The cog Config, for the persisted request states. None: RAM only.
        self.config = config
        # Async callback (session_id, channel, harness text): a song event
        # without a user message wakes the agent. Wired by the cog.
        self.on_event = None
        # Async callable (session_id) returning the ids of the active users
        # of the conversation, for the majority rule of a guild vote.
        self.participants_getter = None
        # Async callables of the provider audio flow, wired by the cog:
        # queuer(body) -> (model, queue_id), retriever(model, queue_id) -> (name, bytes).
        self.queuer = None
        self.retriever = None
        self.active: dict = {}

    def _embed(self, state: dict, status: str | None = None) -> discord.Embed:
        """The embed of a request: the parameters as fields, the price in the footer
        (the bottom line of the embed, directly above the buttons)."""
        colors = {
            None: discord.Color.gold()
            , "approved": discord.Color.green()
            , "posted": discord.Color.green()
            , "rejected": discord.Color.red()
            , "expired": discord.Color.light_gray()
            , "replaced": discord.Color.light_gray()
            , "failed": discord.Color.red()
        }
        request = state["request"]
        embed = discord.Embed(title="Song request", color=colors.get(status, discord.Color.gold()))
        embed.add_field(name="Model", value=f"{request['preset']} ({request['model']})", inline=False)
        embed.add_field(name="Prompt", value=_clip(request["prompt"]), inline=False)
        if request.get("lyrics"):
            embed.add_field(name="Lyrics", value=_clip(request["lyrics"]), inline=False)
        if request.get("duration") is not None:
            embed.add_field(name="Duration", value=f"{request['duration']} s", inline=False)
        if request.get("instrumental"):
            embed.add_field(name="Instrumental", value="forced instrumental", inline=False)
        if state["phase"] == "pending" and state["votes"]:
            approve = sum(1 for picked in state["votes"].values() if picked == "approve")
            reject = len(state["votes"]) - approve
            embed.add_field(name="Votes", value=f"Approve {approve} — Reject {reject}", inline=False)
        requester = request.get("requester_id")
        footer = f"Cost ${request['cost']:.2f}"
        if status is not None:
            notes = {
                "approved": "approved, generating"
                , "posted": "posted"
                , "rejected": "rejected"
                , "expired": "expired without a decision"
                , "replaced": "replaced by a newer request"
                , "failed": "failed"
            }
            footer += f" — {notes.get(status, status)}"
        if requester is not None:
            footer += f" — requested by <@{requester}>"
        embed.set_footer(text=footer)
        return embed

    async def _save(self) -> None:
        """Persist the live states to Config. A terminal state never persists."""
        if self.config is None:
            return
        data = {}
        for session_id, state in self.active.items():
            data[str(session_id)] = {
                "request": state["request"]
                , "phase": state["phase"]
                , "votes": {str(user_id): picked for user_id, picked in state["votes"].items()}
                , "names": {str(user_id): name for user_id, name in state.get("names", {}).items()}
                , "queue": list(state["queue"]) if state.get("queue") else None
                , "channel_id": state["channel"].id
                , "message_id": state["message_id"]
                , "created": state["created"]
            }
        await self.config.music.set(data)

    async def restore(self, add_view) -> None:
        """Restore the persisted requests after a reload: views, idle clocks, queued generations."""
        if self.config is None:
            return
        data = await self.config.music()
        dead = []
        for key, saved in data.items():
            try:
                session_id = int(key)
            except (TypeError, ValueError):
                continue
            if session_id in self.active:
                continue
            channel = await self.channel_getter(saved.get("channel_id")) if self.channel_getter else None
            if channel is None:
                dead.append(key)
                continue
            state = {
                "request": saved["request"]
                , "phase": saved["phase"]
                , "votes": {int(user_id): picked for user_id, picked in saved.get("votes", {}).items()}
                , "names": {int(user_id): name for user_id, name in saved.get("names", {}).items()}
                , "queue": tuple(saved["queue"]) if saved.get("queue") else None
                , "channel": channel
                , "message_id": saved.get("message_id")
                , "created": saved.get("created") or time.time()
                , "idle_task": None
                , "run_task": None
                , "lock": asyncio.Lock()
            }
            if state["phase"] == "pending":
                self.active[session_id] = state
                if state["message_id"] is not None:
                    # The custom ids match the buttons of the old view, so the
                    # clicks of the old message dispatch to this new view.
                    add_view(SongView(self, session_id), message_id=state["message_id"])
                self._restart_idle(session_id, state)
            elif state["phase"] == "queued" and state["queue"] is not None and self.retriever is not None:
                # The generation was running at the reload: the queue id
                # resumes the retrieve polls where they stopped.
                self.active[session_id] = state
                state["run_task"] = asyncio.create_task(self._finish(session_id, state))
            # Any other phase is terminal or unresumable: it stays dropped.
        if dead:
            async with self.config.music() as music:
                for key in dead:
                    music.pop(key, None)

    async def _participants(self, session_id: int, state: dict) -> frozenset:
        """The active users of a guild conversation. Empty in a direct message or without a getter."""
        if getattr(state["channel"], "guild", None) is None:
            return frozenset()
        if self.participants_getter is None:
            return frozenset()
        return frozenset(await self.participants_getter(session_id) or ())

    async def _fire(self, session_id: int, text: str | None, state: dict) -> None:
        """Wake the agent with a harness text: a song event without a user message."""
        if not text:
            return
        if self.on_event is None:
            log.warning("The song trigger of session %s was skipped: no callback.", session_id)
            return
        try:
            await self.on_event(session_id, state["channel"], text)
        except Exception:
            log.exception("The song agent trigger failed for session %s.", session_id)

    def _restart_idle(self, session_id: int, state: dict) -> None:
        task = state["idle_task"]
        if task is not None:
            task.cancel()
        state["idle_task"] = asyncio.create_task(self._idle_watch(session_id, state))

    async def _idle_watch(self, session_id: int, state: dict) -> None:
        try:
            await asyncio.sleep(MUSIC_REQUEST_IDLE)
        except asyncio.CancelledError:
            return
        try:
            await self._expire(session_id, state)
        except Exception:
            log.exception("The song request expiry failed for session %s.", session_id)

    async def _expire(self, session_id: int, state: dict, note: str = "expired", wake: bool = True) -> None:
        """Close an open vote without a decision. No cost was billed.

        wake False serves the replace path of a newer request: the tool
        result of the new request already speaks."""
        async with state["lock"]:
            if state["phase"] != "pending" or self.active.get(session_id) is not state:
                return
            state["phase"] = "finished"
            task = state["idle_task"]
            state["idle_task"] = None
            if task is not None and task is not asyncio.current_task():
                task.cancel()
            self.active.pop(session_id, None)
            await self.discord_call(
                lambda: state["channel"].get_partial_message(state["message_id"]).edit(
                    embed=self._embed(state, note), view=None
                )
                , "The song request close"
            )
        await self._save()
        if wake:
            await self._fire(session_id, f"The song request {state['request']['preset']!r} {note} without a decision. Nothing was billed.", state)

    async def request(self, session_id: int, channel_id: int, request: dict) -> str:
        """Post one song request for approval. Return the tool result text.

        A newer request of the session closes the open one as replaced: one
        request runs per conversation."""
        channel = await self.channel_getter(channel_id) if self.channel_getter else None
        if channel is None:
            return "Error: the current channel is unknown."
        open_state = self.active.get(session_id)
        if open_state is not None and open_state["phase"] == "pending":
            await self._expire(session_id, open_state, note="replaced", wake=False)
        state = {
            "request": request
            , "phase": "pending"
            , "votes": {}
            , "channel": channel
            , "message_id": None
            , "queue": None
            , "created": time.time()
            , "idle_task": None
            , "run_task": None
            , "lock": asyncio.Lock()
        }
        view = SongView(self, session_id)
        message = await self.discord_call(lambda: channel.send(embed=self._embed(state), view=view), "The song request send")
        if message is None:
            return "Error: the song request message could not be sent."
        state["message_id"] = message.id
        self.active[session_id] = state
        self._restart_idle(session_id, state)
        await self._save()
        return (
            f"The song request {request['preset']!r} has been posted for approval at ${request['cost']:.2f}. "
            "The outcome arrives with the next messages."
        )

    async def vote(self, session_id: int, interaction: discord.Interaction, choice: str) -> None:
        """Record one click and answer the interaction.

        The defer acknowledges the click at once: the lock can wait behind a
        close with retries, and the 3-second window of the initial response
        is shorter than a retry. The edit then runs as the original
        response, valid for 15 minutes."""
        state = self.active.get(session_id)
        if state is None:
            await interaction.response.send_message("This song request has expired.", ephemeral=True)
            return
        try:
            await interaction.response.defer()
        except discord.NotFound:
            return
        async with state["lock"]:
            if state["phase"] != "pending":
                await interaction.followup.send("This song request has expired.", ephemeral=True)
                return
            user_id = interaction.user.id
            state.setdefault("names", {})[user_id] = interaction.user.display_name
            # The last click of a user is the vote.
            state["votes"][user_id] = choice
            participants = await self._participants(session_id, state)
            # A click is activity: every voter counts as an active user.
            participants = frozenset(participants | state["votes"].keys())
            single = getattr(state["channel"], "guild", None) is None or len(participants) <= 1
            approve = sum(1 for picked in state["votes"].values() if picked == "approve")
            reject = len(state["votes"]) - approve
            approved = choice == "approve" if single else approve * 5 >= len(participants) * 3
            rejected = choice == "reject" if single else reject * 5 >= len(participants) * 3
            if not approved and not rejected:
                self._restart_idle(session_id, state)
                try:
                    await interaction.edit_original_response(embed=self._embed(state))
                except discord.HTTPException:
                    pass
                await self._save()
                return
            task = state["idle_task"]
            state["idle_task"] = None
            if task is not None:
                task.cancel()
            if approved:
                if self.queuer is None or self.retriever is None:
                    state["phase"] = "finished"
                    self.active.pop(session_id, None)
                    try:
                        await interaction.edit_original_response(embed=self._embed(state, "failed"), view=None)
                    except discord.HTTPException:
                        pass
                    await self._save()
                    await self._fire(session_id, f"The song request {state['request']['preset']!r} was approved, but this cog wires no generation flow.", state)
                    return
                state["phase"] = "approved"
                try:
                    await interaction.edit_original_response(embed=self._embed(state, "approved"), view=None)
                except discord.HTTPException:
                    # A dead interaction loses only the message update: the vote stands.
                    pass
                state["run_task"] = asyncio.create_task(self._run(session_id, state))
                await self._save()
                return
            # The vote rejected the request.
            state["phase"] = "finished"
            self.active.pop(session_id, None)
            try:
                await interaction.edit_original_response(embed=self._embed(state, "rejected"), view=None)
            except discord.HTTPException:
                pass
            await self._save()
            await self._fire(session_id, f"The song request {state['request']['preset']!r} was rejected by the vote. Nothing was billed.", state)

    async def _run(self, session_id: int, state: dict) -> None:
        """The approved generation: the queue call first, then the shared tail."""
        request = state["request"]
        try:
            model, queue_id = await self.queuer(request["body"])
        except ChatError as e:
            await self._fail(session_id, state, str(e))
            return
        except Exception as e:
            log.exception("The song queue call failed for session %s.", session_id)
            await self._fail(session_id, state, f"{type(e).__name__}: {e}")
            return
        async with state["lock"]:
            state["queue"] = (model, queue_id)
            state["phase"] = "queued"
        await self._save()
        await self._finish(session_id, state)

    async def _finish(self, session_id: int, state: dict) -> None:
        """The tail of a generation: the retrieve polls, the audio post, the outcome.

        The resumed generation of a reload enters here with its queue id."""
        request = state["request"]
        try:
            name, data = await self.retriever(*state["queue"])
            result = await post_file(state["channel"], data, name)
        except ChatError as e:
            await self._fail(session_id, state, str(e))
            return
        except Exception as e:
            log.exception("The song generation failed for session %s.", session_id)
            await self._fail(session_id, state, f"{type(e).__name__}: {e}")
            return
        if result.startswith("Error:"):
            await self._fail(session_id, state, result)
            return
        async with state["lock"]:
            state["phase"] = "finished"
            self.active.pop(session_id, None)
            await self.discord_call(
                lambda: state["channel"].get_partial_message(state["message_id"]).edit(embed=self._embed(state, "posted"), view=None)
                , "The song request close"
            )
        await self._save()
        await self._fire(
            session_id
            , f"The song request {request['preset']!r} was approved and the song has been posted to the conversation. {result}"
            , state
        )

    async def _fail(self, session_id: int, state: dict, error: str) -> None:
        """Close a failed generation: the embed reports the failure, the agent learns it."""
        async with state["lock"]:
            state["phase"] = "finished"
            self.active.pop(session_id, None)
            await self.discord_call(
                lambda: state["channel"].get_partial_message(state["message_id"]).edit(embed=self._embed(state, "failed"), view=None)
                , "The song request close"
            )
        await self._save()
        await self._fire(session_id, f"The song request {state['request']['preset']!r} was approved but the generation failed: {error}", state)

    async def drop_user(self, user_id: int) -> None:
        """Drop the votes of a user and the request of a direct message with them (EUD)."""
        state = self.active.pop(user_id, None)
        if state is not None:
            for key in ("idle_task", "run_task"):
                task = state.get(key)
                if task is not None:
                    task.cancel()
        for open_state in list(self.active.values()):
            async with open_state["lock"]:
                open_state.get("names", {}).pop(user_id, None)
                if open_state["votes"].pop(user_id, None) is None:
                    continue
                if open_state["phase"] == "pending" and open_state["message_id"] is not None:
                    await self.discord_call(
                        lambda open_state=open_state: open_state["channel"].get_partial_message(open_state["message_id"]).edit(embed=self._embed(open_state))
                        , "The song request update"
                    )
        await self._save()

    def close(self) -> None:
        """Cancel the watch tasks and drop the RAM state. The persisted states restore on the next load."""
        for state in self.active.values():
            for key in ("idle_task", "run_task"):
                task = state.get(key)
                if task is not None:
                    task.cancel()
        self.active.clear()
