"""Interactive polls: a button view first, a native Discord poll after an idle time."""

import asyncio
import logging
import time
from datetime import timedelta

import discord

log = logging.getLogger("red.agenteliza")

# The view phase of a poll: idle this long and the vote converts to a
# native Discord poll. Same span as the provider cache TTL: the view lives
# while the conversation is hot. Only a vote resets the clock.
POLL_VIEW_IDLE = 300
# The duration of the native poll after the conversion.
POLL_DURATION_HOURS = 24
# The Discord limits of a poll.
POLL_ANSWERS_MAX = 10
POLL_ANSWER_MAX_CHARS = 55
POLL_QUESTION_MAX_CHARS = 300


class PollView(discord.ui.View):
    """The buttons of an active poll: one button per answer."""

    def __init__(self, manager: "PollManager", session_id: int, answers: list):
        super().__init__(timeout=None)
        self.manager = manager
        self.session_id = session_id
        for index, answer in enumerate(answers):
            button = discord.ui.Button(
                style=discord.ButtonStyle.primary
                , label=answer[:80]
                , custom_id=f"eliza-poll:{session_id}:{index}"
            )
            button.callback = self._make_callback(index)
            self.add_item(button)

    def _make_callback(self, index: int):
        async def callback(interaction: discord.Interaction):
            await self.manager.vote(self.session_id, interaction, index)
        return callback


class PollManager:
    """The poll of each session: the view, the votes, the idle conversion.

    The state persists in the Config global `polls`: a reload re-registers
    the live views with the bot and keeps the pending final counts. Every
    Discord failure drops the poll, never the reply of the agent.
    """

    def __init__(self, channel_getter, discord_call, config=None):
        # Async callable returning the channel of an id, cache first then the API.
        self.channel_getter = channel_getter
        # The retry wrapper of the cog for Discord API calls.
        self.discord_call = discord_call
        # The cog Config, for the persisted poll states. None: RAM only.
        self.config = config
        # Async callback (session_id, channel, harness text): a poll event
        # without a user message wakes the agent. Wired by the cog.
        self.on_event = None
        # Async callable (session_id) returning the ids of the active users
        # of the conversation, for the majority rule of a guild poll.
        self.participants_getter = None
        self.active: dict = {}

    def _counts(self, state: dict) -> list:
        """The vote count of each answer."""
        counts = [0] * len(state["answers"])
        for picked in state["votes"].values():
            for index in picked:
                counts[index] += 1
        return counts

    def _text(self, state: dict) -> str:
        """The view message: the question and the answers with their counts."""
        lines = [f"📊 **{state['question']}**"]
        counts = self._counts(state)
        for index, answer in enumerate(state["answers"]):
            lines.append(f"{index + 1}. {answer} — {counts[index]}")
        return "\n".join(lines)

    def _final_note(self, state: dict, participants: frozenset) -> str:
        """The close note of a completed poll: the vote coverage and the duration, in small text."""
        seconds = max(0, int(time.time() - state["created"]))
        if seconds < 60:
            span = "1 second" if seconds == 1 else f"{seconds} seconds"
        elif seconds < 3600:
            minutes = max(1, round(seconds / 60))
            span = "1 minute" if minutes == 1 else f"{minutes} minutes"
        else:
            hours = max(1, round(seconds / 3600))
            span = "1 hour" if hours == 1 else f"{hours} hours"
        head = f"{len(state['votes'])} of {len(participants)} active users voted. " if len(participants) > 1 else ""
        return f"\n-# {head}Completed in {span}."

    async def _save(self) -> None:
        """Persist the active states to Config."""
        if self.config is None:
            return
        data = {}
        for session_id, state in self.active.items():
            data[str(session_id)] = {
                "question": state["question"]
                , "answers": state["answers"]
                , "multiple": state["multiple"]
                , "votes": {str(user_id): sorted(picked) for user_id, picked in state["votes"].items()}
                , "state": state["state"]
                , "channel_id": state["channel"].id
                , "message_id": state["message_id"]
                , "native_id": state["native_id"]
                , "created": state["created"]
            }
        await self.config.polls.set(data)

    async def restore(self, add_view) -> None:
        """Restore the persisted polls after a reload: views, idle clocks, native polls."""
        if self.config is None:
            return
        data = await self.config.polls()
        for key, saved in data.items():
            try:
                session_id = int(key)
            except (TypeError, ValueError):
                continue
            if session_id in self.active:
                continue
            channel = await self.channel_getter(saved.get("channel_id")) if self.channel_getter else None
            if channel is None:
                # The entry stays in Config: the next _save drops it.
                continue
            state = {
                "question": saved["question"]
                , "answers": saved["answers"]
                , "multiple": saved["multiple"]
                , "votes": {int(user_id): set(picked) for user_id, picked in saved.get("votes", {}).items()}
                , "state": saved["state"]
                , "channel": channel
                , "message_id": saved.get("message_id")
                , "native_id": saved.get("native_id")
                , "created": saved.get("created") or time.time()
                , "idle_task": None
                , "lock": asyncio.Lock()
            }
            self.active[session_id] = state
            if state["state"] != "active" or state["message_id"] is None:
                continue
            # The custom ids match the buttons of the old view, so the
            # clicks of the old message dispatch to this new view.
            add_view(PollView(self, session_id, state["answers"]), message_id=state["message_id"])
            self._restart_idle(session_id, state)

    async def _participants(self, session_id: int, state: dict) -> frozenset:
        """The active users of a guild conversation. Empty in a direct message or without a getter."""
        if getattr(state["channel"], "guild", None) is None:
            return frozenset()
        if self.participants_getter is None:
            return frozenset()
        return frozenset(await self.participants_getter(session_id) or ())

    async def _fire(self, session_id: int, text: str | None, state: dict) -> None:
        """Wake the agent with a harness text: a poll event without a user message."""
        if self.on_event is None or not text:
            return
        try:
            await self.on_event(session_id, state["channel"], text)
        except Exception:
            log.exception("The poll agent trigger failed for session %s.", session_id)

    async def create(self, session_id: int, channel, question: str, answers: list, multiple: bool) -> str | None:
        """Post the view poll of a session. Return an error text, or None on success."""
        if session_id in self.active:
            return "Error: choices are already open in this conversation."
        state = {
            "question": question
            , "answers": answers
            , "multiple": multiple
            , "votes": {}
            , "state": "active"
            , "channel": channel
            , "message_id": None
            , "native_id": None
            , "created": time.time()
            , "idle_task": None
            , "lock": asyncio.Lock()
        }
        view = PollView(self, session_id, answers)
        message = await self.discord_call(lambda: channel.send(self._text(state), view=view), "The vote send")
        if message is None:
            return "Error: the vote message could not be sent."
        state["message_id"] = message.id
        self.active[session_id] = state
        self._restart_idle(session_id, state)
        await self._save()
        return None

    async def vote(self, session_id: int, interaction: discord.Interaction, index: int) -> None:
        """Record one click and answer the interaction.

        The lock keeps a late click from overtaking the idle conversion:
        the click waits, then answers expired instead of overwriting the
        conversion note.
        """
        state = self.active.get(session_id)
        if state is None:
            await interaction.response.send_message("These choices have expired.", ephemeral=True)
            return
        async with state["lock"]:
            if state["state"] != "active":
                await interaction.response.send_message("These choices have expired.", ephemeral=True)
                return
            user_id = interaction.user.id
            if state["multiple"]:
                picked = state["votes"].setdefault(user_id, set())
                if index in picked:
                    picked.discard(index)
                else:
                    picked.add(index)
                if not picked:
                    del state["votes"][user_id]
            else:
                # Single choice: the last click of a user is the vote.
                state["votes"][user_id] = {index}
            participants = await self._participants(session_id, state)
            # A click is activity: every voter counts as an active user,
            # also a voter who never talked to the bot.
            participants = frozenset(participants | state["votes"].keys())
            single = getattr(state["channel"], "guild", None) is None or len(participants) <= 1
            # Close at once when the single active user answers a
            # single-choice poll (the direct message behavior), or when
            # 60 percent of the active users or more answered.
            close_now = (single and not state["multiple"]) or (
                not single and len(state["votes"]) * 5 >= len(participants) * 3
            )
            if not close_now:
                self._restart_idle(session_id, state)
            text = self._text(state)
            if close_now:
                # The result is final: close the view at once, no idle wait.
                task = state["idle_task"]
                if task is not None:
                    task.cancel()
                    state["idle_task"] = None
                state["state"] = "closed"
                await interaction.response.edit_message(content=text + self._final_note(state, participants), view=None)
            else:
                await interaction.response.edit_message(content=text)
        await self._save()
        if close_now:
            # A button click is no user message: wake the agent with the counts.
            await self._fire(session_id, await self.status_text(session_id), state)

    def _restart_idle(self, session_id: int, state: dict) -> None:
        task = state["idle_task"]
        if task is not None:
            task.cancel()
        state["idle_task"] = asyncio.create_task(self._idle_watch(session_id, state))

    async def _idle_watch(self, session_id: int, state: dict) -> None:
        try:
            await asyncio.sleep(POLL_VIEW_IDLE)
        except asyncio.CancelledError:
            return
        await self._convert(session_id, state)

    async def _convert(self, session_id: int, state: dict) -> None:
        """End the view phase: a native poll in a guild, an expired view elsewhere."""
        channel = state["channel"]
        guild = getattr(channel, "guild", None)
        # Native polls are guild-only and need the send_polls permission.
        can_native = (
            guild is not None and guild.me is not None
            and channel.permissions_for(guild.me).send_polls
        )
        native_id = None
        if can_native:
            poll = discord.Poll(
                state["question"]
                , duration=timedelta(hours=POLL_DURATION_HOURS)
                , allow_multiselect=state["multiple"]
            )
            for answer in state["answers"]:
                poll.add_answer(answer)
            native = await self.discord_call(lambda: channel.send(poll=poll), "The vote conversion")
            if native is not None:
                native_id = native.id
        # The lock holds the state flip and the view edit together: a click
        # in flight finishes first, and its counts join the final text.
        async with state["lock"]:
            if state["state"] != "active" or self.active.get(session_id) is not state:
                return
            state["state"] = "converted" if native_id is not None else "closed"
            state["native_id"] = native_id
            if native_id is not None and self.on_event is not None:
                # The first status call after a conversion stays silent (the
                # trigger below): the native poll keeps running. The next one
                # ends it and reports the counts.
                state["fresh_native"] = True
            note = "\n(the choices continue as a Discord poll)" if native_id is not None else "\n(the choices have expired)"
            await self.discord_call(
                lambda: channel.get_partial_message(state["message_id"]).edit(content=self._text(state) + note, view=None)
                , "The vote view close"
            )
        await self._save()
        if state["state"] == "converted":
            if self.active.get(session_id) is state:
                # A majority close in between (native_vote) already reported.
                await self._fire(session_id, f"The choices on {state['question']!r} continue as a Discord poll.", state)
        else:
            await self._fire(session_id, await self.status_text(session_id), state)

    async def native_vote(self, message_id: int, user_id: int, added: bool) -> None:
        """Track one native poll vote. At the majority, end the poll and wake the agent."""
        found = None
        for session_id, state in self.active.items():
            if state["state"] == "converted" and state["native_id"] == message_id:
                found = (session_id, state)
                break
        if found is None:
            return
        session_id, state = found
        native_votes = state.setdefault("native_votes", set())
        if added:
            native_votes.add(user_id)
        else:
            native_votes.discard(user_id)
        participants = await self._participants(session_id, state)
        # The same majority rule as the view phase: the active users and
        # every voter, at 60 percent.
        participants = frozenset(participants | state["votes"].keys() | native_votes)
        if not participants or len(native_votes) * 5 < len(participants) * 3:
            return
        # The status call below owns the fresh flag from now on.
        state.pop("fresh_native", None)
        await self._fire(session_id, await self.status_text(session_id), state)

    async def status_text(self, session_id: int) -> str | None:
        """The harness status of the poll of a session, or None without a poll.

        On an active poll: the live counts. A multiple-choice poll with
        one active user closes: the reply of the single voter ends the
        choices. On a converted poll: the first call ends the native poll
        and answers the final counts. On a closed poll: the first call
        answers the final counts of the view.
        """
        state = self.active.get(session_id)
        if state is None:
            return None
        if state["state"] == "active":
            participants = await self._participants(session_id, state)
            # A voter counts as an active user, the same rule as in vote.
            participants = frozenset(participants | state["votes"].keys())
            single = getattr(state["channel"], "guild", None) is None or len(participants) <= 1
            if not (single and state["multiple"]):
                return f"Choices are open:\n{self._text(state)}"
            # One active user answers a multiple-choice poll with a reply:
            # the reply closes the choices, no idle wait.
            task = state["idle_task"]
            if task is not None:
                task.cancel()
                state["idle_task"] = None
            state["state"] = "closed"
            await self.discord_call(
                lambda: state["channel"].get_partial_message(state["message_id"]).edit(
                    content=self._text(state) + self._final_note(state, participants), view=None
                )
                , "The vote view close"
            )
        if state["state"] == "converted" and state.pop("fresh_native", False):
            # The conversion trigger already reached the agent: this status
            # call stays silent so the native poll keeps running.
            return None
        self.active.pop(session_id, None)
        await self._save()
        if state["state"] == "converted" and state["native_id"] is not None:
            message = None
            try:
                native = await state["channel"].fetch_message(state["native_id"])
                message = await native.end_poll()
            except (discord.NotFound, discord.Forbidden, discord.HTTPException):
                message = None
            poll = getattr(message, "poll", None)
            if poll is not None:
                parts = [f"{answer.poll_media.text}: {getattr(answer, 'vote_count', 0)}" for answer in poll.answers]
                return f"The choices on {state['question']!r} have expired. Final counts: {', '.join(parts)}."
            return f"The choices on {state['question']!r} have expired. The final counts are unknown."
        counts = self._counts(state)
        parts = [f"{answer}: {counts[index]}" for index, answer in enumerate(state["answers"])]
        return f"The choices on {state['question']!r} have expired. Final counts: {', '.join(parts)}."

    async def drop_user(self, user_id: int) -> None:
        """Drop the votes of a user and the poll of a direct message with them (EUD)."""
        state = self.active.pop(user_id, None)
        if state is not None and state["idle_task"] is not None:
            state["idle_task"].cancel()
        for poll in self.active.values():
            if poll["votes"].pop(user_id, None) is None:
                continue
            if poll["state"] == "active" and poll["message_id"] is not None:
                await self.discord_call(
                    lambda poll=poll: poll["channel"].get_partial_message(poll["message_id"]).edit(content=self._text(poll))
                    , "The vote count update"
                )
        await self._save()

    def close(self) -> None:
        """Cancel the idle tasks and drop the RAM state. The persisted states restore on the next load."""
        for state in self.active.values():
            task = state["idle_task"]
            if task is not None:
                task.cancel()
        self.active.clear()
