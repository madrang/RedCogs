"""The read_history tool: the channel, thread, or direct message history."""

import re
from datetime import datetime, timezone

import discord

from .base import MESSAGE_TIME_FORMAT, _cap, attachments_text, poll_result_suffix

# Caps of the read_history tool: the raw messages scanned, and the
# qualifying messages returned. Discord has no search endpoint for bots:
# the query parameter filters the scanned window only.
HISTORY_READ_SCAN_MAX = 400
HISTORY_READ_MAX_RESULTS = 64
HISTORY_READ_DEFAULT_RESULTS = 20


def _message_text(message) -> str:
    """The display text of one message: content, poll results, attachments.

    A poll result notification has empty content: the outcome rides in its
    embed. An attachment-only message still shows its files. The backfill of
    a fresh session reads the same parts, so both views of history agree.
    """
    content = message.content.strip()
    if message.type == discord.MessageType.poll_result:
        content += poll_result_suffix(message)
    content += attachments_text([(a.filename, a.content_type, a.url) for a in message.attachments])
    return content.strip()


class HistoryTools:
    """The read_history tool and the channel resolution."""

    def history_tools(self) -> list:
        """The OpenAI function schema of the history tool."""
        return [
            {
                "type": "function"
                , "function": {
                    "name": "read_history"
                    , "description": (
                        "Read past messages of a channel, a thread, or direct messages. Oldest first."
                    )
                    , "parameters": {
                        "type": "object"
                        , "properties": {
                            "target": {
                                "type": "string"
                                , "description": (
                                    "Leave blank to use the current channel and in direct messages, leave it blank. "
                                    "Otherwise give a channel or thread of this server: a Discord id, a mention (<#id>), or an exact name, with or without the #."
                                )
                            }
                            , "query": {
                                "type": "string"
                                , "description": "Keep only the scanned messages that contain this optional text filter."
                            }
                            , "limit": {
                                "type": "integer"
                                , "description": f"The optional maximum number of messages to return, 1 to {HISTORY_READ_MAX_RESULTS}. Default: {HISTORY_READ_DEFAULT_RESULTS}."
                            }
                            , "after": {
                                "type": "string"
                                , "description": (
                                    "Optional. A UTC date or date-time, for example 2026-08-11 or 2026-08-11T14:30Z. "
                                    "With before, the messages between the two times. Alone, the messages around that time."
                                )
                            }
                            , "before": {
                                "type": "string"
                                , "description": (
                                    "Optional. A UTC date or date-time, for example 2026-08-11 or 2026-08-11T14:30Z. "
                                    "With after, the messages between the two times. Alone, the messages around that time."
                                )
                            }
                        }
                    }
                }
            }
        ]

    @staticmethod
    def _involves_bot(message, bot_id: int) -> bool:
        """True when the message belongs to the conversation with the agent.

        The same rule as the context backfill, plus the messages of the bot
        itself: its own messages, the user messages that mention it, the
        replies to it. Other bots stay out, also in a direct message. A
        direct message always qualifies.
        """
        if message.author.id == bot_id:
            return True
        if message.author.bot:
            return False
        if message.guild is None:
            return True
        if any(user.id == bot_id for user in message.mentions):
            return True
        if message.type != discord.MessageType.reply:
            return False
        resolved = message.reference.resolved if message.reference else None
        return isinstance(resolved, discord.Message) and resolved.author.id == bot_id

    async def _resolve_channel(self, target, *, guild_id, channel_id):
        """Resolve the optional target of read_history to (channel, label, error text).

        A blank target keeps the current channel or direct message. An id,
        a <#id> mention, or an exact name resolves inside the current
        server only: a direct message stays private to its own context.
        """
        text = str(target or "").strip()
        if not text:
            channel = await self.channel_getter(channel_id) if self.channel_getter else None
            if channel is None:
                return None, None, "Error: the current channel is unknown."
            label = f"#{channel.name}" if getattr(channel, "guild", None) is not None else "this direct message"
            return channel, label, None
        guild = self.guild_getter(guild_id) if self.guild_getter and guild_id is not None else None
        if guild is None:
            return None, None, "Error: a target works only inside a server. In a direct message, leave the target blank."
        mention = re.fullmatch(r"<#(\d+)>", text)
        if mention:
            text = mention.group(1)
        if text.isdigit():
            channel = guild.get_channel_or_thread(int(text))
            if channel is None:
                return None, None, f"Error: no channel or thread with the id {text} in this server."
        else:
            name = text.removeprefix("#")
            pool = [
                entry for entry in [*guild.channels, *guild.threads]
                if isinstance(entry, discord.abc.Messageable)
            ]
            matches = [entry for entry in pool if entry.name == name]
            if not matches:
                return None, None, f"Error: no channel or thread named {text!r} in this server. Use the Discord id to be sure."
            if len(matches) > 1:
                return None, None, f"Error: {len(matches)} matches for {text!r} in this server. Use the Discord id."
            channel = matches[0]
        return channel, f"#{channel.name}", None

    @staticmethod
    def _history_time(value, name: str):
        """Parse one UTC date or date-time of the history window, or an error text."""
        if value is None:
            return None, None
        if not isinstance(value, str) or not value.strip():
            return None, f"Error: the {name} time must be a string, for example 2026-08-11 or 2026-08-11T14:30Z."
        try:
            parsed = datetime.fromisoformat(value.strip())
        except ValueError:
            return None, f"Error: cannot read the {name} time {value!r}. Use a UTC date or date-time, for example 2026-08-11T14:30Z."
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed, None

    async def _tool_read_history(self, arguments: dict, *, guild_id, channel_id, user_id) -> str:
        channel, label, error = await self._resolve_channel(arguments.get("target"), guild_id=guild_id, channel_id=channel_id)
        if error:
            return error
        bot_id = self.bot_id_getter() if self.bot_id_getter else None
        if bot_id is None:
            return "Error: the bot user is not ready."
        try:
            requested = int(arguments.get("limit") or HISTORY_READ_DEFAULT_RESULTS)
            limit = max(1, min(requested, HISTORY_READ_MAX_RESULTS))
        except (TypeError, ValueError):
            requested = HISTORY_READ_DEFAULT_RESULTS
            limit = HISTORY_READ_DEFAULT_RESULTS
        after, error = self._history_time(arguments.get("after"), "after")
        if error:
            return error
        before, error = self._history_time(arguments.get("before"), "before")
        if error:
            return error
        query = arguments.get("query")
        query = query.strip().casefold() if isinstance(query, str) and query.strip() else None
        guild = getattr(channel, "guild", None)
        if guild is not None and guild.me is not None and not channel.permissions_for(guild.me).read_message_history:
            return f"Error: I do not have the permission to read the history of {label}."
        # oldest_first=True alone would scan from the channel start
        # (discord.py: reverse=True starts at OLDEST_OBJECT). The default
        # window scans the newest messages instead, reversed on output.
        bounded = after is not None and before is not None
        window = {"limit": HISTORY_READ_SCAN_MAX}
        if bounded:
            window.update(after=after, before=before, oldest_first=True)
        elif after is not None or before is not None:
            # around accepts a limit of at most 101, capped to 100 by discord.py.
            window.update(around=after if after is not None else before, oldest_first=True, limit=100)
        raw = []
        skipped = set()
        try:
            # The whole window is collected before the filter: a bounded
            # window reads oldest first, so a notice can come after the
            # message it answers. The skip set must be complete first.
            async for message in channel.history(**window):
                if message.author.id == bot_id and message.type == discord.MessageType.reply:
                    # A bot reply is a harness notice, never an agent answer.
                    # It and the message it answers stay out of the result.
                    if message.reference is not None and message.reference.message_id is not None:
                        skipped.add(message.reference.message_id)
                    continue
                raw.append(message)
        except (discord.Forbidden, discord.HTTPException) as e:
            return f"Error: the history read failed: {e}"
        messages = []
        qualifying = 0
        matched = 0
        for message in raw:
            if message.id in skipped:
                continue
            if not self._involves_bot(message, bot_id):
                continue
            content = _message_text(message)
            if not content:
                continue
            qualifying += 1
            if query and query not in content.casefold():
                continue
            matched += 1
            if len(messages) < limit:
                messages.append(message)
        if not bounded and not window.get("oldest_first"):
            messages.reverse()
        scanned = len(raw)
        if raw:
            first = min(message.created_at for message in raw)
            last = max(message.created_at for message in raw)
            span = f" from {first:{MESSAGE_TIME_FORMAT}} to {last:{MESSAGE_TIME_FORMAT}}"
        else:
            span = ""
        if not messages:
            note = f", 0 of {qualifying} matched the query" if query else ""
            return f"(no matching messages in {label}: {scanned} raw messages scanned{span}{note})"
        # The header states every internal limit the result hit: the model
        # trusts a count it can explain.
        parts = [f"{len(messages)} messages of {label}, oldest first", f"{scanned} raw messages scanned{span}"]
        if query:
            parts.append(f"{matched} of {qualifying} matched the query")
        if requested != limit:
            parts.append(f"the requested limit was capped to {limit}")
        if matched > len(messages):
            end = "earliest" if window.get("oldest_first") else "most recent"
            parts.append(f"the limit is {limit}: only the {end} {len(messages)} of {matched} are shown")
        if scanned >= window["limit"]:
            parts.append(f"the scan window of {window['limit']} raw messages was reached: older messages were not read: use a narrower time range with after and before")
        lines = ["(" + ". ".join(parts) + ")"]
        for message in messages:
            # The message text keeps the shape it was posted with, newlines included.
            lines.append(f"{message.created_at:{MESSAGE_TIME_FORMAT}} {message.author.display_name} <@{message.author.id}>: {_message_text(message)}")
        return _cap("\n".join(lines))
