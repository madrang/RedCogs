"""The memory tools: read, write, and append the Config memory scopes."""

import re

import discord

from ..history import SUMMARY_MAX_CHARS
from ..memory import MEMORY_MAX_CHARS, Memory

# Agent-facing scope name -> internal Memory scope.
SCOPE_ALIASES = {
    "server": "guild",
    "channel": "channel",
    "user": "user",
}


class MemoryTools:
    """The memory_read, memory_write, and memory_append tools and the scope resolution."""

    def memory_tools(self) -> list:
        """The OpenAI function schemas of the memory tools."""
        scope_property = {
            "type": "string"
            , "enum": list(SCOPE_ALIASES)
            , "description": "The scope to target. \"server\" is not available in direct messages."
        }
        target_property = {
            "type": "string"
            , "description": (
                "Leave blank to use the active channel or user. "
                "Otherwise give a channel or user of this server: a Discord id, a mention (<@id> or <#id>), or an exact name. "
                "A user name matches the server display name, the global name, or the username. "
                "A channel name works with or without the #. Ignored for the server scope."
            )
        }
        kind_property = {
            "type": "string"
            , "enum": ["memory", "summary"]
            , "description": "Optionally select the summary of the scope instead of the memory. Default: memory."
        }
        return [
            {
                "type": "function"
                , "function": {
                    "name": "memory_read"
                    , "description": (
                        "Read the memory of one scope. "
                        "Set kind to summary to read the summary of the scope. "
                        "The optional target selects another channel or user of this server."
                    )
                    , "parameters": {
                        "type": "object"
                        , "properties": {
                            "scope": scope_property
                            , "target": target_property
                            , "kind": kind_property
                        }
                        , "required": ["scope"]
                    }
                }
            }
            , {
                "type": "function"
                , "function": {
                    "name": "memory_write"
                    , "description": (
                        "Replace the full text of one memory scope. "
                        "To keep the old content, read the scope first and merge it. "
                        f"An empty content erases the scope. The harness truncates content over {MEMORY_MAX_CHARS} characters. "
                        "Set kind to summary to replace the summary of the scope. "
                        "The optional target selects another channel or user of this server."
                    )
                    , "parameters": {
                        "type": "object"
                        , "properties": {
                            "scope": scope_property
                            , "target": target_property
                            , "kind": kind_property
                            , "content": {
                                "type": "string"
                                , "description": "The new memory text."
                            }
                        }
                        , "required": ["scope", "content"]
                    }
                }
            }
            , {
                "type": "function"
                , "function": {
                    "name": "memory_append"
                    , "description": (
                        "Add text at the end of the long-term memory of one scope. "
                        "The result warns when the scope is full and a part of the content did not fit. "
                        "Set kind to summary to add to the summary of the scope. "
                        "The optional target selects another channel or user of this server."
                    )
                    , "parameters": {
                        "type": "object"
                        , "properties": {
                            "scope": scope_property
                            , "target": target_property
                            , "kind": kind_property
                            , "content": {
                                "type": "string"
                                , "description": "The text to add."
                            }
                        }
                        , "required": ["scope", "content"]
                    }
                }
            }
        ]

    def _scope_ids(self, scope: str, guild_id, channel_id, user_id):
        """Resolve an agent-facing scope name to (scope, scope_id, label), or an error text."""
        internal = SCOPE_ALIASES.get(scope)
        if internal is None:
            return None, None, None, f"Error: unknown scope {scope!r}. Use one of: {', '.join(SCOPE_ALIASES)}."
        scope_id = {
            "guild": guild_id
            , "channel": channel_id
            , "user": user_id
        }[internal]
        if scope_id is None:
            return None, None, None, "Error: there is no server in a direct message."
        return internal, scope_id, Memory.SCOPES[internal][1], None

    def _resolve_target(self, internal: str, target, *, guild_id, channel_id, user_id):
        """Resolve the optional target of a memory tool to (scope_id, name, error text).

        A blank target keeps the active conversation. An id or a mention
        (<@id>, <#id>) resolves directly. A name resolves against the
        members or channels of the current server: the Discord API has no
        user search by name outside a guild. Every target is checked
        against the current server, so a call cannot reach the memory of
        another server. name is None for the active conversation.
        """
        current = {"guild": guild_id, "channel": channel_id, "user": user_id}
        text = str(target or "").strip()
        if not text:
            return current[internal], None, None
        if internal == "guild":
            return None, None, "Error: the server scope is always the current server. Leave the target blank."
        guild = self.guild_getter(guild_id) if self.guild_getter and guild_id is not None else None
        if guild is None:
            return None, None, "Error: a target works only inside a server. In a direct message, leave the target blank."
        mention = re.fullmatch(r"<[@#][!&]?(\d+)>", text)
        if mention:
            text = mention.group(1)
        kind = "channel" if internal == "channel" else "member"
        if text.isdigit():
            wanted = int(text)
            found = guild.get_channel(wanted) if internal == "channel" else guild.get_member(wanted)
            if found is None:
                return None, None, f"Error: no {kind} with the id {wanted} in this server."
            return wanted, self._target_name(internal, found), None
        if internal == "channel":
            name = text.removeprefix("#")
            matches = [
                channel for channel in guild.channels
                if channel.name == name and isinstance(channel, discord.abc.Messageable)
            ]
        else:
            matches = [
                member for member in guild.members
                if text == member.name or text == member.display_name
                or (member.global_name and text == member.global_name)
            ]
        if not matches:
            return None, None, f"Error: no {kind} named {text!r} in this server. Use the Discord id to be sure."
        if len(matches) > 1:
            return None, None, f"Error: {len(matches)} matches for {text!r} in this server. Use the Discord id."
        found = matches[0]
        return found.id, self._target_name(internal, found), None

    @staticmethod
    def _target_name(internal: str, found) -> str:
        """The display name of a resolved target, for the confirmation text."""
        return found.name if internal == "channel" else found.display_name

    async def _tool_memory_read(self, arguments: dict, *, guild_id, channel_id, user_id) -> str:
        scope, scope_id, label, error = self._scope_ids(arguments.get("scope", ""), guild_id, channel_id, user_id)
        if error:
            return error
        scope_id, name, error = self._resolve_target(scope, arguments.get("target"), guild_id=guild_id, channel_id=channel_id, user_id=user_id)
        if error:
            return error
        target = f" for {name}" if name else ""
        if arguments.get("kind") == "summary":
            text = await self.memory.read_summary(scope, scope_id)
            if not text:
                return f"(no summary stored for the {label.lower()} scope{target})"
            return text
        text = await self.memory.read(scope, scope_id)
        if not text:
            return f"(no memory stored for the {label.lower()} scope{target})"
        return text

    async def _tool_memory_write(self, arguments: dict, *, guild_id, channel_id, user_id) -> str:
        scope, scope_id, label, error = self._scope_ids(arguments.get("scope", ""), guild_id, channel_id, user_id)
        if error:
            return error
        scope_id, name, error = self._resolve_target(scope, arguments.get("target"), guild_id=guild_id, channel_id=channel_id, user_id=user_id)
        if error:
            return error
        target = f" of {name}" if name else ""
        content = arguments.get("content")
        if not isinstance(content, str):
            return "Error: the content must be a string."
        if arguments.get("kind") == "summary":
            stored = content[:SUMMARY_MAX_CHARS]
            await self.memory.store_summary(scope, scope_id, stored)
            if not stored:
                return f"The {label.lower()} summary{target} has been erased."
            if len(stored) < len(content):
                return (
                    f"Warning: the content was truncated to {len(stored)} characters. "
                    f"The {label.lower()} summary{target} now ends mid-text. Read it and rewrite it shorter."
                )
            return f"The {label.lower()} summary{target} has been updated ({len(stored)} characters)."
        stored = await self.memory.store(scope, scope_id, content)
        if not stored:
            return f"The {label.lower()} memory{target} has been erased."
        if len(stored) < len(content):
            return (
                f"Warning: the content was truncated from {len(content)} to {len(stored)} characters "
                f"(Config storage limit). The {label.lower()} memory{target} now ends mid-text. "
                "Read it and rewrite it shorter."
            )
        return f"The {label.lower()} memory{target} has been updated ({len(stored)} characters)."

    async def _tool_memory_append(self, arguments: dict, *, guild_id, channel_id, user_id) -> str:
        scope, scope_id, label, error = self._scope_ids(arguments.get("scope", ""), guild_id, channel_id, user_id)
        if error:
            return error
        scope_id, name, error = self._resolve_target(scope, arguments.get("target"), guild_id=guild_id, channel_id=channel_id, user_id=user_id)
        if error:
            return error
        target = f" of {name}" if name else ""
        content = arguments.get("content")
        if not isinstance(content, str) or not content.strip():
            return "Error: the content must be a non-empty string."
        if arguments.get("kind") == "summary":
            current = await self.memory.read_summary(scope, scope_id)
            combined = f"{current}\n{content}" if current else content
            stored = combined[:SUMMARY_MAX_CHARS]
            await self.memory.store_summary(scope, scope_id, stored)
            if len(stored) < len(combined):
                dropped = len(combined) - len(stored)
                return (
                    f"Warning: the {label.lower()} summary{target} is full. Only part of the content was added: "
                    f"the last {dropped} characters were dropped. Read the scope and rewrite it shorter."
                )
            return f"The {label.lower()} summary{target} now holds {len(stored)} characters."
        current = await self.memory.read(scope, scope_id)
        combined = f"{current}\n{content}" if current else content
        stored = await self.memory.store(scope, scope_id, combined)
        if len(stored) < len(combined):
            dropped = len(combined) - len(stored)
            return (
                f"Warning: the {label.lower()} memory{target} is full. Only part of the content was added: "
                f"the last {dropped} characters were dropped (Config storage limit). "
                "Read the scope and rewrite it shorter."
            )
        return f"The {label.lower()} memory{target} now holds {len(stored)} characters."
