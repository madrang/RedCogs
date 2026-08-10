from .memory import MEMORY_MAX_CHARS, Memory

# Agent-facing scope name -> internal Memory scope.
SCOPE_ALIASES = {
    "server": "guild",
    "channel": "channel",
    "user": "user",
}


class HarnessTools:
    """Tools the harness itself provides to the agent, beside the MCP tools.

    Each tool maps to a `_tool_<name>` method. Memory tools resolve the
    scope from the ids of the conversation the agent is answering, so the
    agent can read and update the memory of the current server, channel,
    or user at any time.
    """

    def __init__(self, memory: Memory):
        self.memory = memory

    def tools(self) -> list:
        """The OpenAI function schemas of the harness tools."""
        scope_property = {
            "type": "string"
            , "enum": list(SCOPE_ALIASES)
            , "description": "The memory scope."
        }
        return [
            {
                "type": "function"
                , "function": {
                    "name": "memory_read"
                    , "description": (
                        "Read the long-term memory of one scope: server (facts shared by the whole Discord server), "
                        "channel (facts of the current channel), or user (facts about the person talking to you)."
                    )
                    , "parameters": {
                        "type": "object"
                        , "properties": {"scope": scope_property}
                        , "required": ["scope"]
                    }
                }
            }
            , {
                "type": "function"
                , "function": {
                    "name": "memory_write"
                    , "description": (
                        "Replace the long-term memory of one scope: server, channel, or user. Read the "
                        "scope first and merge when you want to keep the old content. An empty content "
                        f"erases the scope. Content over {MEMORY_MAX_CHARS} characters is truncated."
                    )
                    , "parameters": {
                        "type": "object"
                        , "properties": {
                            "scope": scope_property
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
                        "Add text at the end of the long-term memory of one scope: server, channel, or "
                        "user. Use it for one new fact. The result warns when the scope is full and part "
                        "of the content was dropped."
                    )
                    , "parameters": {
                        "type": "object"
                        , "properties": {
                            "scope": scope_property
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

    async def run(self, name: str, arguments: dict, *, guild_id, channel_id, user_id) -> str:
        """Run one harness tool and return its output as text."""
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return f"Error: unknown harness tool {name}"
        return await handler(arguments, guild_id=guild_id, channel_id=channel_id, user_id=user_id)

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

    async def _tool_memory_read(self, arguments: dict, *, guild_id, channel_id, user_id) -> str:
        scope, scope_id, label, error = self._scope_ids(arguments.get("scope", ""), guild_id, channel_id, user_id)
        if error:
            return error
        text = await self.memory.read(scope, scope_id)
        if not text:
            return f"(no memory stored for the {label.lower()} scope)"
        return text

    async def _tool_memory_write(self, arguments: dict, *, guild_id, channel_id, user_id) -> str:
        scope, scope_id, label, error = self._scope_ids(arguments.get("scope", ""), guild_id, channel_id, user_id)
        if error:
            return error
        content = arguments.get("content")
        if not isinstance(content, str):
            return "Error: the content must be a string."
        stored = await self.memory.store(scope, scope_id, content)
        if not stored:
            return f"The {label.lower()} memory has been erased."
        if len(stored) < len(content):
            return (
                f"Warning: the content was truncated from {len(content)} to {len(stored)} characters "
                f"(Config storage limit). The {label.lower()} memory now ends mid-text. "
                "Read it and rewrite it shorter."
            )
        return f"The {label.lower()} memory has been updated ({len(stored)} characters)."

    async def _tool_memory_append(self, arguments: dict, *, guild_id, channel_id, user_id) -> str:
        scope, scope_id, label, error = self._scope_ids(arguments.get("scope", ""), guild_id, channel_id, user_id)
        if error:
            return error
        content = arguments.get("content")
        if not isinstance(content, str) or not content.strip():
            return "Error: the content must be a non-empty string."
        current = await self.memory.read(scope, scope_id)
        combined = f"{current}\n{content}" if current else content
        stored = await self.memory.store(scope, scope_id, combined)
        if len(stored) < len(combined):
            dropped = len(combined) - len(stored)
            return (
                f"Warning: the {label.lower()} memory is full. Only part of the content was added: "
                f"the last {dropped} characters were dropped (Config storage limit). "
                "Read the scope and rewrite it shorter."
            )
        return f"The {label.lower()} memory now holds {len(stored)} characters."
