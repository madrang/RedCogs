import time

# Upper bound of one memory text. Red Config serializes values as JSON and the size
# limit depends on the backend (JSON file, PostgreSQL). 4000 characters stays far
# below every practical limit and near 1000 tokens.
MEMORY_MAX_CHARS = 4000


class Memory:
    """Long-term memory of the AgentEliza cog, stored in Red Config.

    Memory lives at three scopes: guild (server), channel, and user.
    Each scope holds a memory text plus the time of the last write.
    The cog instantiates this class with its Config, which also registers the scope defaults on the existing "agenteliza" identifier.
    """

    # Scope name -> (Config accessor, recall label).
    # Order matters: recall goes broad to specific, so the system prompt prefix stays stable.
    SCOPES = {
        "guild": ("guild_from_id", "Server"),
        "channel": ("channel_from_id", "Channel"),
        "user": ("user_from_id", "User"),
    }

    def __init__(self, config):
        self.config = config
        self.config.register_guild(
              memory=""
            , memory_updated=0
            , summary=""
        )
        self.config.register_channel(
              memory=""
            , memory_updated=0
            , summary=""
        )
        self.config.register_user(
              memory=""
            , memory_updated=0
            , summary=""
        )

    def _group(self, scope: str, scope_id: int):
        """The Config group of one scope instance."""
        accessor, _ = self.SCOPES[scope]
        return getattr(self.config, accessor)(scope_id)

    async def recall(self, guild_id: int | None, channel_id: int, user_id: int) -> list[tuple[str, str]]:
        """Return the stored memory texts as (label, text) pairs, broad to specific.

        Empty scopes are skipped. The guild scope is skipped in direct messages, where guild_id is None.
        """
        ids = {
            "guild": guild_id
          , "channel": channel_id
          , "user": user_id
        }
        entries = []
        for scope, (_, label) in self.SCOPES.items():
            scope_id = ids[scope]
            if scope_id is None:
                continue
            text = await self.read(scope, scope_id)
            if text:
                entries.append((label, text))
        return entries

    async def read(self, scope: str, scope_id: int) -> str:
        """The memory text of one scope, or an empty string."""
        return await self._group(scope, scope_id).memory()

    async def store(self, scope: str, scope_id: int, text: str) -> str:
        """Write the memory text of one scope and stamp the write time.

        Text longer than MEMORY_MAX_CHARS is truncated, so the value stays safe
        for every Config backend. An empty text clears the scope.
        Returns the stored text.
        """
        if not text:
            await self.clear(scope, scope_id)
            return ""
        text = text[:MEMORY_MAX_CHARS]
        group = self._group(scope, scope_id)
        await group.memory.set(text)
        await group.memory_updated.set(int(time.time()))
        return text

    async def clear(self, scope: str, scope_id: int) -> None:
        """Reset one scope to the registered defaults."""
        group = self._group(scope, scope_id)
        await group.memory.clear()
        await group.memory_updated.clear()

    async def last_updated(self, scope: str, scope_id: int) -> int:
        """The unix time of the last write of one scope, or 0."""
        return await self._group(scope, scope_id).memory_updated()

    async def read_summary(self, scope: str, scope_id: int) -> str:
        """The persisted conversation summary of one scope, or an empty string."""
        return await self._group(scope, scope_id).summary()

    async def store_summary(self, scope: str, scope_id: int, text: str) -> None:
        """Persist the conversation summary of one scope, capped like a memory text."""
        await self._group(scope, scope_id).summary.set(text[:MEMORY_MAX_CHARS])
