"""The harness tools: the tools the cog itself provides to the agent, beside the MCP tools."""

from dataclasses import dataclass

from ..memory import Memory
from .files import FileTools
from .history import HistoryTools
from .mcp import MCPTools
from .memory import SCOPE_ALIASES, MemoryTools
from .poll import PollTools
from .web import WebTools
from .workspace import WorkspaceTools


@dataclass
class HarnessOptions:
    """The dependencies of the harness tools, one field each. Build it over a few lines, pass it as one argument."""

    # The Config memory store, for the memory tools.
    memory: Memory = None
    # Callable returning the shared aiohttp session of the cog, for the web tools.
    session_getter: object = None
    # Callable returning the guild object of an id, for target name resolution.
    guild_getter: object = None
    # Async callable returning the channel of an id, cache first then the API, for the history tool.
    channel_getter: object = None
    # Callable returning the bot user id, for the involvement filter.
    bot_id_getter: object = None
    # Callable returning the bot token, for authorized Discord file downloads.
    bot_token_getter: object = None
    # The poll manager, for the poll tool.
    polls: object = None
    # The workspace store, for the workspace tools and send_file.
    workspace: object = None
    # The MCP manager, for the resource tools.
    mcp: object = None


class HarnessTools(MemoryTools, HistoryTools, FileTools, PollTools, WebTools, WorkspaceTools, MCPTools):
    """The set of the harness tools, one mixin per tool family.

    Each tool maps to a `_tool_<name>` method. Memory tools resolve the
    scope from the ids of the conversation the agent is answering, so the
    agent can read and update the memory of the current server, channel,
    or user at any time. The optional target parameter selects another
    user or channel of the same server, by id, mention, or name. Names
    resolve through the guild object from guild_getter: the Discord API
    has no user search by name outside a guild.
    """

    def __init__(self, options: HarnessOptions):
        self.memory = options.memory
        self.session_getter = options.session_getter
        self.guild_getter = options.guild_getter
        self.channel_getter = options.channel_getter
        self.bot_id_getter = options.bot_id_getter
        self.bot_token_getter = options.bot_token_getter
        self.polls = options.polls
        self.workspace = options.workspace
        self.mcp = options.mcp

    def tools(self) -> list:
        """The OpenAI function schemas of the harness tools, in a stable order."""
        return [*self.memory_tools(), *self.history_tools(), *self.file_tools(), *self.poll_tools(), *self.web_tools(), *self.workspace_tools(), *self.mcp_tools()]

    async def run(self, name: str, arguments: dict, *, guild_id, channel_id, user_id, is_owner: bool = False) -> str:
        """Run one harness tool and return its output as text."""
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return f"Error: unknown harness tool {name}"
        if name.startswith("memory_") and guild_id is None and not is_owner:
            # A direct message has no server boundary: a non-owner reaches
            # only their own user memory (a blank target), nothing shared.
            own_user = SCOPE_ALIASES.get(arguments.get("scope", "")) == "user" and not str(arguments.get("target") or "").strip()
            if not own_user:
                return (
                    "Error: in a direct message the memory tools reach only the user memory of this conversation. "
                    "Use scope user with no target. Every other scope answers only the bot owner."
                )
        return await handler(arguments, guild_id=guild_id, channel_id=channel_id, user_id=user_id)
