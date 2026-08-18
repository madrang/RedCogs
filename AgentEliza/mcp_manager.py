import asyncio
import contextlib
import json
import logging
import time
from contextlib import AsyncExitStack

from discord.ext import tasks

from .tools import TOOL_RESULT_MAX_CHARS

log = logging.getLogger("red.agenteliza.mcp")

try:
    import httpx
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client
    from mcp.types import TextContent
except ImportError as e:
    # Not installed and a broken install both land here: the log names the cause.
    log.warning("MCP disabled: the import failed: %s: %s", type(e).__name__, e)
    httpx = None
    Client = None
    streamable_http_client = None
    TextContent = None

MCP_IDLE_TIMEOUT = 600
# Cap of the connect and tool-list phase of a server. A hung server must
# not block a reply forever.
MCP_CONNECT_TIMEOUT = 30
# Cap of the tool arguments in the log line.
MCP_LOG_ARGS_MAX_CHARS = 500


class MCPConnection:
    """One MCP server endpoint: session, tools, and state.

    The session is persistent and lazy: it connects on first use,
    caches the tool list, and closes after MCP_IDLE_TIMEOUT idle.
    """

    def __init__(self, name: str, config):
        self.name = name
        self.config = config
        self.stack: AsyncExitStack | None = None
        self.client: Client | None = None
        self.tools: list = []
        self.last_used: float = 0.0
        self.error: str | None = None
        self.lock: asyncio.Lock = asyncio.Lock()

    def touch(self) -> None:
        self.last_used = time.monotonic()

    def is_idle(self, timeout: float) -> bool:
        return self.client is not None and time.monotonic() - self.last_used > timeout

    @property
    def state(self) -> str:
        """One-line state for the `mcp list` command."""
        if self.client is not None:
            return f"connected, {len(self.tools)} tools"
        if self.error:
            return f"error: {self.error}"
        return "idle"

    async def get_client(self):
        """Return the live client, connecting lazily."""
        if Client is None:
            self.error = "The `mcp` package is not installed."
            return None
        if self.client is not None:
            self.touch()
            return self.client
        async with self.lock:
            if self.client is not None:
                self.touch()
                return self.client
            servers = await self.config.mcp_servers()
            # Provider servers come from the manager, refreshed per reply.
            spec = servers.get(self.name) or self.extra_servers.get(self.name)
            if spec is None:
                self.error = "Not configured."
                return None
            if spec["transport"] != "http":
                # Only remote web MCP servers are supported, no local commands.
                self.error = "Only http MCP servers are supported."
                return None
            stack = AsyncExitStack()
            try:
                async with asyncio.timeout(MCP_CONNECT_TIMEOUT):
                    headers = spec.get("headers")
                    if headers and streamable_http_client is not None and httpx is not None:
                        # Headers ride a custom httpx client: the
                        # transport has no headers parameter.
                        transport = streamable_http_client(
                            spec["url"], http_client=httpx.AsyncClient(headers=headers)
                        )
                        client = await stack.enter_async_context(Client(transport))
                    else:
                        client = await stack.enter_async_context(Client(spec["url"]))
                    result = await client.list_tools()
            except Exception as e:
                with contextlib.suppress(Exception):
                    await stack.aclose()
                self.error = f"{type(e).__name__}: {e or 'timed out'}"
                return None
            self.stack = stack
            self.client = client
            self.tools = list(result.tools)
            self.touch()
            self.error = None
            return client

    async def run_tool(self, tool_name: str, arguments: dict) -> str:
        """Run one tool on this server and return its output as text."""
        client = await self.get_client()
        if client is None:
            log.info("Tool call %s.%s skipped: the server is unavailable: %s", self.name, tool_name, self.error or "unknown error")
            return f"Error: MCP server {self.name} is unavailable: {self.error or 'unknown error'}"
        args_text = json.dumps(arguments, default=str)
        if len(args_text) > MCP_LOG_ARGS_MAX_CHARS:
            args_text = args_text[:MCP_LOG_ARGS_MAX_CHARS] + "..."
        log.info("Tool call %s.%s(%s)", self.name, tool_name, args_text)
        try:
            result = await client.call_tool(tool_name, arguments)
        except Exception as e:
            # Drop the session so the next call reconnects.
            await self.close()
            log.warning("Tool call %s.%s failed: %s: %s", self.name, tool_name, type(e).__name__, e)
            return f"Error: the tool call failed: {e}"
        parts = [block.text for block in result.content if isinstance(block, TextContent)]
        text = "\n".join(parts)
        if not text and result.structured_content:
            text = json.dumps(result.structured_content, default=str)
        if len(text) > TOOL_RESULT_MAX_CHARS:
            dropped = len(text) - TOOL_RESULT_MAX_CHARS
            text = text[:TOOL_RESULT_MAX_CHARS] + f"\n[truncated: {dropped} characters dropped]"
        if result.is_error:
            error_text = text
            if len(error_text) > MCP_LOG_ARGS_MAX_CHARS:
                error_text = error_text[:MCP_LOG_ARGS_MAX_CHARS] + "..."
            log.warning("Tool call %s.%s returned an error: %s", self.name, tool_name, error_text)
            return f"Error: {text}"
        return text or "(no output)"

    async def close(self) -> None:
        """Close the session. Safe to call from any task."""
        stack, self.stack = self.stack, None
        self.client = None
        self.tools = []
        self.last_used = 0.0
        if stack is not None:
            with contextlib.suppress(Exception):
                await stack.aclose()


class MCPManager:
    """Manage the MCP connections of the AgentEliza cog.

    Holds one MCPConnection per server name. Server definitions
    come from the Config global `mcp_servers`.
    """

    def __init__(self, config):
        self.config = config
        # MCP servers of the active provider, refreshed at every gather_tools.
        self.extra_servers: dict = {}
        self.connections: dict[str, MCPConnection] = {}

    @property
    def available(self) -> bool:
        """True when the `mcp` package is installed."""
        return Client is not None

    def get_connection(self, name: str) -> MCPConnection:
        """Return the connection object for a server, creating it empty."""
        return self.connections.setdefault(name, MCPConnection(name, self.config))

    def connected_count(self) -> int:
        return sum(1 for connection in self.connections.values() if connection.client is not None)

    def tool_count(self) -> int:
        return sum(len(connection.tools) for connection in self.connections.values())

    def start(self) -> None:
        """Start the idle-session reaper."""
        self._reap_idle.start()

    async def close(self) -> None:
        """Stop the reaper and close every session."""
        self._reap_idle.cancel()
        await self.close_all()

    async def close_server(self, name: str) -> None:
        """Close and drop the connection of one server."""
        connection = self.connections.pop(name, None)
        if connection is not None:
            await connection.close()

    async def close_all(self) -> None:
        for name in list(self.connections):
            await self.close_server(name)

    async def gather_tools(self, provider=None, api_key=None):
        """Lazy-connect every server. Return (OpenAI tools, name routes, replaced harness names).

        The servers of the active provider join the Config servers. A
        provider tool named in the `replaces` map of its server takes the
        place and the name of the harness tool: the model sees one tool,
        routed to the provider server.
        """
        self.extra_servers = provider.mcp_servers(api_key) if provider is not None and api_key else {}
        servers = {**await self.config.mcp_servers(), **self.extra_servers}
        tools = []
        routes = {}
        replaced = set()
        for name, spec in servers.items():
            connection = self.get_connection(name)
            client = await connection.get_client()
            if client is None:
                continue
            replaces = spec.get("replaces") or {}
            for tool in connection.tools:
                harness_name = next((harness for harness, target in replaces.items() if target == tool.name), None)
                if harness_name is not None:
                    exposed = harness_name
                    replaced.add(harness_name)
                else:
                    exposed = f"{name}__{tool.name}"
                tools.append({
                    "type": "function"
                    , "function": {
                        "name": exposed
                        , "description": tool.description or ""
                        , "parameters": tool.input_schema or {"type": "object", "properties": {}}
                    }
                })
                routes[exposed] = (name, tool.name)
        return tools, routes, replaced

    async def run_tool(self, exposed: str, arguments: dict, routes: dict) -> str:
        """Run one exposed tool and return its output as text."""
        target = routes.get(exposed)
        if target is None:
            return f"Error: unknown tool {exposed}"
        name, tool_name = target
        connection = self.connections.get(name)
        if connection is None:
            return f"Error: MCP server {name} is not connected."
        return await connection.run_tool(tool_name, arguments)

    @tasks.loop(seconds=60)
    async def _reap_idle(self) -> None:
        for name, connection in list(self.connections.items()):
            if connection.is_idle(MCP_IDLE_TIMEOUT):
                await self.close_server(name)
