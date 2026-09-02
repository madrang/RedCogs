import asyncio
import contextlib
import json
import logging
import time
from contextlib import AsyncExitStack
from pathlib import Path

from discord.ext import tasks

from .tools.base import TOOL_RESULT_MAX_CHARS, _cap

log = logging.getLogger("red.agenteliza.mcp")

try:
    # mcp 2 rides httpx2: the AsyncClient API is unchanged, the package
    # name is new, and classic httpx is never installed alongside it.
    import httpx2 as httpx
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

try:
    # A clean JSON-RPC error answer (no resources capability, an unknown
    # resource) keeps the session: only a broken transport drops it. The
    # import stands alone: a miss must not disable the whole MCP block.
    from mcp.shared.exceptions import MCPError
except ImportError:
    MCPError = None

MCP_IDLE_TIMEOUT = 600
# Cap of the connect and tool-list phase of a server. A hung server must
# not block a reply forever.
MCP_CONNECT_TIMEOUT = 30
# Timeouts of a custom header-carrying client. The values mirror the SDK
# default client (mcp.shared._httpx_utils, a private module): the httpx2
# default of 5 s on all operations kills a slow tool call mid-wait.
MCP_HTTP_TIMEOUT = 30
MCP_SSE_READ_TIMEOUT = 300
# Cap of the tool arguments in the log line.
MCP_LOG_ARGS_MAX_CHARS = 500
# Page cap of one resources/list sweep. A server that paginates past the cap
# ends the list with a note instead of an endless cursor chase.
MCP_RESOURCE_PAGES_MAX = 10
# The built-in resource set of the harness: the files of the cog `resources/`
# folder, served through the resource tools as a virtual server. A file in
# the folder registers itself: its relative path is the uri path under the
# harness scheme. The server name is reserved.
HARNESS_SERVER_NAME = "harness"
HARNESS_RESOURCE_SCHEME = "harness:///"
HARNESS_RESOURCES_FOLDER = Path(__file__).resolve().parent / "resources"


class HarnessResources:
    """The built-in resource set: the files of the cog `resources/` folder.

    The set rides the resource tools as a virtual server. No MCP connection
    stands behind it: the manager serves it directly.
    """

    def __init__(self, folder: Path = HARNESS_RESOURCES_FOLDER):
        self.folder = folder.resolve()

    def _resolve(self, uri: str) -> Path | None:
        """The confined folder path of one harness uri, or None."""
        if not uri.startswith(HARNESS_RESOURCE_SCHEME):
            return None
        rest = uri[len(HARNESS_RESOURCE_SCHEME):].replace("\\", "/")
        target = (self.folder / rest).resolve()
        if self.folder not in target.parents:
            # The uri path stays inside the folder, like a workspace path.
            return None
        return target

    @staticmethod
    def _mime(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".md":
            return "text/markdown"
        if suffix in (".txt", ""):
            return "text/plain"
        return "application/octet-stream"

    @staticmethod
    def _description(path: Path) -> str:
        """The first heading line of a file, as its description."""
        try:
            with path.open("r", encoding="utf-8") as file:
                for line in file:
                    text = line.strip()
                    if text.startswith("# "):
                        return text[2:].strip()
        except OSError:
            pass
        return ""

    def _scan(self) -> list:
        """The resource lines of the folder: uri, name, mime, description."""
        lines = []
        if not self.folder.is_dir():
            return lines
        for path in sorted(p for p in self.folder.rglob("*") if p.is_file()):
            uri = HARNESS_RESOURCE_SCHEME + path.relative_to(self.folder).as_posix()
            line = f"- {uri} — {path.name} ({self._mime(path)})"
            description = self._description(path)
            if description:
                line += f": {description}"
            lines.append(line)
        return lines

    async def list(self) -> str:
        """The harness section of list_resources."""
        lines = await asyncio.to_thread(self._scan)
        return f"## {HARNESS_SERVER_NAME}\n" + ("\n".join(lines) if lines else "(no resources)")

    async def read(self, uri: str) -> str:
        """One harness uri in the read_resource format."""
        if not uri.startswith(HARNESS_RESOURCE_SCHEME):
            return f"Error: a harness uri starts with {HARNESS_RESOURCE_SCHEME}."
        target = await asyncio.to_thread(self._resolve, uri)
        if target is None or not target.is_file():
            return f"Error: no harness resource at {uri}. Use list_resources to see the uris."
        try:
            text = await asyncio.to_thread(target.read_text, encoding="utf-8", errors="replace")
        except OSError as e:
            return f"Error: the read of {uri} failed: {e}."
        return _cap(f"# {uri} ({self._mime(target)})\n{text}")


class MCPConnection:
    """One MCP server endpoint: session, tools, and state.

    The session is persistent and lazy: it connects on first use,
    caches the tool list, and closes after MCP_IDLE_TIMEOUT idle.
    """

    def __init__(self, name: str, config, manager):
        self.name = name
        self.config = config
        self.manager = manager
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
            stack = AsyncExitStack()
            try:
                async with asyncio.timeout(MCP_CONNECT_TIMEOUT):
                    servers = await self.config.mcp_servers()
                    # Provider servers come from the manager, refreshed per reply.
                    spec = servers.get(self.name) or self.manager.extra_servers.get(self.name)
                    if spec is None:
                        self.error = "Not configured."
                        return None
                    if spec.get("transport") != "http":
                        # Only remote web MCP servers are supported, no local commands.
                        self.error = "Only http MCP servers are supported."
                        return None
                    headers = spec.get("headers")
                    if headers and streamable_http_client is not None and httpx is not None:
                        # Headers ride a custom httpx2 client: the
                        # transport has no headers parameter. Without the
                        # SDK timeouts a slow tool call dies as "SSE stream
                        # ended without a response".
                        transport = streamable_http_client(
                            spec["url"]
                            , http_client=httpx.AsyncClient(
                                headers=headers
                                , timeout=httpx.Timeout(MCP_HTTP_TIMEOUT, read=MCP_SSE_READ_TIMEOUT)
                                , follow_redirects=True
                            )
                        )
                        client = await stack.enter_async_context(Client(transport))
                    else:
                        client = await stack.enter_async_context(Client(spec["url"]))
                    result = await client.list_tools()
            except Exception as e:
                with contextlib.suppress(Exception):
                    await stack.aclose()
                log.warning("MCP connect %s failed: %r", self.name, e)
                # anyio wraps the transport error in an ExceptionGroup: the
                # sub-exception names the failure, the group does not.
                cause = e
                while isinstance(cause, BaseExceptionGroup) and cause.exceptions:
                    cause = cause.exceptions[0]
                self.error = f"{type(cause).__name__}: {cause or 'timed out'}"
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

    async def _resource_failure(self, call: str, error: Exception) -> str:
        """The error text of a failed resource call.

        A clean JSON-RPC error (the server offers no resources, the uri is
        unknown) leaves the session live. Any other failure drops it, like
        a tool call: the stream may be broken.
        """
        if MCPError is not None and isinstance(error, MCPError):
            log.info("Resource call %s %s answered an error: %s", self.name, call, error)
            return f"Error: the server answered {call} with an error: {error}"
        await self.close()
        log.warning("Resource call %s %s failed: %s: %s", self.name, call, type(error).__name__, error)
        return f"Error: the {call} call failed: {error}"

    async def list_resources(self) -> str:
        """resources/list of this server, as agent-facing text."""
        client = await self.get_client()
        if client is None:
            log.info("Resource list %s skipped: the server is unavailable: %s", self.name, self.error or "unknown error")
            return f"Error: MCP server {self.name} is unavailable: {self.error or 'unknown error'}"
        log.info("Resource list %s", self.name)
        resources = []
        cursor = None
        pages = 0
        try:
            while True:
                result = await client.list_resources(cursor=cursor)
                resources.extend(result.resources)
                cursor = result.next_cursor or None
                pages += 1
                if not cursor or pages >= MCP_RESOURCE_PAGES_MAX:
                    break
        except Exception as e:
            return await self._resource_failure("resources/list", e)
        if not resources:
            return "(no resources)"
        lines = []
        for resource in resources:
            line = f"- {resource.uri} — {resource.name}"
            if resource.mime_type:
                line += f" ({resource.mime_type})"
            if resource.description:
                line += f": {resource.description}"
            lines.append(line)
        if cursor:
            lines.append(f"[more resources follow: the list stopped at {MCP_RESOURCE_PAGES_MAX} pages]")
        return _cap("\n".join(lines))

    async def read_resource(self, uri: str) -> str:
        """resources/read of one uri, as agent-facing text."""
        client = await self.get_client()
        if client is None:
            log.info("Resource read %s skipped: the server is unavailable: %s", self.name, self.error or "unknown error")
            return f"Error: MCP server {self.name} is unavailable: {self.error or 'unknown error'}"
        log.info("Resource read %s %s", self.name, uri)
        try:
            result = await client.read_resource(uri)
        except Exception as e:
            return await self._resource_failure("resources/read", e)
        if not result.contents:
            return "(no content)"
        blocks = []
        for content in result.contents:
            head = f"# {content.uri}"
            if content.mime_type:
                head += f" ({content.mime_type})"
            text = getattr(content, "text", None)
            if text is not None:
                blocks.append(f"{head}\n{text or '(empty text)'}")
            else:
                # Base64 in a chat context burns tokens without usable data:
                # the type and the size carry what the agent can act on.
                blob = getattr(content, "blob", "") or ""
                blocks.append(f"{head}\n[binary content: about {int(len(blob) * 3 / 4)} bytes]")
        return _cap("\n\n".join(blocks))

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
        # The built-in resource set, served through the resource tools.
        self.harness = HarnessResources()

    @property
    def available(self) -> bool:
        """True when the `mcp` package is installed."""
        return Client is not None

    def get_connection(self, name: str) -> MCPConnection:
        """Return the connection object for a server, creating it empty."""
        return self.connections.setdefault(name, MCPConnection(name, self.config, self))

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
        servers = await self._server_map()
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

    async def _server_map(self) -> dict:
        """The merged server definitions: Config servers plus the servers of the active provider."""
        return {**await self.config.mcp_servers(), **self.extra_servers}

    @staticmethod
    def _unknown_server(server: str, servers: dict) -> str:
        """The error text for a server name outside the known set."""
        known = ", ".join(sorted([HARNESS_SERVER_NAME, *servers]))
        return f"Error: unknown MCP server {server}. Known servers: {known}."

    async def list_resources(self, server: str = "") -> str:
        """resources/list of one server or of every server, as agent-facing text.

        The built-in harness set lists first: alone when named, with the
        servers when the name is blank.
        """
        if server and server != HARNESS_SERVER_NAME:
            servers = await self._server_map()
            if server not in servers:
                return self._unknown_server(server, servers)
            connection = self.get_connection(server)
            return _cap(f"## {server}\n{await connection.list_resources()}")
        sections = []
        if not server or server == HARNESS_SERVER_NAME:
            sections.append(await self.harness.list())
        if not server:
            for name in await self._server_map():
                connection = self.get_connection(name)
                sections.append(f"## {name}\n{await connection.list_resources()}")
        return _cap("\n\n".join(sections))

    async def read_resource(self, server: str, uri: str) -> str:
        """resources/read of one uri of one server, as agent-facing text."""
        if server == HARNESS_SERVER_NAME:
            return await self.harness.read(uri)
        servers = await self._server_map()
        if server not in servers:
            return self._unknown_server(server, servers)
        connection = self.get_connection(server)
        return await connection.read_resource(uri)

    @tasks.loop(seconds=60)
    async def _reap_idle(self) -> None:
        for name, connection in list(self.connections.items()):
            if connection.is_idle(MCP_IDLE_TIMEOUT):
                await self.close_server(name)
