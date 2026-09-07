"""The AgentEliza test doubles: scripted stand-ins for the cog seams."""


class FakeOption:
    """One Config value: an awaitable getter with an awaitable .set."""

    def __init__(self, store: dict, name: str):
        self._store = store
        self._name = name

    async def __call__(self):
        return self._store.get(self._name)

    async def set(self, value):
        self._store[self._name] = value


class FakeConfig:
    """The redbot Config stand-in: plain stores behind awaitable getters.

    An unknown attribute reads as an option of the root store.
    `guild_from_id` opens a nested store, like the Config group access."""

    def __init__(self, values: dict | None = None):
        self._values = dict(values or {})
        self._groups: dict[str, "FakeConfig"] = {}

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return FakeOption(self._values, name)

    def group(self, key: str) -> "FakeConfig":
        if key not in self._groups:
            self._groups[key] = FakeConfig()
        return self._groups[key]

    def guild_from_id(self, guild_id: int) -> "FakeConfig":
        return self.group(f"guild:{guild_id}")


class FakeMemory:
    """The Memory stand-in: one plain dict of values, one of summaries."""

    def __init__(self):
        self.values: dict[tuple[str, int], str] = {}
        self.summaries: dict[tuple[str, int], str] = {}

    async def recall(self, guild_id, channel_id, user_id) -> list:
        entries = []
        for scope, key in (("server", guild_id), ("channel", channel_id), ("user", user_id)):
            if key is not None and (scope, key) in self.values:
                entries.append((scope, self.values[(scope, key)]))
        return entries

    async def read(self, scope: str, scope_id: int) -> str:
        return self.values.get((scope, scope_id), "")

    async def read_summary(self, scope: str, scope_id: int) -> str:
        return self.summaries.get((scope, scope_id), "")


class FakeResponse:
    """One scripted HTTP answer: a status and a JSON body."""

    def __init__(self, status: int = 200, json_data=None):
        self.status = status
        self._json = json_data

    async def json(self, content_type=None):
        if isinstance(self._json, Exception):
            raise self._json
        return self._json

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


class FakeSession:
    """The aiohttp session stand-in: the answers come in call order.

    An Exception entry raises from the get call itself, the way a
    connection failure raises from the request context."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        answer = self.answers.pop(0) if self.answers else FakeResponse(status=599)
        if isinstance(answer, Exception):
            raise answer
        return answer


class FakeApi:
    """The cog stand-in of the engine: a scripted queue of chat answers.

    The answers are chat-completions bodies. An Exception entry raises
    from chat_request."""

    def __init__(self, answers, *, model="test-model", preset=None, blocked=None):
        self.answers = list(answers)
        self.requests: list[dict] = []
        self.model = model
        self.preset = preset
        self.blocked = blocked

    async def usage_blocked(self):
        return self.blocked

    async def current_preset(self):
        return self.preset

    async def context_length(self, preset):
        return None

    async def model_name(self):
        return self.model

    async def chat_request(self, api_key, payload):
        self.requests.append(payload)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def message_of(self, data):
        try:
            return data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            return None


class FakePreset:
    """The provider preset stand-in: cache facts and a native tool set."""

    cache_ttl = 300
    vision_models: set = set()

    def __init__(self, native=()):
        self._native = list(native)

    def native_tools(self):
        return self._native

    def resolve_model(self, model):
        return model

    def context_length(self, model_name):
        return None

    def extra_payload(self, session_id, model=None, nsfw=False):
        return {}

    def cost_of(self, data):
        return 0.0


class FakeHarnessTools:
    """The harness tool stand-in: a fixed tool list, scripted results.

    A result that is an Exception raises from run, like a failing tool."""

    def __init__(self, names=(), results=None):
        self.names = list(names)
        self.results = dict(results or {})
        self.calls: list[tuple[str, dict]] = []
        self.channel_getter = None

    def tools(self):
        return [
            {
                "type": "function"
                , "function": {
                    "name": name
                    , "description": f"The {name} tool."
                    , "parameters": {"type": "object", "properties": {}}
                }
            }
            for name in self.names
        ]

    async def run(self, name, arguments, *, guild_id=None, channel_id=None, user_id=None, is_owner=False):
        self.calls.append((name, dict(arguments)))
        result = self.results.get(name, "ok")
        if isinstance(result, Exception):
            raise result
        return result


class FakeMCP:
    """The MCP manager stand-in: no servers, no tools."""

    async def gather_tools(self, preset, api_key):
        return [], {}, set()

    async def run_tool(self, name, arguments, routes):
        return f"Error: the tool {name} is unknown."


class FakeScopeStats:
    """The stats stand-in: every call lands in a list, media_refusal
    answers from a field."""

    def __init__(self, refusal=None):
        self.refusal = refusal
        self.media_checks: list = []
        self.media_counts: list = []
        self.records: list[dict] = []

    async def media_refusal(self, *, guild_id, channel_id, user_id):
        self.media_checks.append((guild_id, channel_id, user_id))
        return self.refusal

    async def count_media(self, *, guild_id, channel_id, user_id):
        self.media_counts.append((guild_id, channel_id, user_id))

    async def record(self, *, guild_id, channel_id, user_id, usage):
        self.records.append(usage)


class FakeCompactor:
    """The compactor stand-in: a compaction never fires."""

    async def compact(self, session_id, session, api_key, preset):
        return None


class FakeBot:
    """The bot stand-in: no user, nothing in the guild or channel caches."""

    user = None

    def get_channel(self, channel_id):
        return None

    def get_guild(self, guild_id):
        return None

    async def fetch_channel(self, channel_id):
        return None
