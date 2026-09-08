"""The Venice provider over HTTP: the usage endpoint and the credit fetch."""

import base64
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from aiohttp import ClientConnectionError

from AgentEliza.providers.venice import VeniceApiProvider, _environment_tool
from tests.AgentEliza.fakes import FakeResponse, FakeSession

USAGE_ANSWER = {
    "data": {
        "accessPermitted": True
        , "apiTier": {"id": "paid", "isCharged": True}
        , "balances": {"USD": 1.5, "DIEM": 2}
        , "rateLimits": [
            {"apiModelId": "a", "rateLimits": [{"amount": 150, "type": "RPM"}, {"amount": 3000000, "type": "TPM"}]}
          , {"apiModelId": "b", "rateLimits": [{"amount": 60, "type": "RPM"}]}
        ]
        , "nextEpochBegins": "2026-09-07T00:00:00.000Z"
    }
}


def test_parse_usage_builds_the_balance_row() -> None:
    row = VeniceApiProvider().parse_usage(USAGE_ANSWER)[0]
    assert row["name"] == "Balance"
    assert row["exhausted"] is False
    assert row["used"] is None and row["limit"] is None and row["percent"] is None
    text = row["text"]
    assert "tier paid" in text
    assert "$1.5 USD available" in text
    assert "2 Diem available" in text
    # The highest amount of each type stands for the whole key.
    assert "150 requests/min" in text
    assert "60 requests/min" not in text
    assert "3,000,000 tokens/min" in text
    assert "epoch resets 2026-09-07T00:00:00.000Z" in text


def test_parse_usage_flags_a_refused_key() -> None:
    row = VeniceApiProvider().parse_usage({"data": {"accessPermitted": False}})[0]
    assert row["exhausted"] is True


def test_parse_usage_flags_a_spent_charged_key() -> None:
    data = {"data": {
        "accessPermitted": True
        , "apiTier": {"id": "paid", "isCharged": True}
        , "balances": {"USD": 0, "DIEM": 0}
    }}
    assert VeniceApiProvider().parse_usage(data)[0]["exhausted"] is True


def test_parse_usage_degrades_on_an_empty_answer() -> None:
    row = VeniceApiProvider().parse_usage({})[0]
    assert row["text"] == "tier unknown"
    # The exhausted flag reads None here, a type drift the row carries:
    # every reader treats falsy as not exhausted.
    assert not row["exhausted"]


async def test_fetch_usage_parses_a_200_answer() -> None:
    session = FakeSession(FakeResponse(200, USAGE_ANSWER))
    rows, error = await VeniceApiProvider().fetch_usage(session, "test-key")
    assert error is None
    assert rows[0]["name"] == "Balance"
    url, kwargs = session.calls[0]
    assert url == "https://api.venice.ai/api/v1/api_keys/rate_limits"
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"


async def test_fetch_usage_reports_an_http_error() -> None:
    rows, error = await VeniceApiProvider().fetch_usage(
        FakeSession(FakeResponse(503, {"error": "unavailable"})), "test-key"
    )
    assert rows is None
    assert error == "The usage endpoint returned an error (HTTP 503)."


async def test_fetch_usage_reports_a_connection_failure() -> None:
    rows, error = await VeniceApiProvider().fetch_usage(
        FakeSession(ClientConnectionError("boom")), "test-key"
    )
    assert rows is None
    assert "The connection to the usage endpoint failed" in error


async def test_fetch_usage_degrades_on_a_malformed_body() -> None:
    rows, error = await VeniceApiProvider().fetch_usage(
        FakeSession(FakeResponse(200, ValueError("not json"))), "test-key"
    )
    assert rows is None
    assert error == "The usage endpoint returned an error (HTTP 200)."


async def test_bundled_credits_walks_the_analytics_answer() -> None:
    seed = datetime.now(timezone.utc) - timedelta(days=5)
    data = {"byDate": [{"date": seed.date().isoformat(), "USD": 5}]}
    session = FakeSession(FakeResponse(200, data))
    balance = await VeniceApiProvider().bundled_credits(session, "test-key", seed.timestamp(), 22500)
    assert balance == 22000.0
    url, kwargs = session.calls[0]
    assert url == "https://api.venice.ai/api/v1/billing/usage-analytics"
    assert kwargs["params"]["startDate"] == seed.date().isoformat()
    assert kwargs["params"]["endDate"]


async def test_bundled_credits_answers_none_on_failures() -> None:
    provider = VeniceApiProvider()
    seed = datetime.now(timezone.utc) - timedelta(days=5)
    assert await provider.bundled_credits(
        FakeSession(FakeResponse(500, {})), "test-key", seed.timestamp(), 22500
    ) is None
    assert await provider.bundled_credits(
        FakeSession(FakeResponse(200, ValueError("not json"))), "test-key", seed.timestamp(), 22500
    ) is None
    assert await provider.bundled_credits(
        FakeSession(ClientConnectionError("boom")), "test-key", seed.timestamp(), 22500
    ) is None


def test_preset_fallback_steps_onto_a_smaller_window() -> None:
    # The ladder ranks cost alone: the move from GLM 1M (a 1M window) steps
    # up to Aion (128K), and the switch condenses the session at the move.
    assert VeniceApiProvider().preset_fallback("z-ai-glm-5-3") == "Aion"
    # The ceiling cycles to the cheapest enabled preset.
    assert VeniceApiProvider().preset_fallback("kimi-k3") == "DeepSeek Lite"


async def test_the_environment_tool_names_a_failed_switch() -> None:
    entry = _environment_tool()

    async def refused(model_id):
        return "Error: the condense before the move failed."

    engine = SimpleNamespace(channel_nsfw=None, set_conversation_model=refused)
    answer = await entry["handler"]({"capabilities": ["roleplay"]}, engine)
    assert answer.startswith("Error: no environment preset provides: roleplay.")
    assert "Switch refused (the condense before it failed): Aion Mini, Aion." in answer


async def test_the_environment_tool_returns_the_failed_restore() -> None:
    entry = _environment_tool()

    async def refused(model_id):
        return "Error: the condense before the move failed."

    engine = SimpleNamespace(channel_nsfw=None, set_conversation_model=refused)
    answer = await entry["handler"]({"capabilities": ["default"]}, engine)
    assert answer == "Error: the condense before the move failed."


async def test_the_edit_tool_drops_gpt_image_to_1k() -> None:
    posts: list = []

    async def api_post(path, *, json_body=None, data=None, binary=False, timeout=120):
        posts.append(json_body)
        return b"pngdata", {"content-type": "image/png"}

    async def send_file(name, raw):
        return f"The file {name} has been sent."

    engine = SimpleNamespace(fetch_url=None, api_post=api_post, send_file=send_file, channel_nsfw=None)
    tools = {tool["name"]: tool for tool in VeniceApiProvider().native_tools()}
    ask = {"image": "https://example.com/pic.png", "prompt": "add a red dot"}
    await tools["edit_image"]["handler"]({**ask, "model": "GPT Image"}, engine)
    # The quality field is unreachable on this model, so its 2K bill is
    # the default high: it renders 1K instead.
    assert posts[0]["resolution"] == "1K"
    assert "quality" not in posts[0]
    await tools["edit_image"]["handler"]({**ask, "model": "Nano Banana"}, engine)
    # Every other tier model keeps the 2K preset.
    assert posts[1]["resolution"] == "2K"


async def test_the_background_remove_tool_passes_the_url_or_the_bytes() -> None:
    posts: list = []
    fetches: list = []

    async def api_post(path, *, json_body=None, data=None, binary=False, timeout=120):
        posts.append((path, json_body))
        return b"pngdata", {"content-type": "image/png"}

    async def fetch_url(url):
        fetches.append(url)
        return (b"img", "image/png")

    async def send_file(name, raw):
        return f"The file {name} has been sent."

    engine = SimpleNamespace(fetch_url=fetch_url, api_post=api_post, send_file=send_file, channel_nsfw=None)
    tools = {tool["name"]: tool for tool in VeniceApiProvider().native_tools()}
    entry = tools["remove_background"]
    answer = await entry["handler"]({"image": "https://example.com/pic.jpg"}, engine)
    # A foreign URL rides the body as image_url.
    assert answer.startswith("The file background-removed-")
    assert posts[0] == ("/image/background-remove", {"image_url": "https://example.com/pic.jpg"})
    await entry["handler"]({"image": "https://cdn.discordapp.com/attachments/1/2/pic.png"}, engine)
    # A Discord download rides the body as base64 in image.
    assert fetches == ["https://cdn.discordapp.com/attachments/1/2/pic.png"]
    assert posts[1][0] == "/image/background-remove"
    assert posts[1][1]["image"] == base64.b64encode(b"img").decode("ascii")
