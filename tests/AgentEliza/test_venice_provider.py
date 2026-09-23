"""The Venice provider over HTTP: the usage endpoint, the credit fetch, the song tool, the audio flow."""

import base64
from types import SimpleNamespace

import pytest
from aiohttp import ClientConnectionError

import AgentEliza.providers.venice.audio as audio_flow
from AgentEliza.llm_chat import ChatError
from AgentEliza.providers.venice import (
    VeniceApiProvider, VENICE_CHAT_PRESETS, VENICE_FIXED_TOOLS, VENICE_ROUTING_TRAIT_AT, _environment_tool
  , activity_pick, model_decision_request, select_preset, trait_strengths,
)
from tests.AgentEliza.fakes import FakeResponse, FakeSession

USAGE_ANSWER = {
    "data": {
        "accessPermitted": True
        , "apiTier": {"id": "paid", "isCharged": True}
        , "balances": {"USD": 1.5, "DIEM": 2, "BUNDLED_CREDITS": 1.5}
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
    # The endpoint names USD: the credits ride at 100 a dollar.
    assert "bundled credits 150 of 22,500" in text
    # The highest amount of each type stands for the whole key.
    assert "150 requests/min" in text
    assert "60 requests/min" not in text
    assert "3,000,000 tokens/min" in text


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


async def test_bundled_credits_reads_the_rate_limits_balance() -> None:
    session = FakeSession(FakeResponse(200, USAGE_ANSWER))
    balance = await VeniceApiProvider().bundled_credits(session, "test-key")
    assert balance == 150.0
    url, kwargs = session.calls[0]
    assert url == "https://api.venice.ai/api/v1/api_keys/rate_limits"
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"


async def test_bundled_credits_keeps_a_zero_balance() -> None:
    data = {"data": {"balances": {"BUNDLED_CREDITS": 0}}}
    session = FakeSession(FakeResponse(200, data))
    assert await VeniceApiProvider().bundled_credits(session, "test-key") == 0.0


async def test_bundled_credits_answers_none_on_failures() -> None:
    provider = VeniceApiProvider()
    assert await provider.bundled_credits(FakeSession(FakeResponse(500, {})), "test-key") is None
    assert await provider.bundled_credits(FakeSession(FakeResponse(200, ValueError("not json"))), "test-key") is None
    assert await provider.bundled_credits(FakeSession(ClientConnectionError("boom")), "test-key") is None
    # An answer that names no bundled balance reads as unknown.
    data = {"data": {"balances": {"USD": 1.5}}}
    assert await provider.bundled_credits(FakeSession(FakeResponse(200, data)), "test-key") is None


def test_preset_fallback_steps_onto_a_smaller_window() -> None:
    # The ladder ranks cost alone: the move from GLM 1M (a 1M window) steps
    # up to Qwen Max (262K), and the switch condenses the session at the move.
    assert VeniceApiProvider().preset_fallback("z-ai-glm-5-3") == "Qwen Max"
    # The ceiling cycles to the cheapest enabled preset.
    assert VeniceApiProvider().preset_fallback("kimi-k3") == "Gemma"


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


async def test_the_edit_tool_sends_the_2k_preset_and_no_quality() -> None:
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
    # No model rides the resolution override: every tier model renders at
    # the 2K preset, and the quality field reaches no edit model.
    assert posts[0]["resolution"] == "2K"
    assert "quality" not in posts[0]
    await tools["edit_image"]["handler"]({**ask, "model": "Nano Banana"}, engine)
    assert posts[1]["resolution"] == "2K"


async def test_the_edit_tool_composites_extra_images_through_multi_edit() -> None:
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
    entry = tools["edit_image"]
    ask = {
        "image": "https://cdn.discordapp.com/attachments/1/2/base.png"
      , "extra_images": ["https://example.com/layer.png"]
      , "prompt": "put the hat on the person"
    }
    # The blank model takes the default preset (Muse), a compositing one.
    answer = await entry["handler"](ask, engine)
    path, body = posts[0]
    # Extra images route the call to the multi-edit endpoint, whose model
    # field is modelId: the base image rides as base64 (a Discord
    # download), the extra as the foreign URL.
    assert path == "/image/multi-edit"
    assert body["modelId"] == "muse-image-edit"
    assert body["images"][0] == base64.b64encode(b"img").decode("ascii")
    assert body["images"][1] == "https://example.com/layer.png"
    assert "model" not in body and "image" not in body
    assert fetches == ["https://cdn.discordapp.com/attachments/1/2/base.png"]
    assert answer.startswith("The file muse-image-edit-")
    # One image stays on the edit endpoint, whose model field is model.
    ask.pop("extra_images")
    await entry["handler"](ask, engine)
    assert posts[1][0] == "/image/edit"
    assert posts[1][1]["model"] == "muse-image-edit"
    assert posts[1][1]["image"] == base64.b64encode(b"img").decode("ascii")


async def test_the_edit_tool_refuses_extra_images_the_model_takes_not() -> None:
    posts: list = []

    async def api_post(path, *, json_body=None, data=None, binary=False, timeout=120):
        posts.append((path, json_body))
        return b"pngdata", {"content-type": "image/png"}

    async def send_file(name, raw):
        return f"The file {name} has been sent."

    engine = SimpleNamespace(fetch_url=None, api_post=api_post, send_file=send_file, channel_nsfw=None)
    tools = {tool["name"]: tool for tool in VeniceApiProvider().native_tools()}
    ask = {"image": "https://example.com/pic.png", "prompt": "blend them", "extra_images": ["https://example.com/other.png"]}
    # Luma carries no max_images key: the model composites no extra image.
    answer = await tools["edit_image"]["handler"]({**ask, "model": "Luma"}, engine)
    assert answer.startswith("Error: the model Luma composites no extra image.")
    assert "Models that composite: Muse" in answer
    # Muse caps the input images at six: seven asked refuses.
    answer = await tools["edit_image"]["handler"]({**ask, "model": "Muse", "extra_images": [f"https://example.com/{i}.png" for i in range(6)]}, engine)
    assert answer == "Error: the model Muse takes at most 6 input images (7 asked)."
    # No call left the tool on either refusal.
    assert posts == []


def test_the_edit_model_enum_names_the_compositing_models() -> None:
    tools = {tool["name"]: tool for tool in VeniceApiProvider().native_tools()}
    model = tools["edit_image"]["parameters"]["properties"]["model"]
    description = model["description"]
    # The compositing group names every model that takes extra images, so
    # Luma (no compositing) stays out of it.
    assert "Composites extra images: Muse, Grok, Qwen, Wan, GPT Image, FireRed, Nano Banana, Flux, Seedream." in description
    assert "Uncensored: Seedream." in description


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


def test_the_music_tool_carries_the_availability_flags() -> None:
    tools = {tool["name"]: tool for tool in VeniceApiProvider().native_tools()}
    entry = tools["generate_song"]
    assert entry["media"] == "music"
    assert entry["dm_owner_only"] is True
    assert entry["guild_credit_gate"] is True


async def test_the_music_tool_prices_and_hands_off_the_request() -> None:
    quotes: list = []
    hands: list = []

    async def api_post(path, *, json_body=None, data=None, binary=False, timeout=120):
        quotes.append((path, json_body))
        return {"quote": 0.26}, {}

    async def request_song(request):
        hands.append(request)
        return "The song request 'Sonilo' has been posted for approval."

    engine = SimpleNamespace(api_post=api_post, request_song=request_song)
    tools = {tool["name"]: tool for tool in VeniceApiProvider().native_tools()}
    entry = tools["generate_song"]
    # The blank model takes the default preset (Sonilo) with its 90 s duration default.
    answer = await entry["handler"]({"prompt": "a happy tune"}, engine)
    assert answer.startswith("The song request 'Sonilo'")
    assert quotes[0] == ("/audio/quote", {"model": "sonilo-v1-1-music", "prompt": "a happy tune", "duration_seconds": 90})
    assert hands[0]["preset"] == "Sonilo"
    assert hands[0]["cost"] == 0.26
    assert hands[0]["body"] == quotes[0][1]
    # The dials ride the body only where the model takes them.
    await entry["handler"]({"prompt": "a sad song", "model": "MiniMax", "lyrics": "oh no", "instrumental": True}, engine)
    assert quotes[1][1] == {
        "model": "minimax-music-v26", "prompt": "a sad song", "lyrics_prompt": "oh no", "force_instrumental": True
    }


async def test_the_music_tool_refuses_the_dials_a_model_lacks() -> None:
    async def api_post(path, **kwargs):
        return {"quote": 0.1}, {}

    engine = SimpleNamespace(api_post=api_post, request_song=None)
    tools = {tool["name"]: tool for tool in VeniceApiProvider().native_tools()}
    entry = tools["generate_song"]
    assert (await entry["handler"]({"prompt": "x", "model": "Lyria", "lyrics": "la"}, engine)).startswith("Error: the model Lyria takes no lyrics.")
    assert (await entry["handler"]({"prompt": "x", "model": "Seed Audio", "duration_seconds": 60}, engine)).startswith("Error: the model Seed Audio takes no duration dial.")
    assert (await entry["handler"]({"prompt": "x", "model": "ACE-Step", "instrumental": True}, engine)).startswith("Error: the model ACE-Step takes no force_instrumental dial.")
    assert (await entry["handler"]({"prompt": "x", "model": "nope"}, engine)).startswith("Error: unknown song model nope.")
    assert (await entry["handler"]({"prompt": "x" * 5001, "model": "Lyria"}, engine)).startswith("Error: the prompt is over the 5000-character limit")
    assert (await entry["handler"]({"prompt": "x", "model": "Sonilo", "duration_seconds": 900}, engine)).startswith("Error: the duration_seconds of Sonilo")


async def test_the_music_tool_names_a_missing_approval_flow() -> None:
    async def api_post(path, **kwargs):
        return {"quote": 0.1}, {}

    engine = SimpleNamespace(api_post=api_post, request_song=None)
    tools = {tool["name"]: tool for tool in VeniceApiProvider().native_tools()}
    answer = await tools["generate_song"]["handler"]({"prompt": "x"}, engine)
    assert answer == "Error: the song approval is not available here."


async def test_the_music_tool_names_a_failed_quote() -> None:
    async def api_post(path, **kwargs):
        raise ChatError("http", "The provider endpoint /audio/quote returned an error (HTTP 500): boom")

    engine = SimpleNamespace(api_post=api_post, request_song=None)
    tools = {tool["name"]: tool for tool in VeniceApiProvider().native_tools()}
    answer = await tools["generate_song"]["handler"]({"prompt": "x"}, engine)
    assert answer.startswith("Error: the song quote failed:")


async def test_the_audio_flow_queues_polls_and_completes(monkeypatch) -> None:
    monkeypatch.setattr(audio_flow, "MUSIC_POLL_FIRST_SECONDS", 0)
    calls: list = []

    async def api_post(path, *, json_body=None, data=None, binary=False, timeout=120):
        calls.append((path, json_body))
        if path == "/audio/queue":
            return {"model": "sonilo-v1-1-music", "queue_id": "q1", "status": "QUEUED"}, {}
        if path == "/audio/complete":
            return {"ok": True}, {}
        if len(calls) == 2:
            # The first retrieve answers still processing.
            return {"status": "PROCESSING", "average_execution_time": 100}, {}
        return b"audiodata", {"content-type": "audio/mpeg"}

    name, data = await audio_flow.run_song(api_post, {"model": "sonilo-v1-1-music", "prompt": "x"})
    assert data == b"audiodata"
    assert name.startswith("sonilo-v1-1-music-") and name.endswith(".mp3")
    assert [path for path, _ in calls] == ["/audio/queue", "/audio/retrieve", "/audio/retrieve", "/audio/complete"]
    assert calls[1][1] == {"model": "sonilo-v1-1-music", "queue_id": "q1"}


async def test_the_audio_flow_fails_without_a_queue_id() -> None:
    async def api_post(path, **kwargs):
        return {"model": "m"}, {}

    with pytest.raises(ChatError):
        await audio_flow.queue_song(api_post, {"model": "m", "prompt": "x"})


async def test_the_retrieve_polls_stop_at_the_time_budget(monkeypatch) -> None:
    monkeypatch.setattr(audio_flow, "MUSIC_POLL_FIRST_SECONDS", 0)
    monkeypatch.setattr(audio_flow, "MUSIC_POLL_MAX_SECONDS", 0.05)
    monkeypatch.setattr(audio_flow, "VENICE_MUSIC_TIMEOUT", 0.05)

    async def api_post(path, **kwargs):
        return {"status": "PROCESSING", "average_execution_time": 100000}, {}

    with pytest.raises(ChatError) as caught:
        await audio_flow.retrieve_song(api_post, "m", "q1")
    assert "time budget" in str(caught.value)


def test_model_decision_request_asks_four_questions() -> None:
    body = model_decision_request("Madrang: fix this python bug")
    assert body["model"] == "jev-latest"
    assert body["state"] == "Madrang: fix this python bug"
    assert set(body["questions"]) == {"vision", "nsfw", "detail", "activity"}
    # The binary capabilities ask as noul, no options and no null answer.
    for trait in ("vision", "nsfw"):
        assert body["questions"][trait]["type"] == "noul"
        assert "criteria" not in body["questions"][trait]
    # The detail axis is one score question: an ordered rubric, concise end first.
    detail = body["questions"]["detail"]
    assert detail["type"] == "score"
    assert len(detail["criteria"]) == 3
    # The activity is one choice question: an option per activity, other holds no rubric.
    activity = body["questions"]["activity"]
    assert activity["type"] == "choice"
    assert set(activity["criteria"]) == {
        "coding", "reasoning", "writing", "roleplay", "storytelling", "imagination", "other"
    }
    assert activity["criteria"]["other"] is None


def test_trait_strengths_reads_and_clamps_the_probabilities() -> None:
    data = {"answers": {
        "vision": {"noul": 0.9}
      , "nsfw": {"probability": True}
      , "writing": {"value": 1.7}
      , "roleplay": {"probability": "high"}
      , "unknown question": {"probability": 0.8}
    }}
    # The readable fields parse, a boolean stays out, and the activities
    # never parse as noul judgments.
    assert trait_strengths(data) == {"vision": 0.9}
    # A boolean, a string, and a question outside the question set stay out.
    assert trait_strengths({}) == {}
    assert trait_strengths({"answers": {"vision": {"probability": None}}}) == {}


def test_activity_pick_reads_the_choice_answer() -> None:
    def pick(answer):
        return activity_pick({"answers": {"activity": answer}})
    # The live field and the drift guard both read.
    assert pick({"choice": "coding"}) == "coding"
    assert pick({"pick": "roleplay"}) == "roleplay"
    # The other option, a word outside the list, and a missing judgment name no activity.
    assert pick({"choice": "other"}) is None
    assert pick({"choice": "long context"}) is None
    assert pick({"probability": {"coding": 0.9}}) is None
    assert activity_pick({}) is None


def test_trait_strengths_names_the_activity_at_its_probability() -> None:
    data = {"answers": {"activity": {
        "choice": "coding"
      , "probability": {"coding": 0.8, "other": 0.2}
    }}}
    # The picked activity rides the strengths at its own option probability.
    assert trait_strengths(data) == {"coding": 0.8}
    # A drifted map field keeps the activity out of the pairs, the pick still filters.
    drifted = {"answers": {"activity": {"choice": "coding", "weights": {"coding": 0.7}}}}
    assert trait_strengths(drifted) == {}
    assert activity_pick(drifted) == "coding"


def test_trait_strengths_reads_the_detail_axis_as_one_side() -> None:
    def detail_answer(answer):
        return trait_strengths({"answers": {"detail": answer}})
    # Each end names its side at the distance from the rubric middle.
    assert detail_answer({"score": 0.4}) == {"concise answers": pytest.approx(0.6)}
    assert detail_answer({"score": 1.9}) == {"thorough answers": pytest.approx(0.9)}
    # A score past the rubric clamps onto its end.
    assert detail_answer({"score": 5}) == {"thorough answers": 1.0}
    assert detail_answer({"score": -1}) == {"concise answers": 1.0}
    # A score near the middle names a side under the threshold: the walk keeps the size open.
    assert detail_answer({"score": 1.27}) == {"thorough answers": 0.27}
    # One axis answers with one side, never both ends at once.
    for score in (0.0, 0.7, 1.0, 1.3, 2.0):
        sides = set(detail_answer({"score": score})) & {"concise answers", "thorough answers"}
        assert len(sides) == 1
    # The drift guard reads the value field, the unreadable answers stay out.
    assert detail_answer({"value": 0.2}) == {"concise answers": 0.8}
    assert detail_answer({"score": "low"}) == {}
    assert detail_answer({"probability": 0.9}) == {}


def test_select_preset_filters_then_scores_the_survivors() -> None:
    # No trait at the threshold: every enabled preset survives, the score
    # picks the richest cheap one.
    assert select_preset({}) == "Gemini"
    assert select_preset({"vision": VENICE_ROUTING_TRAIT_AT - 0.01}) == "Gemini"
    # A needed trait removes the presets that miss it, the score ranks the rest.
    assert select_preset({"vision": 0.9}) == "Gemini"
    assert select_preset({}, activity="coding") == "Gemini"
    # The roleplay carriers alone survive the activity, the richer one wins.
    assert select_preset({}, activity="roleplay") == "Aion"
    # A tight balance raises the cost pressure: the cheaper carrier wins.
    assert select_preset({}, activity="roleplay", ratio=1.0) == "Aion Mini"
    assert select_preset({"thorough answers": 0.9}) == "Aion"
    assert select_preset({"thorough answers": 0.9}, ratio=1.0) == "DeepSeek Pro"
    # The nsfw need dies unless the session allows it.
    assert select_preset({"nsfw": 0.9}, nsfw_allowed=True) == "Aion"
    assert select_preset({"nsfw": 0.9}) == "Gemini"
    # The picked activity filters at any strength: the choice question has
    # no threshold of its own.
    assert select_preset({}, activity="reasoning") == "Aion"
    # No enabled preset carries vision and storytelling together.
    assert select_preset({"vision": 0.9}, activity="storytelling") is None
    # The same extra count breaks to the cheaper preset.
    assert select_preset({"vision": 0.9, "reasoning": 0.9, "concise answers": 0.9}) == "Qwen Lite"


async def test_decide_model_posts_and_selects_the_preset() -> None:
    provider = VeniceApiProvider()
    answer = {"answers": {
        "vision": {"probability": 0.91}
      , "activity": {"choice": "coding", "probability": {"coding": 0.3, "other": 0.7}}
    }}
    session = FakeSession(FakeResponse(200, answer))
    # The weak option probability still filters: the choice pick carries no threshold.
    assert await provider.decide_model(session, "test-key", "Madrang: hi") == "Gemini"
    url, kwargs = session.calls[0]
    assert url == "https://api.venice.ai/api/v1/decisions"
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert kwargs["json"]["model"] == "jev-latest"
    assert kwargs["json"]["state"] == "Madrang: hi"
    # The questions: two noul, one score, one choice.
    assert set(kwargs["json"]["questions"]) == {"vision", "nsfw", "detail", "activity"}
    # The nsfw need dies unless the session allows it.
    nsfw = FakeSession(FakeResponse(200, {"answers": {"nsfw": {"probability": 0.9}}}))
    assert await provider.decide_model(nsfw, "test-key", "Madrang: hi") == "Gemini"
    allowed = FakeSession(FakeResponse(200, {"answers": {"nsfw": {"probability": 0.9}}}))
    assert await provider.decide_model(allowed, "test-key", "Madrang: hi", nsfw_allowed=True) == "Aion"


async def test_decide_model_answers_none_on_failures() -> None:
    provider = VeniceApiProvider()
    assert await provider.decide_model(FakeSession(FakeResponse(500, {"error": "beta"})), "k", "hi") is None
    assert await provider.decide_model(FakeSession(FakeResponse(200, ValueError("not json"))), "k", "hi") is None
    assert await provider.decide_model(FakeSession(ClientConnectionError("boom")), "k", "hi") is None
    # An answer with no readable judgment keeps the configured model.
    empty = FakeSession(FakeResponse(200, {"answers": {}}))
    assert await provider.decide_model(empty, "k", "hi") is None
    # A low answer on the noul questions routes like a plain conversation.
    low = FakeSession(FakeResponse(200, {"answers": {"vision": {"probability": 0.1}, "nsfw": {"probability": 0.1}}}))
    assert await provider.decide_model(low, "k", "hi") == "Gemini"


def test_tool_filter_names_the_fixed_tool_presets() -> None:
    provider = VeniceApiProvider()
    fixed = frozenset(VENICE_FIXED_TOOLS)
    # A preset name, its raw id, and its NSFW variant all resolve the filter.
    assert provider.tool_filter("Qwen Lite") == fixed
    assert provider.tool_filter("qwen") == fixed
    assert provider.tool_filter("qwen-3-8-27b") == fixed
    assert provider.tool_filter("gemma-4-uncensored") == fixed
    # The rest of the catalog takes the full tool set.
    assert provider.tool_filter("DeepSeek Lite") is None
    assert provider.tool_filter("kimi-k3") is None
    assert provider.tool_filter("no-such-model") is None


def test_the_fixed_tool_list_holds_the_small_answer_tools() -> None:
    assert set(VENICE_FIXED_TOOLS) == {
        "propose_choices", "configure_environment"
      , "generate_image", "edit_image", "remove_background", "generate_song"
    }


def test_mcp_tools_allowed_bars_the_gemini_preset() -> None:
    provider = VeniceApiProvider()
    # The preset name and the model id of Gemini bar the MCP tools.
    assert provider.mcp_tools_allowed("Gemini") is False
    assert provider.mcp_tools_allowed("gemini-3-8-flash") is False
    # Every other model takes the MCP tools.
    assert provider.mcp_tools_allowed("DeepSeek Lite") is True
    assert provider.mcp_tools_allowed("gemma-4-uncensored") is True
    assert provider.mcp_tools_allowed("no-such-model") is True
