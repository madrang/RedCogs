"""The shared provider surface: the analyze_image tool of every provider
rides the engine context (the ToolContext fields)."""

from types import SimpleNamespace

from AgentEliza.providers.kimi_api import KimiApiProvider
from AgentEliza.providers.kimi_code import KimiCodeProvider
from AgentEliza.providers.zai import ZaiApiProvider, ZaiCodeProvider


def context_with(*, vision_chat=False, fetch=None, shown=False):
    """A ToolContext stand-in: the vision calls recorded, a scripted answer,
    a scripted fetch, an image offer that answers from a field."""
    calls: list = []
    offers: list = []

    async def call_api(payload):
        calls.append(payload)
        return {"choices": [{"message": {"content": "a description"}}]}

    async def fetch_url(url):
        return fetch

    async def show_image(name, data, mime):
        offers.append((name, data, mime))
        return shown

    engine = SimpleNamespace(
        call_api=call_api, fetch_url=fetch_url, api_post=None
        , send_file=None, channel_nsfw=None, set_conversation_model=None
        , vision_chat=vision_chat, show_image=show_image
    )
    return engine, calls, offers


async def test_every_provider_answers_analyze_image_through_the_context() -> None:
    for provider, vision in (
        (KimiApiProvider(), "kimi-k3")
        , (KimiCodeProvider(), "k3-256k")
        , (ZaiCodeProvider(), "glm-5.3-flash")
        , (ZaiApiProvider(), "glm-5v-turbo")
    ):
        entry = provider.native_tools()[0]
        engine, calls, offers = context_with()
        answer = await entry["handler"]({"url": "https://example.com/pic.jpg"}, engine)
        # The one-shot vision call rides the context call_api, on the
        # provider's own vision model, the foreign URL remote.
        assert answer == "a description"
        assert len(calls) == 1
        assert calls[0]["model"] == vision
        assert calls[0]["messages"][0]["content"][0]["image_url"]["url"] == "https://example.com/pic.jpg"
        assert offers == []


async def test_the_zai_vision_model_refuses_an_unreadable_type() -> None:
    entry = ZaiCodeProvider().native_tools()[0]
    engine, calls, _ = context_with(fetch=(b"img", "image/webp"))
    answer = await entry["handler"](
        {"url": "https://cdn.discordapp.com/attachments/1/2/pic.webp"}, engine
    )
    assert answer == (
        "Error: the image type 'image/webp' is not supported. "
        "The vision model reads png and jpg only."
    )
    assert calls == []


async def test_a_vision_conversation_takes_the_provider_image_itself() -> None:
    entry = KimiApiProvider().native_tools()[0]
    engine, calls, offers = context_with(vision_chat=True, fetch=(b"img", "image/jpeg"), shown=True)
    answer = await entry["handler"](
        {"url": "https://cdn.discordapp.com/attachments/1/2/pic.jpg", "question": "what is it"}
        , engine
    )
    # The conversation model sees images: the offer joined, no second model
    # answered between.
    assert answer.startswith("The image joined this conversation")
    assert offers == [("pic.jpg", b"img", "image/jpeg")]
    assert calls == []
