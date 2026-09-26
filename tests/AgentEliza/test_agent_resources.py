"""The agent resources of the providers: the harness registration and the
model documents of Venice."""

from pathlib import Path

from AgentEliza.mcp_manager import HarnessResources
from AgentEliza.providers.kimi_code import KimiCodeProvider
from AgentEliza.providers.venice import VENICE_IMAGE_MODELS, VENICE_EDIT_MODELS, VeniceApiProvider
from AgentEliza.providers.venice.resources import venice_agent_resources


class ResourceProvider:
    """A provider stand-in serving one scripted resource entry."""

    def __init__(self, entries):
        self.entries = entries
        self.contexts = []

    def agent_resources(self, context=None):
        self.contexts.append(context)
        return self.entries


def entries_by_uri(context=None):
    return {entry["uri"]: entry for entry in venice_agent_resources(context)}


def test_the_venice_documents_cover_the_four_catalogs() -> None:
    entries = entries_by_uri({"ratio": 1.4})
    assert set(entries) == {"provider/models.md", "provider/images.md", "provider/edits.md", "provider/music.md"}


def test_the_chat_document_lists_the_presets_and_the_glossary() -> None:
    text = entries_by_uri()["provider/models.md"]["build"]()
    assert "- Kimi (kimi-k3) — $18.75 — long context, vision, coding, thorough answers, writing" in text
    # The quirks and the disabled state ride the line.
    assert "quirks: narrow tools, stiff prose" in text
    assert "Venice Uncensored (venice-uncensored-1-2)" in text
    assert " — disabled" in text
    # The glossary teaches every trait, positive and negative.
    assert "- loops — The model repeats the same words in loops." in text
    assert "- narrow tools — The model carries a fixed tool list only" in text


def test_the_render_documents_carry_the_live_enable_state() -> None:
    # An unknown ratio and the healthy band keep every model enabled.
    for ratio in (None, 1.4):
        text = entries_by_uri({"ratio": ratio})["provider/images.md"]["build"]()
        assert "every model stays enabled" in text
        assert text.count("— enabled") == len(VENICE_IMAGE_MODELS)
    # At the floor band the ceiling pins to the cheapest tier.
    text = entries_by_uri({"ratio": 0.9})["provider/images.md"]["build"]()
    assert "- Muse (muse-image) — $0.02 an image — disabled" in text
    assert "- Lustify (lustify-v8) — $0.01 an image — enabled" in text
    edits = entries_by_uri({"ratio": 0.9})["provider/edits.md"]["build"]()
    assert "- Muse (muse-image-edit) — $0.02 an edit — enabled" in edits
    assert "- GPT Image (gpt-image-2-5-sunburst-edit) — $0.15 an edit — disabled" in edits
    # The compositing ceiling rides the edit lines.
    assert "composites up to 6 images" in edits


def test_the_music_document_names_the_dials_of_each_model() -> None:
    text = entries_by_uri()["provider/music.md"]["build"]()
    assert "- Sonilo (sonilo-v1-1-music) — $0.25875 a song — duration 1-600 s, default 90 s" in text
    assert "MiniMax (minimax-music-v26) — $0.18 a song — lyrics to 1000 characters, force instrumental" in text


def test_a_provider_without_resources_answers_an_empty_list() -> None:
    assert KimiCodeProvider().agent_resources({"ratio": 1.4}) == []
    assert VeniceApiProvider().agent_resources({"ratio": None}) != []


async def test_the_harness_set_lists_and_serves_the_provider_documents(tmp_path) -> None:
    resources = HarnessResources(folder=tmp_path)
    provider = ResourceProvider([
        {"uri": "provider/models.md", "name": "models.md", "description": "The models.", "build": lambda: "the models"}
    ])
    answered = []

    async def getter(session_id):
        answered.append(session_id)
        return provider, {"ratio": 1.4}

    resources.provider_getter = getter
    listing = await resources.list()
    assert "- harness:///provider/models.md — models.md (text/markdown): The models." in listing
    # The read dispatches to the builder: the entries resolve per call, one
    # for the list and one for the read.
    text = await resources.read("harness:///provider/models.md", session_id=7)
    assert text.endswith("the models")
    assert provider.contexts == [{"ratio": 1.4}, {"ratio": 1.4}]
    assert answered == [None, 7]
    # A uri the provider does not serve stays a missing resource.
    assert (await resources.read("harness:///provider/nope.md")).startswith("Error: no harness resource")


async def test_the_harness_read_tolerates_a_wrong_or_missing_scheme(tmp_path) -> None:
    resources = HarnessResources(folder=tmp_path)
    provider = ResourceProvider([
        {"uri": "provider/models.md", "name": "models.md", "description": "The models.", "build": lambda: "the models"}
    ])

    async def getter(session_id):
        return provider, {}

    resources.provider_getter = getter
    (tmp_path / "notes.md").write_text("# Notes\n", encoding="utf-8")
    for asked in (
        "harness:///provider/models.md"  # the canonical form
      , "provider/models.md"  # the scheme left out
      , "/provider/models.md"  # a lone leading slash
      , "file://provider/models.md"  # a wrong scheme
      , "file:///provider/models.md"  # a wrong scheme, three slashes
      , "harness://provider/models.md"  # two slashes short
    ):
        assert (await resources.read(asked)).endswith("the models"), asked
    # A file of the folder serves the same way.
    assert "# Notes" in await resources.read("notes.md")
    # An empty ask keeps the guidance error.
    assert (await resources.read("")).startswith("Error: a harness uri starts with harness:///.")


async def test_the_harness_set_degrades_on_failures(tmp_path) -> None:
    resources = HarnessResources(folder=tmp_path)

    async def none_getter(session_id):
        return None

    resources.provider_getter = none_getter
    assert "provider/" not in await resources.list()

    async def raising_getter(session_id):
        raise RuntimeError("boom")

    resources.provider_getter = raising_getter
    assert "(no resources)" in await resources.list()

    def broken_build():
        raise RuntimeError("broken")

    async def broken_getter(session_id):
        return ResourceProvider([
            {"uri": "provider/models.md", "name": "models.md", "description": "d", "build": broken_build}
        ]), {}

    resources.provider_getter = broken_getter
    listing = await resources.list()
    assert "- harness:///provider/models.md" in listing
    answer = await resources.read("harness:///provider/models.md")
    assert answer.startswith("Error: the provider resource harness:///provider/models.md could not be built")
