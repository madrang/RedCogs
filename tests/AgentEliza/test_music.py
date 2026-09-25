"""The song request manager: the approval vote and the generation outcome."""

import asyncio
from types import SimpleNamespace

import AgentEliza.music as music_module
from AgentEliza.llm_chat import ChatError
from AgentEliza.music import SongManager


class FakeUser:
    def __init__(self, user_id: int, name: str):
        self.id = user_id
        self.display_name = name


class FakeInteractionResponse:
    def __init__(self):
        self.deferred = False
        self.messages: list = []

    async def defer(self):
        self.deferred = True

    async def send_message(self, text, ephemeral=False):
        self.messages.append(text)


class FakeFollowup:
    def __init__(self):
        self.messages: list = []

    async def send(self, text, ephemeral=False):
        self.messages.append(text)


class FakeInteraction:
    def __init__(self, user: FakeUser):
        self.user = user
        self.response = FakeInteractionResponse()
        self.followup = FakeFollowup()
        self.edits: list = []

    async def edit_original_response(self, *, content=None, embed=None, view=None):
        self.edits.append((content, embed, view))


class FakePartialMessage:
    def __init__(self):
        self.edits: list = []

    async def edit(self, *, content=None, embed=None, view=None):
        self.edits.append((content, embed, view))


class FakeMessage:
    def __init__(self, message_id: int):
        self.id = message_id
        self.attachments = [SimpleNamespace(url=f"https://cdn.example.com/{message_id}")]


class FakeGuild:
    me = None
    filesize_limit = 26_214_400


class FakeChannel:
    def __init__(self, channel_id: int, guild=None):
        self.id = channel_id
        self.guild = guild
        self.sends: list = []
        self._next_id = 500
        self.partials: dict[int, FakePartialMessage] = {}

    async def send(self, *, content=None, embed=None, view=None, file=None):
        self.sends.append({"content": content, "embed": embed, "view": view, "file": file})
        self._next_id += 1
        return FakeMessage(self._next_id)

    def get_partial_message(self, message_id: int) -> FakePartialMessage:
        if message_id not in self.partials:
            self.partials[message_id] = FakePartialMessage()
        return self.partials[message_id]


def build_manager(channel, *, participants=(), retriever=None):
    async def channel_getter(channel_id):
        return channel

    async def discord_call(action, what):
        return await action()

    manager = SongManager(channel_getter, discord_call, None)
    fired: list = []

    async def on_event(session_id, target, text):
        fired.append(text)

    async def participants_getter(session_id):
        return set(participants)

    runs: list = []

    async def queuer(body):
        runs.append(("queue", body))
        return "sonilo-v1-1-music", "q1"

    billed: list = []

    async def cost_recorder(channel, user_id, cost):
        billed.append((channel.id, user_id, cost))

    manager.on_event = on_event
    manager.participants_getter = participants_getter
    if retriever is None:
        def retriever(model, queue_id):
            runs.append(("retrieve", model, queue_id))
            return "song.mp3", b"audio"
    if asyncio.iscoroutinefunction(retriever):
        manager.retriever = retriever
    else:
        async def wrapped(model, queue_id):
            return retriever(model, queue_id)

        manager.retriever = wrapped
    manager.queuer = queuer
    manager.cost_recorder = cost_recorder
    return manager, fired, runs, billed


async def settle() -> None:
    for _ in range(200):
        await asyncio.sleep(0.001)


def a_request() -> dict:
    return {
        "preset": "Sonilo"
        , "model": "sonilo-v1-1-music"
        , "prompt": "a happy tune"
        , "lyrics": ""
        , "duration": 90
        , "instrumental": False
        , "requester_id": 7
        , "cost": 0.26
        , "body": {"model": "sonilo-v1-1-music", "prompt": "a happy tune", "duration_seconds": 90}
    }


async def test_a_direct_message_closes_on_the_first_click_and_generates() -> None:
    channel = FakeChannel(100)
    manager, fired, runs, billed = build_manager(channel)
    answer = await manager.request(100, 100, a_request())
    assert answer.startswith("The song request 'Sonilo' has been posted for approval at $0.26.")
    # The embed carries the parameters, the cost sits in the footer.
    embed = channel.sends[0]["embed"]
    assert embed.title == "Song request"
    assert "Cost $0.26" in embed.footer.text
    assert channel.sends[0]["view"] is not None

    interaction = FakeInteraction(FakeUser(7, "Owner"))
    await manager.vote(100, interaction, "approve")
    assert interaction.edits[-1][2] is None
    await settle()
    assert ("queue", a_request()["body"]) in runs
    assert any(kind == "retrieve" for kind, *_ in runs)
    # The audio posts as a file, and the agent learns the outcome.
    assert channel.sends[-1]["file"] is not None
    assert fired and "approved and the song has been posted" in fired[0]
    assert 100 not in manager.active


async def test_a_paid_generation_records_its_quoted_price() -> None:
    channel = FakeChannel(100, guild=FakeGuild())
    manager, fired, runs, billed = build_manager(channel)
    await manager.request(100, 100, a_request())
    await manager.vote(100, FakeInteraction(FakeUser(7, "Owner")), "approve")
    await settle()
    # The quoted price of the request lands in the scope stats at the post.
    assert billed == [(100, 7, 0.26)]


async def test_a_guild_vote_closes_at_sixty_percent() -> None:
    channel = FakeChannel(100, guild=FakeGuild())
    manager, fired, runs, billed = build_manager(channel, participants={1, 2, 3, 4, 5})
    await manager.request(100, 100, a_request())
    # Two of five approve: 10 of 15 stays open, the tally edits the embed.
    first = FakeInteraction(FakeUser(1, "A"))
    await manager.vote(100, first, "approve")
    assert 100 in manager.active
    await manager.vote(100, FakeInteraction(FakeUser(2, "B")), "approve")
    assert 100 in manager.active
    # The third approval reaches 60 percent: the generation starts.
    third = FakeInteraction(FakeUser(3, "C"))
    await manager.vote(100, third, "approve")
    assert third.edits[-1][2] is None
    await settle()
    assert runs
    assert fired and "approved and the song has been posted" in fired[0]


async def test_a_reject_majority_bills_nothing() -> None:
    channel = FakeChannel(100, guild=FakeGuild())
    manager, fired, runs, billed = build_manager(channel, participants={1, 2})
    await manager.request(100, 100, a_request())
    await manager.vote(100, FakeInteraction(FakeUser(1, "A")), "reject")
    await manager.vote(100, FakeInteraction(FakeUser(2, "B")), "reject")
    await settle()
    assert runs == []
    assert fired and "was rejected by the vote" in fired[0]
    assert billed == []
    assert 100 not in manager.active


async def test_a_newer_request_replaces_the_open_one_without_a_wake() -> None:
    channel = FakeChannel(100, guild=FakeGuild())
    manager, fired, runs, billed = build_manager(channel, participants={1, 2})
    await manager.request(100, 100, a_request())
    second = dict(a_request(), preset="Lyria")
    await manager.request(100, 100, second)
    assert list(manager.active) == [100]
    assert manager.active[100]["request"]["preset"] == "Lyria"
    assert fired == []


async def test_an_idle_request_expires_unbilled(monkeypatch) -> None:
    monkeypatch.setattr(music_module, "MUSIC_REQUEST_IDLE", 0.01)
    channel = FakeChannel(100, guild=FakeGuild())
    manager, fired, runs, billed = build_manager(channel, participants={1, 2})
    await manager.request(100, 100, a_request())
    await settle()
    assert runs == []
    assert fired and "expired without a decision" in fired[0]
    assert 100 not in manager.active


async def test_a_failed_generation_reports_the_error() -> None:
    channel = FakeChannel(100)

    async def broken(model, queue_id):
        raise ChatError("http", "The provider endpoint /audio/retrieve returned an error (HTTP 500): boom")

    manager, fired, runs, billed = build_manager(channel, retriever=broken)
    await manager.request(100, 100, a_request())
    await manager.vote(100, FakeInteraction(FakeUser(7, "Owner")), "approve")
    await settle()
    assert fired and "was approved but the generation failed" in fired[0]
    assert "boom" in fired[0]
    assert billed == []


async def test_an_unpostable_song_reports_the_post_error() -> None:
    guild = FakeGuild()
    guild.filesize_limit = 4
    channel = FakeChannel(100, guild=guild)
    manager, fired, runs, billed = build_manager(channel)
    await manager.request(100, 100, a_request())
    await manager.vote(100, FakeInteraction(FakeUser(7, "Owner")), "approve")
    await settle()
    assert fired and "was approved but the generation failed" in fired[0]
    assert "over the Discord upload limit" in fired[0]
    assert billed == []
