"""The file posting path: the per-channel ledger the reply stream reads."""

from types import SimpleNamespace

from AgentEliza.tools.files import channel_post_count, post_file


class _Channel:
    """The direct-message stand-in: no guild, a plain send."""

    def __init__(self, channel_id: int):
        self.id = channel_id
        self.guild = None

    async def send(self, content=None, file=None):
        # One distinct id per test: the ledger is module state.
        return SimpleNamespace(
            attachments=[SimpleNamespace(url="https://example.invalid/file.txt")] if file else []
        )


class _NoAttachChannel(_Channel):
    """The guild stand-in that lost the attach permission."""

    def __init__(self, channel_id: int):
        super().__init__(channel_id)
        self.guild = SimpleNamespace(me=True, filesize_limit=26_214_400)

    def permissions_for(self, member):
        return SimpleNamespace(attach_files=False)


async def test_a_successful_post_counts_for_its_channel() -> None:
    before = channel_post_count(990001)
    result = await post_file(_Channel(990001), b"data", "a.txt")
    assert result.startswith("The file a.txt")
    assert channel_post_count(990001) == before + 1


async def test_other_channels_stay_unaffected() -> None:
    before = channel_post_count(990002)
    await post_file(_Channel(990003), b"data", "b.txt")
    assert channel_post_count(990002) == before


async def test_a_refused_post_does_not_count() -> None:
    before = channel_post_count(990004)
    result = await post_file(_NoAttachChannel(990004), b"data", "c.txt")
    assert result.startswith("Error: I do not have the permission")
    assert channel_post_count(990004) == before
