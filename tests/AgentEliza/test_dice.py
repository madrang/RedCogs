"""The roll_dice tool: schema shape, validation, the posted roll, and the result."""

import re

from AgentEliza.tools.dice import DICE_MAX_COUNT, DICE_MIN_SIDES, DiceTools, _reading


class FakeChannel:
    """The channel the roll posts to."""

    def __init__(self, failure: Exception | None = None):
        self.posts: list[str] = []
        self.failure = failure

    async def send(self, text: str) -> None:
        if self.failure is not None:
            raise self.failure
        self.posts.append(text)


def build_tools(channel: FakeChannel | None = None) -> DiceTools:
    tools = DiceTools()

    async def getter(channel_id):
        return channel

    tools.channel_getter = getter
    return tools


async def test_the_schema_names_the_tool_with_defaults() -> None:
    (schema,) = DiceTools().dice_tools()
    function = schema["function"]
    assert function["name"] == "roll_dice"
    properties = function["parameters"]["properties"]
    assert properties["sides"]["default"] == 6
    assert properties["count"]["default"] == 1
    assert "enum" not in properties["sides"]


async def test_a_bare_call_posts_one_d6() -> None:
    channel = FakeChannel()
    result = await build_tools(channel)._tool_roll_dice({}, guild_id=1, channel_id=2, user_id=3)
    assert re.fullmatch(r"🎲 d6: [1-6]", channel.posts[0])
    assert re.fullmatch(
        r"The roll has been posted to the conversation: d6: [1-6]\. The roll reads .+\. Do not repeat the roll line in your answer\."
        , result
    )


def test_the_reading_ladder() -> None:
    assert _reading([1], 6) == "a critical failure: the worst possible roll"
    assert _reading([6], 6) == "a perfect success: the best possible roll"
    assert _reading([1, 1, 1], 6) == "a critical failure: the worst possible roll"
    assert _reading([6, 6, 6], 6) == "a perfect success: the best possible roll"
    assert _reading([1], 20) == "a critical failure: the worst possible roll"
    assert _reading([20], 20) == "a perfect success: the best possible roll"
    assert _reading([4], 6) == "a success with a cost"
    assert _reading([5], 6) == "a success"
    assert _reading([19], 20) == "a strong success"
    assert _reading([10], 20) == "a near miss with a small gain"
    assert _reading([2], 6) == "a failure"


async def test_odd_shapes_roll_their_range() -> None:
    channel = FakeChannel()
    result = await build_tools(channel)._tool_roll_dice({"sides": 7}, guild_id=1, channel_id=2, user_id=3)
    assert re.fullmatch(r"🎲 d7: [1-7]", channel.posts[0])
    assert result.startswith("The roll has been posted to the conversation: d7:")


async def test_count_rolls_each_die_and_totals() -> None:
    channel = FakeChannel()
    result = await build_tools(channel)._tool_roll_dice({"sides": 100, "count": 3}, guild_id=1, channel_id=2, user_id=3)
    posted = channel.posts[0]
    match = re.fullmatch(r"🎲 3d100: (\d+), (\d+), (\d+) = (\d+)", posted)
    assert match is not None
    dice = [int(value) for value in match.groups()[:-1]]
    assert all(1 <= die <= 100 for die in dice)
    assert sum(dice) == int(match.group(4))
    assert result.startswith(f"The roll has been posted to the conversation: 3d100:")


async def test_float_forms_of_whole_numbers_pass() -> None:
    channel = FakeChannel()
    result = await build_tools(channel)._tool_roll_dice({"sides": 6.0, "count": 1.0}, guild_id=1, channel_id=2, user_id=3)
    assert re.fullmatch(r"🎲 d6: [1-6]", channel.posts[0])
    assert result.startswith("The roll has been posted")


async def test_one_side_refuses() -> None:
    result = await build_tools()._tool_roll_dice({"sides": 1}, guild_id=1, channel_id=2, user_id=3)
    assert result == f"Error: the sides must be a whole number of at least {DICE_MIN_SIDES}."


async def test_out_of_range_count_refuses() -> None:
    tools = build_tools()
    assert (await tools._tool_roll_dice({"count": 0}, guild_id=1, channel_id=2, user_id=3)).startswith("Error: the count must be")
    assert (await tools._tool_roll_dice({"count": DICE_MAX_COUNT + 1}, guild_id=1, channel_id=2, user_id=3)).startswith("Error: the count must be")


async def test_non_numbers_refuse() -> None:
    tools = build_tools()
    assert (await tools._tool_roll_dice({"sides": "20"}, guild_id=1, channel_id=2, user_id=3)).startswith("Error: sides")
    assert (await tools._tool_roll_dice({"count": True}, guild_id=1, channel_id=2, user_id=3)).startswith("Error: count")


async def test_an_unknown_channel_refuses() -> None:
    result = await build_tools()._tool_roll_dice({}, guild_id=1, channel_id=2, user_id=3)
    assert result == "Error: the current channel is unknown."


async def test_a_failed_post_answers_an_error() -> None:
    import discord

    class FakeHTTPResponse:
        status = 403
        reason = "Forbidden"

    channel = FakeChannel(failure=discord.HTTPException(FakeHTTPResponse(), "forbidden"))
    result = await build_tools(channel)._tool_roll_dice({}, guild_id=1, channel_id=2, user_id=3)
    assert result.startswith("Error: the roll post failed:")
    assert channel.posts == []


async def test_the_harness_composition_carries_the_tool() -> None:
    from AgentEliza.tools.harness import HarnessOptions, HarnessTools

    names = [tool["function"]["name"] for tool in HarnessTools(HarnessOptions()).tools()]
    assert "roll_dice" in names
