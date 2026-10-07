"""The dice tool: random rolls for roleplay games."""

import random

import discord

# The sides floor of one die: one side has no roll range.
DICE_MIN_SIDES = 2
# The dice ceiling of one roll. The result stays one short line.
DICE_MAX_COUNT = 25
# The reading ladder of a roll, lowest first. The ends carry the accents:
# the lowest possible roll reads a critical failure, the highest a perfect
# success. The six steps between follow the graded ladder of "Dice and
# System Matters" (perilplanet.com), spread over the possible span.
DICE_READINGS = (
    "a critical failure: the worst possible roll"
  , "a failure with a complication"
  , "a failure"
  , "a near miss with a small gain"
  , "a success with a cost"
  , "a success"
  , "a strong success"
  , "a perfect success: the best possible roll"
)


def _reading(rolls: list, sides: int) -> str:
    """The guidance text of one roll: its position in the possible span."""
    total = sum(rolls)
    lowest, highest = len(rolls), len(rolls) * sides
    if total == lowest:
        return DICE_READINGS[0]
    if total == highest:
        return DICE_READINGS[-1]
    index = 1 + min(5, (total - lowest) * 6 // (highest - lowest))
    return DICE_READINGS[index]


def _whole(raw, name: str):
    """The validated whole number of an argument, or an error text.

    A model often emits a whole number as a float. The float form of an
    integer reads as the integer, anything else refuses. None passes
    through, so the caller keeps the default.
    """
    if raw is None:
        return None, None
    if isinstance(raw, float) and raw.is_integer():
        raw = int(raw)
    if isinstance(raw, bool) or not isinstance(raw, int):
        return None, f"Error: {name} must be a whole number."
    return raw, None


class DiceTools:
    """The roll_dice tool."""

    def dice_tools(self) -> list:
        """The OpenAI function schema of the dice tool."""
        return [
            {
                "type": "function"
                , "function": {
                    "name": "roll_dice"
                    , "description": (
                        "Roll dice for a roleplay game. "
                        "Use it whenever an encounter or an action needs a random result. "
                        "The roll posts to the conversation at once, and the user sees it. "
                        "Weave the result into your narration, but do not repeat the roll line."
                    )
                    , "parameters": {
                        "type": "object"
                        , "properties": {
                            "sides": {
                                "type": "integer"
                                , "default": 6
                                , "description": f"The sides of one die, at least {DICE_MIN_SIDES}. The standard roleplay dice are 4, 6, 8, 10, 12, 20, and 100."
                            }
                            , "count": {
                                "type": "integer"
                                , "default": 1
                                , "description": f"How many dice to roll, 1 to {DICE_MAX_COUNT}."
                            }
                        }
                    }
                }
            }
        ]

    async def _tool_roll_dice(self, arguments: dict, *, guild_id, channel_id, user_id) -> str:
        sides, error = _whole(arguments.get("sides"), "sides")
        if error:
            return error
        count, error = _whole(arguments.get("count"), "count")
        if error:
            return error
        if sides is not None and sides < DICE_MIN_SIDES:
            return f"Error: the sides must be a whole number of at least {DICE_MIN_SIDES}."
        if count is not None and not 1 <= count <= DICE_MAX_COUNT:
            return f"Error: the count must be 1 to {DICE_MAX_COUNT}."
        sides = sides or 6
        count = count or 1
        rolls = [random.randint(1, sides) for _ in range(count)]
        if count == 1:
            line = f"d{sides}: {rolls[0]}"
        else:
            listed = ", ".join(str(roll) for roll in rolls)
            line = f"{count}d{sides}: {listed} = {sum(rolls)}"
        channel = await self.channel_getter(channel_id) if self.channel_getter else None
        if channel is None:
            return "Error: the current channel is unknown."
        try:
            await channel.send(f"🎲 {line}")
        except discord.HTTPException as e:
            return f"Error: the roll post failed: {e}"
        return f"The roll has been posted to the conversation: {line}. The roll reads {_reading(rolls, sides)}. Do not repeat the roll line in your answer."
