"""The system message: barred rules leave when the model carries the trait."""

from AgentEliza.prompt import GATHER_RULE, HISTORY_RULE, SYSTEM_PROMPT, system_text


def test_the_barred_rules_ride_the_base_prompt() -> None:
    assert HISTORY_RULE in SYSTEM_PROMPT
    assert GATHER_RULE in SYSTEM_PROMPT


def test_the_barred_rules_leave_only_for_their_trait() -> None:
    plain = system_text("Bot", [])
    assert HISTORY_RULE in plain
    assert GATHER_RULE in plain
    tagged = system_text("Bot", [], quirks=("eager reader",))
    assert HISTORY_RULE not in tagged
    assert GATHER_RULE not in tagged
    # The rules around them stay in place.
    assert "answer with only `[no-reply]`" in tagged
    assert "ask with the ask_question tool" in tagged


def test_an_unlisted_trait_changes_nothing() -> None:
    assert HISTORY_RULE in system_text("Bot", [], quirks=("stiff prose",))
    assert GATHER_RULE in system_text("Bot", [], quirks=("stiff prose",))
