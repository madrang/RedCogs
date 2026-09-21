# The decision flow of the Venice API: the chat model routing of a new conversation.
# The endpoint reads a state and typed questions, and answers structured
# judgments, not generated text.

from .catalog import (
  JEV_MODEL_ID, VENICE_CHAT_CAPABILITIES, VENICE_CHAT_PRESETS
  , VENICE_ROUTING_LITE_COST, VENICE_ROUTING_TRAIT_AT,
)

# The answer fields a noul judgment may carry its probability under: the live
# endpoint names probability, the rest guard a drift of the field name.
_PROBABILITY_KEYS = ("probability", "noul", "value")


def model_decision_request(state_text: str) -> dict:
    """
       The decision body of a new conversation: one noul question per capability.
       Each judgment names the probability the conversation needs the trait.
    """
    questions = {}
    for trait in VENICE_CHAT_CAPABILITIES:
        questions[trait] = {
            "type": "noul"
            , "instructions": f"Does this conversation need {trait}? The probability: 0 reads no, 1 reads yes."
        }
    return {
        "model": JEV_MODEL_ID
        , "state": state_text
        , "questions": questions
    }


def trait_strengths(data: dict) -> dict:
    """
       The capability strengths of a decision answer: the trait names over their
       probabilities, each clamped between 0 and 1. A missing judgment or an
       unreadable value stays out.
    """
    answers = data.get("answers") if isinstance(data, dict) else None
    if not isinstance(answers, dict):
        return {}
    strengths = {}
    for trait in VENICE_CHAT_CAPABILITIES:
        judgment = answers.get(trait)
        if not isinstance(judgment, dict):
            continue
        for key in _PROBABILITY_KEYS:
            value = judgment.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                strengths[trait] = min(max(float(value), 0.0), 1.0)
                break
    return strengths


def preset_for_traits(strengths: dict, lite: bool = False) -> str | None:
    """
       The preset the capability strengths route to: the traits at or above
       VENICE_ROUTING_TRAIT_AT count as needed, and the first enabled preset
       whose traits cover them wins, the catalog order as the preference
       order. lite keeps the walk at the lite cost ceiling.
       None when no preset covers the set: the caller keeps the configured
       model.
    """
    needed = {trait for trait, strength in strengths.items() if strength >= VENICE_ROUTING_TRAIT_AT}
    for name, preset in VENICE_CHAT_PRESETS.items():
        if preset.get("disabled"):
            continue
        if lite and preset.get("cost", 0.0) > VENICE_ROUTING_LITE_COST:
            continue
        if all(trait in preset.get("traits", ()) for trait in needed):
            return name
    return None
