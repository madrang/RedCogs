# The decision flow of the Venice API: the chat model routing of a new conversation.
# The endpoint reads a state and typed questions, and answers structured
# judgments, not generated text.

from .catalog import JEV_MODEL_ID, VENICE_CHAT_PRESETS


def model_decision_request(state_text: str) -> dict:
    """
       The decision body of a new conversation: one choice question over the enabled chat presets.
       Each option rubric names the traits and the operating cost of the preset.
       The option other carries no rubric, it answers a conversation that needs no special capability.
    """
    criteria = {}
    for name, preset in VENICE_CHAT_PRESETS.items():
        if preset.get("disabled"):
            continue
        traits = ", ".join(preset.get("traits") or ()) or "general conversation"
        criteria[name] = f"{traits}. Operating cost {preset.get('cost', 0.0):.2f} of the priciest catalog model."
    criteria["other"] = None
    return {
        "model": JEV_MODEL_ID
        , "state": state_text
        , "questions": {
            "model": {
                "type": "choice"
                , "instructions": (
                    "Which chat model should answer this conversation? "
                    "Pick the cheapest model whose capabilities fit the conversation. "
                    "Answer other when no special capability is needed."
                )
                , "criteria": criteria
            }
        }
    }


def model_decision_answer(data: dict) -> str | None:
    """
       The chosen preset name of a decision answer.
       None when the answer names no enabled preset, the option other included: the caller keeps the configured model.
    """
    answers = data.get("answers") if isinstance(data, dict) else None
    answer = answers.get("model") if isinstance(answers, dict) else None
    choice = answer.get("choice") if isinstance(answer, dict) else None
    preset = VENICE_CHAT_PRESETS.get(choice) if isinstance(choice, str) else None
    if preset is None or preset.get("disabled"):
        return None
    return choice
