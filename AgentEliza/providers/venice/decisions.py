# The decision flow of the Venice API: the chat model routing of a new conversation.
# The endpoint reads a state and typed questions, and answers structured
# judgments, not generated text.

from .catalog import (
  JEV_MODEL_ID, VENICE_CHAT_PRESETS
  , VENICE_CREDIT_GATE_BUFFER, VENICE_CREDIT_ROUTING_FLOOR
  , VENICE_ROUTING_PRESSURE_MAX, VENICE_ROUTING_PRESSURE_MIN, VENICE_ROUTING_QUIRK_MALUS, VENICE_ROUTING_TRAIT_AT,
)

# The answer fields a noul judgment may carry its probability under: the live
# endpoint names probability, the rest guard a drift of the field name.
_PROBABILITY_KEYS = ("probability", "noul", "value")
# The answer fields a score judgment may carry its weighted position under.
_SCORE_KEYS = ("score", "value")
# The answer fields a choice judgment may carry its picked option under.
_CHOICE_KEYS = ("choice", "pick")
# The answer fields a choice judgment may carry the option probabilities
# under: a map of option name to probability.
_PROBABILITY_MAP_KEYS = ("probability", "probabilities", "options")
# The binary capabilities the router judges as noul questions. The long
# context trait never becomes a question: the opening message cannot show
# the conversation length, the selection scores the trait instead.
_NOUL_TRAITS = ("vision", "nsfw")
# The ends of the answer detail axis: the score question judges them as one
# axis, the catalog traits and the tool vocabulary keep both words.
_CONCISE_SIDE, _THOROUGH_SIDE = "concise answers", "thorough answers"
# The activities the choice question offers: the catalog traits and the tool
# vocabulary keep the same words.
_ACTIVITY_TRAITS = ("coding", "reasoning", "writing", "roleplay", "storytelling", "imagination")
# The rubric of each activity: one fragment that names the work, the question
# instructions already name the conversation.
_ACTIVITY_RUBRICS = {
    "coding": "Writing, fixing, or reviewing code"
  , "reasoning": "Analyzing or solving a hard problem"
  , "writing": "Polished prose or text work"
  , "roleplay": "Playing characters or personas"
  , "storytelling": "Stories or narrative"
  , "imagination": "Original ideas beyond the routine"
}
# The question ids.
_DETAIL_QUESTION = "detail"
_ACTIVITY_QUESTION = "activity"
# The rubric of the detail axis: ordered from the concise end to the thorough
# end, indexed from 0. The weighted answer lands between levels. The levels
# name the detail a task needs, never the length of the text: a thorough
# answer carries more specific detail, not more words.
_DETAIL_RUBRIC = (
    "The facts alone, no depth"
  , "Ordinary conversational depth"
  , "Every specific detail the task needs"
)


def model_decision_request(state_text: str) -> dict:
    """
       The decision body of a new conversation: a noul question per binary
       capability, one score question on the answer detail axis, and one choice
       question on the activity of the conversation.
    """
    questions = {}
    for trait in _NOUL_TRAITS:
        questions[trait] = {
            "type": "noul"
            , "instructions": f"Does this conversation need {trait}? The probability: 0 reads no, 1 reads yes."
        }
    questions[_DETAIL_QUESTION] = {
        "type": "score"
        , "instructions": "How thorough should the answers of this conversation be?"
        , "criteria": list(_DETAIL_RUBRIC)
    }
    questions[_ACTIVITY_QUESTION] = {
        "type": "choice"
        , "instructions": "Which activity fits this conversation best?"
        , "criteria": {**_ACTIVITY_RUBRICS, "other": None}
    }
    return {
        "model": JEV_MODEL_ID
        , "state": state_text
        , "questions": questions
    }


def trait_strengths(data: dict) -> dict:
    """
       The capability strengths of a decision answer: the noul probabilities,
       the detail side at its distance from the rubric middle, and the picked
       activity at its option probability. A missing judgment or an
       unreadable value stays out.
    """
    answers = data.get("answers") if isinstance(data, dict) else None
    if not isinstance(answers, dict):
        return {}
    strengths = {}
    for trait in _NOUL_TRAITS:
        judgment = answers.get(trait)
        if not isinstance(judgment, dict):
            continue
        for key in _PROBABILITY_KEYS:
            value = judgment.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                strengths[trait] = min(max(float(value), 0.0), 1.0)
                break
    judgment = answers.get(_DETAIL_QUESTION)
    if isinstance(judgment, dict):
        for key in _SCORE_KEYS:
            value = judgment.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                position = min(max(float(value), 0.0), 2.0)
                side = _CONCISE_SIDE if position < 1.0 else _THOROUGH_SIDE
                strengths[side] = abs(position - 1.0)
                break
    activity = activity_pick(data)
    if activity is not None:
        judgment = answers.get(_ACTIVITY_QUESTION)
        for key in _PROBABILITY_MAP_KEYS:
            probabilities = judgment.get(key)
            if isinstance(probabilities, dict):
                value = probabilities.get(activity)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    strengths[activity] = min(max(float(value), 0.0), 1.0)
                    break
    return strengths


def activity_pick(data: dict) -> str | None:
    """
       The activity trait the choice judgment picks: the most representative
       activity of the conversation. None when the pick names other, nothing,
       or a word outside the activity list.
    """
    answers = data.get("answers") if isinstance(data, dict) else None
    judgment = answers.get(_ACTIVITY_QUESTION) if isinstance(answers, dict) else None
    if not isinstance(judgment, dict):
        return None
    for key in _CHOICE_KEYS:
        pick = judgment.get(key)
        if pick in _ACTIVITY_TRAITS:
            return pick
    return None


def _cost_pressure(ratio: float | None) -> float:
    """
       The weight of the preset cost in the selection tiebreak: the pressure
       names the cost step one extra trait must stay under. The ratio of a
       healthy balance reads the min, the routing floor reads the max, and
       None (no cycle read) reads the min.
    """
    if ratio is None:
        return VENICE_ROUTING_PRESSURE_MIN
    span = VENICE_CREDIT_GATE_BUFFER - VENICE_CREDIT_ROUTING_FLOOR
    part = (VENICE_CREDIT_GATE_BUFFER - ratio) / span
    part = min(max(part, 0.0), 1.0)
    return VENICE_ROUTING_PRESSURE_MIN + (VENICE_ROUTING_PRESSURE_MAX - VENICE_ROUTING_PRESSURE_MIN) * part


def select_preset(strengths: dict, activity: str | None = None, nsfw_allowed: bool = False, ratio: float | None = None) -> str | None:
    """
       The preset the decision answer routes to: the traits at or above
       VENICE_ROUTING_TRAIT_AT count as needed, the picked activity joins
       them, and an nsfw need dies unless the session allows it. The stages
       remove the presets that miss a needed trait: nsfw first, the activity,
       the detail side, then vision. The survivors score by extra traits minus
       the cost pressure and the quirk malus (two quirks read like one
       missing trait): for the same traits the cheaper preset wins, and a
       costlier preset wins only through the traits it adds. Equal scores
       keep the catalog order. None when no enabled preset carries every
       needed trait: the caller keeps the configured model.
    """
    needed = {trait for trait, strength in strengths.items() if strength >= VENICE_ROUTING_TRAIT_AT}
    if activity is not None:
        needed.add(activity)
    if not nsfw_allowed:
        needed.discard("nsfw")
    pressure = _cost_pressure(ratio)
    best_name = None
    best_score = None
    for name, preset in VENICE_CHAT_PRESETS.items():
        if preset.get("disabled"):
            continue
        traits = set(preset.get("traits", ()))
        if not needed <= traits:
            continue
        score = (
            len(traits - needed)
            - pressure * preset.get("cost", 0.0)
            - VENICE_ROUTING_QUIRK_MALUS * len(preset.get("negative_traits", ()))
        )
        if best_score is None or score > best_score:
            best_name = name
            best_score = score
    return best_name
