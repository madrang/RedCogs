# The agent resources of the Venice provider: the model documents served
# under the harness scheme, read through the resource tools of the harness.

from ...stats import cost_ceiling
from .catalog import (
    VENICE_CHAT_PRESETS
  , VENICE_CHAT_TRAIT_TEXT
  , VENICE_IMAGE_DIALECTS
  , VENICE_IMAGE_MODELS
  , VENICE_IMAGE_PROMPT_LIMITS
  , VENICE_IMAGE_TRAIT_LABELS
  , VENICE_EDIT_MODELS
  , VENICE_MUSIC_CONSTRAINTS
  , VENICE_MUSIC_MODELS
  , VENICE_PROMPT_MAX_CHARS
)

# The uris of the four documents. The tools name them in their refusal
# answers, so the registration and the pointers share one source.
MODELS_URI = "harness:///provider/models.md"
IMAGES_URI = "harness:///provider/images.md"
EDITS_URI = "harness:///provider/edits.md"
MUSIC_URI = "harness:///provider/music.md"


def _chat_text() -> str:
    """The chat presets: id, cost, traits, quirks, and the trait glossary."""
    lines = [
        "# The chat presets of Venice"
      , ""
      , "The cost is the output price in USD per 1M output tokens. A disabled preset stays a manual choice: the routing never lands on it."
      , ""
    ]
    for name, preset in VENICE_CHAT_PRESETS.items():
        traits = ", ".join(preset.get("traits", ())) or "plain"
        line = f"- {name} ({preset['normal']}) — ${preset.get('cost', 0):g} — {traits}"
        quirks = preset.get("negative_traits")
        if quirks:
            line += f" — quirks: {', '.join(quirks)}"
        if preset.get("disabled"):
            line += " — disabled"
        lines.append(line)
    lines += ["", "## The traits", ""]
    for trait, text in VENICE_CHAT_TRAIT_TEXT.items():
        lines.append(f"- {trait} — {text}")
    return "\n".join(lines)


def _render_text(ratio, catalog: dict, what: str, unit: str) -> str:
    """One render catalog: price, dialect, traits, and the live enable
    state of every model at the credit ratio of the context."""
    costs = [entry["cost"] for entry in catalog.values()]
    ceiling = cost_ceiling(ratio, min(costs), max(costs))
    lines = [f"# The {what} models of Venice", ""]
    if ratio is None:
        lines.append("The credit ratio reads unknown, so every model stays enabled.")
    elif ceiling is None:
        lines.append(f"The credit ratio sits at {ratio:.2f}, the healthy band: every model stays enabled.")
    else:
        lines.append(
            f"The credit ratio sits at {ratio:.2f}: a model stays enabled at or under ${ceiling:.4g} {unit}."
        )
    lines.append("")
    for name, entry in catalog.items():
        model = entry["model"]
        enabled = "enabled" if ceiling is None or entry["cost"] <= ceiling else "disabled"
        line = f"- {name} ({model}) — ${entry['cost']:g} {unit} — {enabled}"
        extras = entry.get("max_images")
        if extras is not None:
            line += f" — composites up to {extras} images"
        traits = ", ".join(
            VENICE_IMAGE_TRAIT_LABELS[trait] for trait in entry.get("traits", ()) if trait in VENICE_IMAGE_TRAIT_LABELS
        )
        if traits:
            line += f" — {traits}"
        prompt_limit = VENICE_IMAGE_PROMPT_LIMITS.get(model, VENICE_PROMPT_MAX_CHARS)
        line += f" — prompt limit {prompt_limit} characters — {VENICE_IMAGE_DIALECTS.get(model, 'ratio')} sizing"
        lines.append(line)
    return "\n".join(lines)


def _music_text() -> str:
    """The song models: price, the dials each model takes, the prompt limits."""
    lines = [
        "# The song models of Venice"
      , ""
      , "The cost is the price of the default request of the model. A request posts for approval first, the price rides the approval embed."
      , ""
    ]
    for name, entry in VENICE_MUSIC_MODELS.items():
        model = entry["model"]
        constraint = VENICE_MUSIC_CONSTRAINTS.get(model, {})
        line = f"- {name} ({model}) — ${entry['cost']:g} a song"
        dials = []
        if "lyrics" in constraint:
            dials.append(f"lyrics to {constraint['lyrics']} characters")
        if "duration" in constraint:
            low, high, default = constraint["duration"]
            dials.append(f"duration {low}-{high} s, default {default} s")
        if constraint.get("instrumental"):
            dials.append("force instrumental")
        if dials:
            line += f" — {', '.join(dials)}"
        line += f" — prompt limit {constraint.get('prompt', VENICE_PROMPT_MAX_CHARS)} characters"
        lines.append(line)
    return "\n".join(lines)


def venice_agent_resources(context: dict | None = None) -> list:
    """The live model documents of the provider: the chat presets, the image
    and edit catalogs with their enable state at the credit ratio of the
    context, and the song models. The builders render at read time."""
    ratio = (context or {}).get("ratio")
    return [
        {
            "uri": "provider/models.md"
          , "name": "models.md"
          , "description": "The chat presets of the active provider: ids, costs, traits, and quirks."
          , "build": _chat_text
        }
      , {
            "uri": "provider/images.md"
          , "name": "images.md"
          , "description": "The image models: price, traits, and the live enable state at the current credit ratio."
          , "build": lambda: _render_text(ratio, VENICE_IMAGE_MODELS, "image", "an image")
        }
      , {
            "uri": "provider/edits.md"
          , "name": "edits.md"
          , "description": "The image edit models: price, compositing, and the live enable state at the current credit ratio."
          , "build": lambda: _render_text(ratio, VENICE_EDIT_MODELS, "image edit", "an edit")
        }
      , {
            "uri": "provider/music.md"
          , "name": "music.md"
          , "description": "The song models: price, the dials each model takes, and the prompt limits."
          , "build": _music_text
        }
    ]
