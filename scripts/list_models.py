"""List the models of a provider: context, prices, capability flags.

The price summary follows the model type. A text model reports the
operating cost of a 1M-token workload: 10x the input price, plus 1x the
output price and 1x the cache-read price (each per 1M tokens). An image
model reports the per-image cost of the request the AgentEliza image tool
sends: the 2K preset on the resolution-tier models (gpt-image-2 at 2K
medium), the flat generation price elsewhere, the default 1K tier when a
model prices by tier and the tool sends no resolution. An inpaint model
(an image edit model) reports its per-edit price at the same 2K preset:
the quality table wins when the model has one (2K medium), else the 2K
resolution tier, else the flat inpaint price. A music model reports the price of one standard song, 3 minutes
30 seconds (210 s): the duration bucket covering it, the per-second rate
scaled to it, or the flat generation price; a model pricing text (speech)
keeps its own unit.

The cost scale of the AgentEliza preset catalog reads these numbers.

Run it when new models land or prices change, to refresh the preset
catalog of AgentEliza:

    python scripts/list_models.py venice
    python scripts/list_models.py venice --type image
    python scripts/list_models.py venice --type inpaint
    python scripts/list_models.py venice --type music
    KIMI_API_KEY=... python scripts/list_models.py kimi-api
    ZAI_API_KEY=... python scripts/list_models.py zai-code

Only the standard library. Venice needs no key for the models list.
"""

import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime

# The base URLs mirror AgentEliza/providers.
BASE_URLS = {
    "venice": "https://api.venice.ai/api/v1"
  , "kimi-code": "https://api.kimi.com/coding/v1"
  , "kimi-api": "https://api.moonshot.ai/v1"
  , "zai-code": "https://api.z.ai/api/coding/paas/v4"
  , "zai-api": "https://api.z.ai/api/paas/v4"
}
ENV_KEYS = {"kimi-code": "KIMI_API_KEY", "kimi-api": "KIMI_API_KEY", "zai-code": "ZAI_API_KEY", "zai-api": "ZAI_API_KEY"}


def fetch(provider: str, key: str, model_type: str) -> list:
    url = BASE_URLS[provider] + "/models"
    if provider == "venice" and model_type:
        url += f"?type={model_type}"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    if key:
        request.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.load(response)
    return data.get("data", [])


# A music "standard song" runs 3 minutes 30 seconds: the per-song price
# scales the duration and per-second rates to it.
MUSIC_SONG_SECONDS = 210


def text_prices(pricing: dict) -> dict:
    """The operating cost of a text model: 10x input + 1x output + 1x
    cache-read, each per 1M tokens. A model that reports no cache price
    does not bill cache reads: the term adds 0."""
    entry = {
        "input_usd_per_m": (pricing.get("input") or {}).get("usd")
      , "cache_input_usd_per_m": (pricing.get("cache_input") or {}).get("usd")
      , "output_usd_per_m": (pricing.get("output") or {}).get("usd")
    }
    prices = (entry["input_usd_per_m"], entry["output_usd_per_m"])
    if all(price is not None for price in prices):
        cache = entry["cache_input_usd_per_m"] or 0
        entry["operating_usd_per_m"] = round(10 * prices[0] + prices[1] + cache, 6)
    return entry


def image_prices(spec: dict) -> dict:
    """The per-image cost of an image model under the AgentEliza request
    preset: the constraints carry the sizing dialect — a resolutions array
    means the tool sends the 2K preset (gpt-image-2 bills 2K medium), the
    rest pay the flat generation price or the default 1K tier."""
    pricing = spec.get("pricing") or {}
    constraints = spec.get("constraints") or {}
    resolutions = pricing.get("resolutions") or {}
    if not constraints.get("resolutions"):
        default = resolutions.get("1K") or {}
        return {"image_usd": default.get("usd") if resolutions else (pricing.get("generation") or {}).get("usd")}
    quality = ((pricing.get("quality") or {}).get("2K") or {}).get("medium") or {}
    return {"image_usd": (quality or resolutions.get("2K") or {}).get("usd")}


def inpaint_prices(spec: dict) -> dict:
    """The per-edit cost of an inpaint (image edit) model at the 2K preset:
    the quality table wins when the model has one (2K medium), else the 2K
    resolution tier, else the flat inpaint price."""
    pricing = spec.get("pricing") or {}
    quality = ((pricing.get("quality") or {}).get("2K") or {}).get("medium")
    if quality is not None:
        return {"edit_usd": quality.get("usd")}
    tier = (pricing.get("resolutions") or {}).get("2K")
    if tier is not None:
        return {"edit_usd": tier.get("usd")}
    return {"edit_usd": (pricing.get("inpaint") or {}).get("usd")}


def music_prices(pricing: dict) -> dict:
    """The price of one standard song (MUSIC_SONG_SECONDS). A model pricing
    text keeps its own unit."""
    if (pricing.get("generation") or {}).get("usd") is not None:
        return {"song_usd": pricing["generation"]["usd"], "unit": "per song"}
    per_second = (pricing.get("per_second") or {}).get("usd")
    if per_second is not None:
        return {"song_usd": round(per_second * MUSIC_SONG_SECONDS, 6), "unit": f"per {MUSIC_SONG_SECONDS}s song"}
    for entry in (pricing.get("durations") or {}).values():
        low = entry.get("min_seconds") or 0
        high = entry.get("max_seconds") or 10 ** 9
        if low <= MUSIC_SONG_SECONDS <= high:
            return {"song_usd": entry.get("usd"), "unit": f"per {MUSIC_SONG_SECONDS}s song"}
    chars = (pricing.get("per_thousand_characters") or {}).get("usd")
    if chars is not None:
        return {"unit_usd": chars, "unit": "per 1k chars"}
    return {}


def released_text(created) -> str:
    """The release date of one model (the created epoch), or '-' when the
    provider reports none. Rendered in the local timezone, so it matches
    what the operator sees on the site."""
    if not created:
        return "-"
    return datetime.fromtimestamp(created).strftime("%b %-d, %Y")


def parse_since(value: str) -> float:
    """The epoch of a YYYY-MM-DD date (start of day, local timezone), for
    the --since filter."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").timestamp()
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a YYYY-MM-DD date")


def rows(models: list) -> list:
    out = []
    for model in models:
        spec = model.get("model_spec", {}) or {}
        caps = spec.get("capabilities", {}) or {}
        mtype = model.get("type") or "text"
        entry = {"id": model.get("id", "?"), "context": model.get("context_length"), "type": mtype}
        created = model.get("created")
        if created:
            entry["created"] = created
            entry["released"] = released_text(created)
        pricing = spec.get("pricing", {}) or {}
        if mtype == "image":
            entry.update(image_prices(spec))
        elif mtype == "inpaint":
            entry.update(inpaint_prices(spec))
        elif mtype == "music":
            entry.update(music_prices(pricing))
        else:
            entry.update(text_prices(pricing))
        if provider_flags(caps):
            entry["flags"] = provider_flags(caps)
        out.append(entry)
    return out


def provider_flags(caps: dict) -> list:
    return sorted(
        name for name in ("supportsVision", "optimizedForCode", "supportsFunctionCalling", "supportsReasoning")
        if caps.get(name)
    )


def price_text(entry: dict) -> str:
    """The price column of one row: the operating cost of a text model, the
    per-image cost of an image model, the per-song cost of a music model."""
    if "operating_usd_per_m" in entry:
        return f"{entry['operating_usd_per_m']} op/Mtok"
    if "image_usd" in entry:
        return f"{entry['image_usd']} $/image"
    if "edit_usd" in entry:
        return f"{entry['edit_usd']} $/edit"
    if "song_usd" in entry:
        return f"{entry['song_usd']} $/song"
    if "unit_usd" in entry:
        return f"{entry['unit_usd']} {entry['unit']}"
    return "-"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("provider", choices=sorted(BASE_URLS))
    parser.add_argument("--type", default="", help="venice only: text, image, inpaint, or music")
    parser.add_argument("--key", default="", help="the API key (defaults to the provider env variable)")
    parser.add_argument("--json", action="store_true", help="print raw JSON rows")
    parser.add_argument(
        "--since", type=parse_since, default=None, metavar="YYYY-MM-DD"
      , help="keep the models released on or after this date (local timezone): the new releases since the last check"
    )
    args = parser.parse_args()
    key = args.key or os.environ.get(ENV_KEYS.get(args.provider, ""), "")
    try:
        models = fetch(args.provider, key, args.type)
    except OSError as e:
        print(f"the request failed: {e}", file=sys.stderr)
        return 1
    if args.since is not None:
        # The created epoch of the provider, the start of the given day local.
        models = [m for m in models if (m.get("created") or 0) >= args.since]
    table = rows(models)
    if args.json:
        print(json.dumps(table, indent=2))
        return 0
    print(f"{'id':36s} {'context':>9s} {'released':>13s} {'price':>16s}  flags")
    for entry in table:
        print(
            f"{entry['id']:36s} {str(entry['context']):>9s} "
            f"{entry.get('released', '-'):>13s} "
            f"{price_text(entry):>16s}  "
            f"{', '.join(entry.get('flags', []))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
