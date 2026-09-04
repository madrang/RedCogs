"""List the chat models of a provider: context, prices, capability flags.

Run it when new models land or prices change, to refresh the preset
catalog of AgentEliza (the cost scale of the traits):

    python scripts/list_models.py venice
    python scripts/list_models.py venice --type image
    KIMI_API_KEY=... python scripts/list_models.py kimi-api
    ZAI_API_KEY=... python scripts/list_models.py zai-code

Only the standard library. Venice needs no key for the models list.
"""

import argparse
import json
import os
import sys
import urllib.request

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


def rows(models: list) -> list:
    out = []
    for model in models:
        spec = model.get("model_spec", {}) or {}
        caps = spec.get("capabilities", {}) or {}
        pricing = spec.get("pricing", {}) or {}
        entry = {
            "id": model.get("id", "?")
          , "context": model.get("context_length")
          , "input_usd_per_m": (pricing.get("input") or {}).get("usd")
          , "cache_input_usd_per_m": (pricing.get("cache_input") or {}).get("usd")
          , "output_usd_per_m": (pricing.get("output") or {}).get("usd")
        }
        if provider_flags(caps):
            entry["flags"] = provider_flags(caps)
        if model.get("type"):
            entry["type"] = model.get("type")
        out.append(entry)
    return out


def provider_flags(caps: dict) -> list:
    return sorted(
        name for name in ("supportsVision", "optimizedForCode", "supportsFunctionCalling", "supportsReasoning")
        if caps.get(name)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("provider", choices=sorted(BASE_URLS))
    parser.add_argument("--type", default="", help="venice only: text or image")
    parser.add_argument("--key", default="", help="the API key (defaults to the provider env variable)")
    parser.add_argument("--json", action="store_true", help="print raw JSON rows")
    args = parser.parse_args()
    key = args.key or os.environ.get(ENV_KEYS.get(args.provider, ""), "")
    try:
        models = fetch(args.provider, key, args.type)
    except OSError as e:
        print(f"the request failed: {e}", file=sys.stderr)
        return 1
    table = rows(models)
    if args.json:
        print(json.dumps(table, indent=2))
        return 0
    print(f"{'id':36s} {'context':>9s} {'in $/M':>8s} {'out $/M':>8s}  flags")
    for entry in table:
        print(
            f"{entry['id']:36s} {str(entry['context']):>9s} "
            f"{str(entry['input_usd_per_m'] or '-'):>8s} {str(entry['output_usd_per_m'] or '-'):>8s}  "
            f"{', '.join(entry.get('flags', []))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
