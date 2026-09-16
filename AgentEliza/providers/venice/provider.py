# The Venice provider class:
#    the request surface
#  , the usage rows
#  , and the bundled credit fetch.

import asyncio
from datetime import datetime

import aiohttp

from ..base import Provider, analyze_image_tool
from .catalog import VENICE_CREDIT_ALLOWANCE, VENICE_CHAT_PRESETS, VENICE_LIMIT_NAMES
from .tools import (
    _background_remove_tool
  , _edit_tool
  , _environment_tool
  , _image_tool
  , _music_tool
  , _parse_tool
  , _scrape_tool
  , _search_tool
  , preset_menu_line
)


class VeniceApiProvider(Provider):
    """Venice API open platform (pay-as-you-go, OpenAI-compatible)."""

    name = "Venice API"
    base_url = "https://api.venice.ai/api/v1"
    # A short list of the ~110 live models (GET /models needs no key); setmodel
    # accepts any other id with a notice. Context sizes from that endpoint.
    # The first entry is the default model of the provider (DeepSeek Lite).
    models = [
        "deepseek-v4-flash-0731"
      , "zai-org-glm-4.7-flash"
      , "z-ai-glm-5-3"
      , "venice-uncensored-1-2"
      , "kimi-k3"
      , "qwen-3-8-max"
      , "z-ai-glm-5-3-flash"
      , "gemini-3-5-flash"
      , "aion-labs-aion-3-0"
      , "inkling"
      , "deepseek-v4-pro"
      , "google-gemma-4-31b-it"
      , "gemma-4-uncensored"
      , "aion-labs-aion-3-0-mini"
      , "qwen-3-8-2-4t-a95b"
      , "qwen-3-8-27b"
    ]
    context_lengths = {
        "z-ai-glm-5-3": 1_000_000
      , "venice-uncensored-1-2": 128_000
      , "kimi-k3": 1_000_000
      , "qwen-3-8-max": 1_000_000
      , "z-ai-glm-5-3-flash": 1_048_576
      , "zai-org-glm-4.7-flash": 128_000
      , "gemini-3-5-flash": 1_000_000
      , "aion-labs-aion-3-0": 128_000
      , "inkling": 524_288
      , "deepseek-v4-flash-0731": 1_000_000
      , "deepseek-v4-pro": 1_000_000
      , "google-gemma-4-31b-it": 256_000
      , "gemma-4-uncensored": 256_000
      , "aion-labs-aion-3-0-mini": 128_000
      , "qwen-3-8-2-4t-a95b": 262_144
      , "qwen-3-8-27b": 262_144
      , "gemini-3-8-flash": 1_000_000
      , "llama-3.2-3b": 128_000
      , "llama-3.3-70b": 128_000
    }
    # The curated models with the supportsVision flag of the live model list: they accept image input through the chat contract.
    # `eliza providers` marks them, and the later direct image path rides on this set.
    vision_models = {
        "gemma-4-uncensored"
      , "z-ai-glm-5-3-flash"

      , "qwen-3-8-27b"
      , "qwen-3-8-max"

      , "gemini-3-5-flash"

      , "inkling"
      , "kimi-k3"

        # Under Tests! TODO: Validate if we keep it.
      , "venice-uncensored-1-2"
    }
    # Documented balance-and-limits endpoint, readable with the inference key.
    usage_url = "https://api.venice.ai/api/v1/api_keys/rate_limits"
    # Prompt caches expire after inactivity "typically 5-10 minutes" (docs);
    # the 300 s floor assumes the cache dead no later than it surely is.
    cache_ttl = 300
    # The vision model of analyze_image on this provider.
    vision_model = "gemma-4-uncensored"
    # The speech-to-text model of POST /audio/transcriptions, a beta
    # endpoint: it reads one audio file and answers the text.
    speech_model = "elevenlabs/scribe-v2"

    def speech_request(self, payload: bytes, filename: str, content_type: str):
        """The transcription POST: the audio file and the model as the multipart form.
           The endpoint takes no other dial (additionalProperties false).
        """
        form = aiohttp.FormData()
        form.add_field("file", payload, filename=filename or "audio", content_type=content_type or "application/octet-stream")
        form.add_field("model", self.speech_model)
        return "/audio/transcriptions", form

    def native_tools(self) -> list:
        """The provider tools: the vision tool, the augment set, the image generation and edit, the background removal, the song generation, and the environment tool.
           web_search takes the harness name, so the Venice search replaces the DuckDuckGo default while this provider is active.
           web_scrape, parse_document, generate_image, edit_image, remove_background, generate_song, and configure_environment join as additions;
           the harness web_fetch keeps its place, it reads the Discord file hosts with the bot token.
        """
        return [
            analyze_image_tool(self.vision_model)
          , _search_tool()
          , _scrape_tool()
          , _parse_tool()
          , _image_tool()
          , _edit_tool()
          , _background_remove_tool()
          , _music_tool()
          , _environment_tool()
        ]

    def preset_name(self, model_id: str) -> str | None:
        """The short preset name of a chat model id, None when unknown."""
        for name, preset in VENICE_CHAT_PRESETS.items():
            if model_id in (preset.get("normal"), preset.get("nsfw")):
                return name
        return None

    def request_model(self, name: str, nsfw: bool = False) -> str:
        """Resolve a model string for one request: a preset name (any casing) maps to its NSFW variant behind the 18+ gate, to its normal id anywhere else.
           A raw id passes through unchanged.
        """
        for catalog_name, preset in VENICE_CHAT_PRESETS.items():
            if catalog_name.lower() == str(name).strip().lower():
                if nsfw and preset.get("nsfw"):
                    return preset["nsfw"]
                return preset["normal"]
        return name

    def resolve_model(self, name: str) -> str:
        """Map a short preset name (any casing) to its model id, pass anything else through unchanged."""
        return self.request_model(name, nsfw=False)

    def context_length(self, model_name: str) -> int | None:
        """The context size of a model in tokens, None when unknown. A preset name resolves to its normal variant first."""
        return self.context_lengths.get(self.request_model(model_name, nsfw=False))

    def cost_of(self, data: dict) -> float:
        """The Venice request cost: the answer carries it split by billing currency.
           The USD part is the metric, the DIEM part fills in when the plan bills that way.
        """
        cost = data.get("cost") or {}
        return float(cost.get("usd") or cost.get("diem") or 0)

    def default_model(self) -> str:
        """The default of a cleared configuration: the preset NAME of the first model, so the default resolves per request like any preset —
           the 18+ variant behind the gate, the normal id elsewhere.
        """
        for name, preset in VENICE_CHAT_PRESETS.items():
            if preset.get("normal") == self.models[0]:
                return name
        return self.models[0]

    def preset_fallback(self, model: str) -> str | None:
        """The enabled preset one step up in cost from the preset holding the model (either variant), cycling to the cheapest at the ceiling.
           The cost of a preset is its cost property, 0 when the entry names none.
           A disabled preset never takes the move — the operator hid it from the agent.
           A preset with a smaller context window takes the move: the switch compacts the session onto its window first (the set_conversation_model closure of the engine).
           A session that sits on an excluded preset (a user choice) moves to the cheapest enabled preset above it.
           None when the model sits in no preset.
        """
        ranked = []
        current = None
        for name, preset in VENICE_CHAT_PRESETS.items():
            if model in (preset.get("normal"), preset.get("nsfw")):
                current = (preset.get("cost", 0.0), name)
            if preset.get("disabled"):
                continue
            ranked.append((preset.get("cost", 0.0), name))
        if current is None or not ranked:
            return None
        ranked.sort()
        if current not in ranked:
            above = [entry for entry in ranked if entry > current]
            return (above[0] if above else ranked[0])[1]
        nxt = ranked[(ranked.index(current) + 1) % len(ranked)]
        return None if nxt == current else nxt[1]

    def extra_payload(self, session_id: int, model: str = "", nsfw: bool = False) -> dict:
        # Venice appends its own system prompt unless told off; the harness
        # ships its own. prompt_cache_key routes a session to one backend.
        # A preset name in the model string resolves here, at request time:
        # the NSFW variant behind the 18+ gate, the normal id elsewhere.
        payload = {
            "venice_parameters": {"include_venice_system_prompt": False}
          , "prompt_cache_key": str(session_id)
        }
        if model:
            payload["model"] = self.request_model(model, nsfw)
        return payload

    def preset_menu(self) -> list:
        """One line per chat preset for the eliza providers list."""
        return [preset_menu_line(name, preset) for name, preset in VENICE_CHAT_PRESETS.items()]

    async def bundled_credits(self, session, api_key: str):
        """The bundled credit balance in credits of the rate-limits answer: balances.BUNDLED_CREDITS names USD, the credits ride at 100 a dollar.
           A plan without bundled credits answers zero. None when the answer cannot be read or names no balance.
        """
        try:
            async with session.get(
                "https://api.venice.ai/api/v1/api_keys/rate_limits"
                , headers={"Authorization": f"Bearer {api_key}"}
                , timeout=aiohttp.ClientTimeout(total=15),
            ) as response:
                if response.status != 200:
                    return None
                try:
                    data = await response.json(content_type=None)
                except Exception:
                    return None
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return None
        payload = data.get("data") if isinstance(data, dict) else {}
        balances = payload.get("balances") if isinstance(payload, dict) else None
        usd = balances.get("BUNDLED_CREDITS") if isinstance(balances, dict) else None
        if not isinstance(usd, (int, float)) or usd < 0:
            return None
        return float(usd) * 100

    def parse_usage(self, data: dict) -> list:
        payload = data.get("data") if isinstance(data.get("data"), dict) else {}
        balances = payload.get("balances") or {}
        usd = balances.get("USD")
        diem = balances.get("DIEM")
        tier = payload.get("apiTier") or {}
        is_charged = tier.get("isCharged")
        exhausted = payload.get("accessPermitted") is False or (
            is_charged
            and isinstance(usd, (int, float))
            and isinstance(diem, (int, float))
            and usd <= 0
            and diem <= 0
        )
        parts = [f"tier {tier.get('id') or 'unknown'}"]
        if isinstance(usd, (int, float)):
            parts.append(f"${usd:g} USD available")
        if isinstance(diem, (int, float)):
            parts.append(f"{diem:g} Diem available")
        bundled = balances.get("BUNDLED_CREDITS") if isinstance(balances, dict) else None
        if isinstance(bundled, (int, float)):
            # The endpoint names USD: the credits ride at 100 a dollar.
            parts.append(f"bundled credits {bundled * 100:,.6g} of {VENICE_CREDIT_ALLOWANCE:,}")
        text = ", ".join(parts)
        # The endpoint reports per-model limits only: the consumption lives in
        # the x-ratelimit-* response headers, which a poll cannot read. The
        # highest amount of each type stands for the whole key.
        amounts = {}
        for entry in payload.get("rateLimits") or []:
            for limit in entry.get("rateLimits") or []:
                limit_type = limit.get("type")
                amount = self._num(limit.get("amount"))
                if limit_type and amount is not None:
                    amounts[limit_type] = max(amounts.get(limit_type, 0), amount)
        if amounts:
            limits = ", ".join(
                f"{amount:,} {VENICE_LIMIT_NAMES.get(limit_type, limit_type)}"
                for limit_type, amount in sorted(amounts.items())
            )
            text += f"; {limits}"
        # nextEpochBegins names the epoch reset of the rate-limit windows: the row reset
        # carries its unix time, the surfaces render it beside the balance.
        reset = None
        epoch = payload.get("nextEpochBegins")
        if isinstance(epoch, str):
            try:
                reset = datetime.fromisoformat(epoch.replace("Z", "+00:00")).timestamp()
            except ValueError:
                reset = None
        return [{
            "name": "Balance"
            , "used": None
            , "limit": None
            , "percent": None
            , "reset": reset
            , "text": text
            , "exhausted": exhausted
        }]
