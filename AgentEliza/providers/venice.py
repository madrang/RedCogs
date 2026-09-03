from urllib.parse import urlparse

import base64
import random
import re

import aiohttp

from ..llm_chat import ChatError
from ..tools.base import _cap
from .base import Provider, analyze_image_tool

# The limit types of the rate-limits endpoint, spelled out for the usage rows.
VENICE_LIMIT_NAMES = {"RPM": "requests/min", "RPD": "requests/day", "TPM": "tokens/min"}
# The augment endpoints are experimental and billed per request ($0.01 each).
VENICE_QUERY_MAX_CHARS = 400
VENICE_SEARCH_MAX_LIMIT = 20
# Image generation: the endpoint-wide prompt cap.
VENICE_PROMPT_MAX_CHARS = 7500
# The default image model, Venice's own pixel-based one. The other ids come
# from the live model list (GET /models?type=image needs no key).
VENICE_IMAGE_MODEL = "venice-sd35"
VENICE_IMAGE_MODELS = (
    "venice-sd35"
  , "flux-2-pro"
  , "gpt-image-2"
  , "nano-banana-2"
  , "qwen-image-2"
  , "grok-imagine-image"
  , "seedream-v4"
)
# Prompt caps under the endpoint-wide 7500 (the per-model constraints of the
# live model list).
VENICE_IMAGE_PROMPT_LIMITS = {"venice-sd35": 1500, "flux-2-pro": 3000}
# The sizing dialect of each curated model, an unknown id takes the ratio
# dialect: pixel models size through width and height, the resolution-tier
# models carry a fixed resolution (and gpt-image-2 a quality) preset, the
# rest take the aspect ratio as-is.
VENICE_IMAGE_DIALECTS = {
    "venice-sd35": "pixel"
  , "flux-2-pro": "ratio"
  , "qwen-image-2": "ratio"
  , "seedream-v4": "ratio"
  , "gpt-image-2": "resolution"
  , "nano-banana-2": "resolution"
  , "grok-imagine-image": "resolution"
}
# The presets of the resolution-tier models. 2K and medium for now, adjust
# after some use.
VENICE_IMAGE_RESOLUTION = "2K"
VENICE_IMAGE_QUALITY = "medium"
# aspect_ratio to pixels for venice-sd35: sides at most 1280, multiples of
# its widthHeightDivisor 16, about one megapixel.
VENICE_PIXEL_RATIOS = {
    "1:1": (1024, 1024)
  , "4:3": (1024, 768)
  , "3:4": (768, 1024)
  , "3:2": (1216, 832)
  , "2:3": (832, 1216)
  , "16:9": (1280, 720)
  , "9:16": (720, 1280)
  , "21:9": (1280, 544)
  , "4:5": (896, 1120)
}
# The seed range of the endpoint.
VENICE_SEED_MAX = 999_999_999


def _search_tool() -> dict:
    """The augment search as a native tool, under the harness web_search name."""

    async def handler(arguments, call_api, fetch_url=None, api_post=None, send_file=None, channel_nsfw=None):
        query = str(arguments.get("query") or "").strip()
        if not query:
            return "Error: the query must be a non-empty string."
        if len(query) > VENICE_QUERY_MAX_CHARS:
            return f"Error: the query is over the {VENICE_QUERY_MAX_CHARS}-character limit of the search endpoint."
        body = {"query": query}
        limit = arguments.get("limit")
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = None
        if limit is not None:
            body["limit"] = max(1, min(VENICE_SEARCH_MAX_LIMIT, limit))
        try:
            data = await api_post("/augment/search", json_body=body)
        except ChatError as e:
            return f"Error: the search failed: {e}"
        results = data.get("results") or []
        if not results:
            return "(no results)"
        lines = []
        for index, result in enumerate(results, 1):
            title = " ".join(str(result.get("title") or "").split())
            snippet = " ".join(str(result.get("content") or "").split())
            date = " ".join(str(result.get("date") or "").split())
            suffix = f" ({date})" if date else ""
            lines.append(f"{index}. {title}{suffix}\n{result.get('url')}\n{snippet}")
        return _cap("\n\n".join(lines))

    return {
        "name": "web_search"
        , "description": (
            "Search the web through the Venice search. The result is a numbered list. "
            "Each entry has a title, a URL, and a snippet. "
            "Use web_fetch or web_scrape on a result URL to read the page."
        )
        , "parameters": {
            "type": "object"
            , "properties": {
                "query": {"type": "string", "description": "The search query, at most 400 characters."}
                , "limit": {"type": "integer", "description": f"How many results to return, 1 to {VENICE_SEARCH_MAX_LIMIT}. Default 10."}
            }
            , "required": ["query"]
        }
        , "handler": handler
    }


def _scrape_tool() -> dict:
    """The augment scrape as a native tool: one page as markdown."""

    async def handler(arguments, call_api, fetch_url=None, api_post=None, send_file=None, channel_nsfw=None):
        url = str(arguments.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            return "Error: the url must be an http(s) URL."
        try:
            data = await api_post("/augment/scrape", json_body={"url": url})
        except ChatError as e:
            return f"Error: the scrape failed: {e}"
        content = data.get("content")
        if not content:
            return "Error: the scrape returned no content."
        return _cap(str(content))

    return {
        "name": "web_scrape"
        , "description": (
            "Fetch one web page through the Venice scraper and return its markdown. "
            "The scraper renders the page in a browser, so pages built by scripts work too. "
            "The endpoint refuses X (Twitter) and Reddit pages. "
            "The Discord file hosts need web_fetch instead."
        )
        , "parameters": {
            "type": "object"
            , "properties": {
                "url": {"type": "string", "description": "The http(s) URL of the page."}
            }
            , "required": ["url"]
        }
        , "handler": handler
    }


def _parse_tool() -> dict:
    """The augment text parser as a native tool: one document as text."""

    async def handler(arguments, call_api, fetch_url=None, api_post=None, send_file=None, channel_nsfw=None):
        url = str(arguments.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            return "Error: the url must be the http(s) URL of a document."
        if fetch_url is None:
            return "Error: the document download is not available here."
        fetched = await fetch_url(url)
        if fetched is None:
            return "Error: the download of the document failed."
        body, content_type = fetched
        name = urlparse(url).path.rsplit("/", 1)[-1][:100] or "document"
        form = aiohttp.FormData()
        form.add_field("file", body, filename=name, content_type=content_type)
        try:
            data = await api_post("/augment/text-parser", data=form)
        except ChatError as e:
            return f"Error: the parse failed: {e}"
        text = data.get("text")
        if not text:
            return "Error: the parser returned no text."
        return _cap(str(text))

    return {
        "name": "parse_document"
        , "description": (
            "Extract the text of one document at an http(s) URL: pdf, docx, pptx, xlsx, "
            "epub, csv, json, yaml, xml, or a code file. The attachment URLs of messages work. "
            "Legacy .doc and .ppt files are refused. The result is the extracted text."
        )
        , "parameters": {
            "type": "object"
            , "properties": {
                "url": {"type": "string", "description": "The http(s) URL of the document."}
            }
            , "required": ["url"]
        }
        , "handler": handler
    }


def _image_tool() -> dict:
    """The image generation endpoint as a native tool: one image, posted to
    the conversation. The endpoint answers in base64 JSON (return_binary
    stays false): the binary mode does not fit provider_post. The agent
    controls the prompt, the model, the aspect ratio, the negative prompt,
    and the cfg scale. The resolution, the quality, the watermark, the
    exif metadata, and the seed are preset here; safe_mode drops only on an
    age-restricted channel."""

    async def handler(arguments, call_api, fetch_url=None, api_post=None, send_file=None, channel_nsfw=None):
        prompt = str(arguments.get("prompt") or "").strip()
        if not prompt:
            return "Error: the prompt must be a non-empty string."
        model = str(arguments.get("model") or "").strip() or VENICE_IMAGE_MODEL
        prompt_limit = VENICE_IMAGE_PROMPT_LIMITS.get(model, VENICE_PROMPT_MAX_CHARS)
        if len(prompt) > prompt_limit:
            return f"Error: the prompt is over the {prompt_limit}-character limit of the model {model}."
        body = {
            "model": model
            , "prompt": prompt
            , "hide_watermark": True
            , "embed_exif_metadata": True
            # A fresh random seed per request.
            , "seed": random.randint(-VENICE_SEED_MAX, VENICE_SEED_MAX)
        }
        dialect = VENICE_IMAGE_DIALECTS.get(model, "ratio")
        aspect_ratio = str(arguments.get("aspect_ratio") or "").strip()
        if dialect == "pixel":
            # The pixel model has no aspect_ratio field: the ratio maps to
            # width and height, an unmapped ratio keeps the 1024x1024 default.
            pixels = VENICE_PIXEL_RATIOS.get(aspect_ratio)
            if pixels is not None:
                body["width"], body["height"] = pixels
        elif aspect_ratio:
            body["aspect_ratio"] = aspect_ratio
        if dialect == "resolution":
            body["resolution"] = VENICE_IMAGE_RESOLUTION
            if model == "gpt-image-2":
                # gpt-image-2 is the only curated model with a quality field,
                # and its default high bills high.
                body["quality"] = VENICE_IMAGE_QUALITY
        cfg_scale = arguments.get("cfg_scale")
        try:
            cfg_scale = float(cfg_scale)
        except (TypeError, ValueError):
            cfg_scale = None
        if cfg_scale is not None:
            if not 0 < cfg_scale <= 20:
                return "Error: the cfg_scale must be a number over 0 and at most 20."
            body["cfg_scale"] = cfg_scale
        negative_prompt = str(arguments.get("negative_prompt") or "").strip()
        if negative_prompt:
            body["negative_prompt"] = negative_prompt
        if channel_nsfw is not None and await channel_nsfw():
            # The endpoint default true blurs adult content: it drops only
            # where Discord itself gates the channel behind 18+.
            body["safe_mode"] = False
        try:
            data = await api_post("/image/generate", json_body=body)
        except ChatError as e:
            return f"Error: the image generation failed: {e}"
        images = data.get("images") or []
        if not images:
            return "Error: the image generation returned no image."
        if send_file is None:
            return "Error: the image posting is not available here."
        try:
            raw = base64.b64decode(images[0])
        except (ValueError, TypeError):
            return "Error: the image generation returned unreadable image data."
        if not raw:
            return "Error: the image generation returned an empty image."
        # The format stays the endpoint default (webp). The id names the file:
        # each image of a conversation lands under its own name.
        name = re.sub(r"[\s/\\]+", "-", str(data.get("id") or "venice-image"))[:100]
        return await send_file(f"{name}.webp", raw)

    return {
        "name": "generate_image"
        , "description": (
            "Generate one image from a text prompt through Venice and post it to the conversation. "
            "The size comes from the aspect_ratio alone; the tool sets the resolution and the quality itself. "
            f"Known models: {', '.join(VENICE_IMAGE_MODELS)}. The default {VENICE_IMAGE_MODEL} is the only pixel model "
            "and takes a cfg_scale."
        )
        , "parameters": {
            "type": "object"
            , "properties": {
                "prompt": {"type": "string", "description": "What to draw."}
                , "model": {"type": "string", "description": f"The image model. Default {VENICE_IMAGE_MODEL}."}
                , "aspect_ratio": {"type": "string", "description": "The aspect ratio of the image, for example 1:1, 16:9, or 9:16."}
                , "negative_prompt": {"type": "string", "description": "What to keep out of the image."}
                , "cfg_scale": {"type": "number", "description": "How strictly the pixel model follows the prompt, over 0 up to 20. Default of the endpoint."}
            }
            , "required": ["prompt"]
        }
        , "handler": handler
    }


class VeniceApiProvider(Provider):
    """Venice API open platform (pay-as-you-go, OpenAI-compatible)."""

    name = "Venice API"
    base_url = "https://api.venice.ai/api/v1"
    # A short list of the ~110 live models (GET /models needs no key); setmodel
    # accepts any other id with a notice. Context sizes from that endpoint.
    models = [
        "z-ai-glm-5-3"
      , "venice-uncensored-1-2"
      , "kimi-k3"
      , "qwen-3-8-max"
      , "z-ai-glm-5-3-flash"
      , "gemini-3-5-flash"
      , "aion-labs-aion-3-0"
      , "inkling"
    ]
    context_lengths = {
        "z-ai-glm-5-3": 1_000_000
      , "venice-uncensored-1-2": 128_000
      , "kimi-k3": 1_000_000
      , "qwen-3-8-max": 1_000_000
      , "z-ai-glm-5-3-flash": 1_048_576
      , "gemini-3-5-flash": 1_000_000
      , "aion-labs-aion-3-0": 128_000
      , "inkling": 524_288
    }
    # The curated models with the supportsVision flag of the live model list:
    # they accept image input through the chat contract. `eliza providers`
    # marks them, and the later direct image path rides on this set.
    vision_models = {
        "venice-uncensored-1-2"
      , "kimi-k3"
      , "qwen-3-8-max"
      , "z-ai-glm-5-3-flash"
      , "gemini-3-5-flash"
      , "inkling"
    }
    # Documented balance-and-limits endpoint, readable with the inference key.
    usage_url = "https://api.venice.ai/api/v1/api_keys/rate_limits"
    # Prompt caches expire after inactivity "typically 5-10 minutes" (docs);
    # the 300 s floor assumes the cache dead no later than it surely is.
    cache_ttl = 300
    # The vision model of analyze_image on this provider.
    vision_model = "venice-uncensored-1-2"

    def native_tools(self) -> list:
        """The provider tools: the vision tool, the augment set, and the
        image generation. web_search takes the harness name, so the Venice
        search replaces the DuckDuckGo default while this provider is
        active. web_scrape, parse_document, and generate_image join as
        additions; the harness web_fetch keeps its place, it reads the
        Discord file hosts with the bot token."""
        return [
            analyze_image_tool(self.vision_model)
          , _search_tool()
          , _scrape_tool()
          , _parse_tool()
          , _image_tool()
        ]

    def extra_payload(self, session_id: int) -> dict:
        # Venice appends its own system prompt unless told off; the harness
        # ships its own. prompt_cache_key routes a session to one backend.
        return {
            "venice_parameters": {"include_venice_system_prompt": False}
          , "prompt_cache_key": str(session_id)
        }

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
        text = f"available ${usd} USD, {diem} Diem, tier {tier.get('id') or 'unknown'}"
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
            epoch = payload.get("nextEpochBegins")
            text += f"; {limits}"
            if epoch:
                text += f" (epoch resets {epoch})"
        return [{
            "name": "Balance"
            , "used": None
            , "limit": None
            , "percent": None
            , "reset": None
            , "text": text
            , "exhausted": exhausted
        }]
