from urllib.parse import urlparse

import asyncio
import base64
import random
import re

import aiohttp

from ..llm_chat import ChatError
from ..tools.base import DISCORD_FILE_HOSTS, _cap
from .base import Provider, analyze_image_tool

# The limit types of the rate-limits endpoint, spelled out for the usage rows.
VENICE_LIMIT_NAMES = {"RPM": "requests/min", "RPD": "requests/day", "TPM": "tokens/min"}
# The augment endpoints are experimental and billed per request ($0.01 each).
VENICE_QUERY_MAX_CHARS = 400
VENICE_SEARCH_MAX_LIMIT = 20
# Image generation: the endpoint-wide prompt cap.
VENICE_PROMPT_MAX_CHARS = 7500
# The curated image models, ordered by cost per image: the entries run in
# descending cost, the first entry is the default model, like the first
# entry of PROVIDERS is the default provider. Each entry holds its
# capability traits (plain names) and its real cost in USD per image: the
# price of the request the tool sends — the 2K preset on the
# resolution-tier models (gpt-image-2 at 2K medium), the flat generation
# price on the rest, the default 1K tier when a model prices by tier and
# the tool sends no resolution. The comment on each entry names the
# release date (the created field of the live model list, rendered in the
# operator's timezone UTC-4). Read from the live model list
# (GET /models?type=image needs no key), 2026-09-04: image models price
# per image, not per million tokens.
VENICE_IMAGE_MODELS = {
    "recraft-v4-pro": {"traits": [], "cost": 0.29}  # released Feb 11, 2026
  , "nano-banana-2": {"traits": [], "cost": 0.14}  # released Feb 25, 2026
  , "gpt-image-2": {"traits": [], "cost": 0.13}  # released Apr 20, 2026
  , "luma-uni-1-max": {"traits": [], "cost": 0.12}  # released Jun 16, 2026
  , "seedream-v5-pro": {"traits": ["uncensored"], "cost": 0.11}  # released Jul 7, 2026
  , "grok-imagine-image-2-0": {"traits": [], "cost": 0.10}  # released Aug 10, 2026
  , "wan-2-7-pro-text-to-image": {"traits": [], "cost": 0.09375}  # released Mar 31, 2026
  , "flux-2-max": {"traits": [], "cost": 0.09}  # released Nov 25, 2025
  , "hunyuan-image-v3": {"traits": [], "cost": 0.09}  # released Feb 28, 2026
  , "qwen-image-3-pro": {"traits": ["uncensored"], "cost": 0.09}  # released Jul 15, 2026
  , "krea-v2-large": {"traits": [], "cost": 0.07}  # released May 21, 2026
  , "ideogram-v4": {"traits": [], "cost": 0.06}  # released Jun 2, 2026
  , "imagineart-1.5-pro": {"traits": [], "cost": 0.06}  # released Jan 26, 2026
  , "muse-image": {"traits": [], "cost": 0.02}  # released Sep 2, 2026
  , "z-image-turbo": {"traits": [], "cost": 0.01}  # released Dec 3, 2025
  , "lustify-v8": {"traits": ["uncensored"], "cost": 0.01}  # released Mar 29, 2026
  , "venice-sd35": {"traits": [], "cost": 0.01}  # released Mar 27, 2025
  , "wai-Illustrious": {"traits": ["uncensored"], "cost": 0.01}  # released Jan 11, 2025
  , "chroma": {"traits": ["uncensored"], "cost": 0.01}  # released Jan 29, 2026
}
# The promptCharacterLimit of each curated model (model_spec.constraints
# of the live list), the generate catalog first, the edit catalog after.
# An entry may sit above the endpoint-wide VENICE_PROMPT_MAX_CHARS: the
# model accepts the longer prompt, so its entry lifts the cap. An id
# missing here keeps the endpoint-wide fallback.
VENICE_IMAGE_PROMPT_LIMITS = {
    "recraft-v4-pro": 10000
  , "nano-banana-2": 32768
  , "gpt-image-2": 10000
  , "luma-uni-1-max": 6000
  , "seedream-v5-pro": 10000
  , "grok-imagine-image-2-0": 7500
  , "wan-2-7-pro-text-to-image": 3000
  , "flux-2-max": 3000
  , "hunyuan-image-v3": 3000
  , "qwen-image-3-pro": 10000
  , "krea-v2-large": 5000
  , "ideogram-v4": 10000
  , "imagineart-1.5-pro": 10000
  , "muse-image": 10000
  , "z-image-turbo": 7500
  , "lustify-v8": 1500
  , "venice-sd35": 1500
  , "wai-Illustrious": 1500
  , "chroma": 7500
  , "gpt-image-2-edit": 10000
  , "flux-2-max-edit": 3000
  , "nano-banana-2-edit": 32768
  , "qwen-image-3-pro-edit": 10000
  , "wan-2-7-pro-edit": 5000
  , "grok-imagine-image-2-0-edit": 7500
  , "luma-uni-1-edit": 6000
  , "seedream-v4-edit": 10000
  , "firered-image-edit": 1500
  , "muse-image-edit": 10000
}
# The sizing dialect of each curated model, an unknown id takes the ratio
# dialect: pixel models size through width and height (model_spec.constraints
# carries a widthHeightDivisor and no aspectRatios), the resolution-tier
# models carry a fixed resolution (and gpt-image-2 a quality) preset
# (constraints carries a resolutions array), the rest take the aspect ratio
# as-is. Read from the live model list, 2026-09-04. The tool description
# reports the dialect beside the traits of each model, so the agent knows
# which parameters apply.
VENICE_IMAGE_DIALECTS = {
    "recraft-v4-pro": "ratio"
  , "nano-banana-2": "resolution"
  , "gpt-image-2": "resolution"
  , "luma-uni-1-max": "ratio"
  , "grok-imagine-image-2-0": "resolution"
  , "wan-2-7-pro-text-to-image": "ratio"
  , "flux-2-max": "ratio"
  , "hunyuan-image-v3": "ratio"
  , "krea-v2-large": "ratio"
  , "seedream-v5-pro": "resolution"
  , "ideogram-v4": "ratio"
  , "imagineart-1.5-pro": "ratio"
  , "qwen-image-3-pro": "resolution"
  , "muse-image": "ratio"
  , "z-image-turbo": "pixel"
  , "lustify-v8": "pixel"
  , "venice-sd35": "pixel"
  , "wai-Illustrious": "pixel"
  , "chroma": "pixel"
}
# The presets of the resolution-tier models. 2K and medium for now, adjust
# after some use.
VENICE_IMAGE_RESOLUTION = "2K"
VENICE_IMAGE_QUALITY = "medium"
# aspect_ratio to pixels for the pixel-dialect models: sides at most 1280,
# multiples of 16 — every pixel model's widthHeightDivisor divides it (16 on
# venice-sd35 and wai-Illustrious, 8 on z-image-turbo, lustify-v8, chroma),
# so one table serves them all. About one megapixel.
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
# The curated edit models of /image/edit, cost-ordered like the generate
# catalog (descending, the first entry is the default model). The source
# is the live list under the inpaint type (GET /models?type=inpaint):
# the edit models are inpaint models, their price is the inpaint price at
# the 2K preset — the quality table wins when the model has one (2K
# medium), else the 2K resolution tier, else the flat price — mirroring
# the generate convention (the edit tool sends the same presets).
# Each entry holds its capability traits (plain names) and its real cost
# in USD per edit; the comment names the release date (the created field,
# rendered UTC-4).
VENICE_EDIT_MODELS = {
    "gpt-image-2-edit": {"traits": [], "cost": 0.14}  # released Apr 20, 2026
  , "nano-banana-2-edit": {"traits": [], "cost": 0.14}  # released Feb 25, 2026
  , "flux-2-max-edit": {"traits": [], "cost": 0.12}  # released Jan 4, 2026
  , "grok-imagine-image-2-0-edit": {"traits": [], "cost": 0.10}  # released Aug 10, 2026
  , "wan-2-7-pro-edit": {"traits": [], "cost": 0.094}  # released Apr 22, 2026
  , "qwen-image-3-pro-edit": {"traits": [], "cost": 0.09}  # released Jul 15, 2026
  , "luma-uni-1-edit": {"traits": [], "cost": 0.06}  # released Jun 16, 2026
  , "seedream-v4-edit": {"traits": ["uncensored"], "cost": 0.05}  # released Jan 3, 2026
  , "firered-image-edit": {"traits": [], "cost": 0.04}  # released Mar 24, 2026
  , "muse-image-edit": {"traits": [], "cost": 0.02}  # released Sep 2, 2026
}
# The edit models that price and render by resolution tier, and the
# subset that takes a quality preset: the edit tool sends the same 2K and
# medium constants as generate, so the catalog cost matches the bill.
VENICE_EDIT_TIER_MODELS = {
    "gpt-image-2-edit"
  , "nano-banana-2-edit"
  , "grok-imagine-image-2-0-edit"
  , "qwen-image-3-pro-edit"
}
VENICE_EDIT_QUALITY_MODELS = {
    "gpt-image-2-edit"
  , "grok-imagine-image-2-0-edit"
}
# The edit answer formats of the endpoint, mapped to file extensions.
VENICE_EDIT_FORMATS = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}
# The agent-facing label of each trait flag. The flags ride the catalog
# entries (VENICE_IMAGE_MODELS, VENICE_EDIT_MODELS). The
# copyrighted_material trait: the model refuses a prompt that names
# copyrighted material (verified live 2026-09-03 on flux-2-pro and
# grok-imagine-image — the answer is a uniform blank image, no error; both
# were replaced on 2026-09-04, so no entry carries the flag until a sweep
# re-tests their successors, the full table lives in the vault note
# Venice.AI/HTTP API.md). The uncensored trait: the live model list
# reports the model as applying minimal content-based filtering
# (model_spec.uncensored true).
VENICE_IMAGE_TRAIT_LABELS = {
    "copyrighted_material": "refuses copyrighted material"
  , "uncensored": "uncensored"
}
# The refusal asset of the endpoint: a uniform blank image that compresses
# far below any real render (exactly 1926 bytes live, while the smallest
# real render of the 2026-09-03 sweep weighed 193 KB). No image decoder
# rides the cog, so the decoded size stands in for the pixel check. The
# ceiling sits just above the observed asset.
VENICE_IMAGE_BLANK_MAX_BYTES = 2048
# The capabilities the agent can ask of the environment tool. One settled
# vocabulary: the same words in the schema enum, the catalog traits, and
# the tool answers. The code trait mirrors the optimizedForCode flag of
# the live model list.
VENICE_CHAT_CAPABILITIES = ("large context", "vision", "uncensored", "code")
# Chat presets: a short display name for the agent and the user, the model
# id behind it, an optional NSFW variant id for conversations behind the
# 18+ gate (a different model with its own capability set), the capability
# names the preset provides, and the cost scale of the preset. The cost is
# the operating cost of the model — 10x the input price plus 1x the output
# and cache-read prices, each per 1M tokens (the operating_usd_per_m column
# of scripts/list_models.py) — over the priciest catalog model (Kimi at the
# ceiling, read 2026-09-04), anchored at DeepSeek Lite = 0: a preset
# cheaper than the default carries a negative cost. The catalog order is
# preference order: the first preset that satisfies a request wins. The
# short names are the only model handle the agent ever sees.
VENICE_CHAT_PRESETS = {
    "GLM 1M": {
        "normal": "z-ai-glm-5-3"
      , "traits": ["large context", "code"]
      , "cost": 0.39
    }
  , "GLM Lite": {
        "normal": "zai-org-glm-4.7-flash"
      , "traits": []
      , "cost": -0.02
      , "nsfw": "olafangensan-glm-4.7-flash-heretic"
      , "nsfw_traits": []
    }
  , "GLM Vision": {
        "normal": "z-ai-glm-5-3-flash"
      , "traits": ["large context", "vision", "code"]
      , "cost": 0.0
    }
  , "Kimi": {
        "normal": "kimi-k3"
      , "traits": ["large context", "vision", "code"]
      , "cost": 1.0
    }
  , "Venice Uncensored": {
        "normal": "venice-uncensored-1-2"
      , "traits": ["vision", "uncensored"]
      , "cost": 0.01
    }
  , "Inkling": {
        "normal": "inkling"
      , "traits": ["vision", "code"]
      , "cost": 0.29
    }
  , "DeepSeek Lite": {
        "normal": "deepseek-v4-flash-0731"
      , "traits": ["large context", "code"]
      , "cost": 0.0
    }
  , "DeepSeek Pro": {
        "normal": "deepseek-v4-pro"
      , "traits": ["large context", "code"]
      , "cost": 0.33
    }
  , "Gemma": {
        "normal": "google-gemma-4-31b-it"
      , "traits": ["vision"]
      , "cost": 0.0
      , "nsfw": "gemma-4-uncensored"
      , "nsfw_traits": ["vision"]
    }
  , "Aion Mini": {
        "normal": "aion-labs-aion-3-0-mini"
      , "traits": ["uncensored"]
      , "cost": 0.16
    }
  , "Aion": {
        "normal": "aion-labs-aion-3-0"
      , "traits": ["uncensored"]
      , "cost": 0.80
    }
  , "Qwen Lite": {
        "normal": "qwen3-6-35b-a3b"
      , "traits": ["vision", "code", "uncensored"]
      , "cost": 0.0
    }
  , "Qwen": {
        "normal": "qwen-3-8-27b"
      , "traits": ["vision", "code", "uncensored"]
      , "cost": 0.10
    }
}


def _header_flag(headers, name: str) -> str:
    """The yes/no/unknown text of a boolean response header."""
    value = headers.get(name) if headers is not None else None
    if value is None:
        return "unknown"
    return "yes" if str(value).strip().lower() == "true" else "no"


def _moderation_status(headers) -> tuple[str, str]:
    """The (violation flag, [venice] status line) of an image answer. A flag
    reaches the model only when it reads yes: a clean answer carries no
    line at all. The generate and the edit endpoint answer the same
    headers."""
    violation = _header_flag(headers, "x-venice-is-content-violation")
    raised = [
        f"{label}: {flag}"
        for label, flag in (
            ("content violation", violation)
            , ("blurred", _header_flag(headers, "x-venice-is-blurred"))
        )
        if flag == "yes"
    ]
    return violation, f"[venice] {', '.join(raised)}" if raised else ""


def _search_tool() -> dict:
    """The augment search as a native tool, under the harness web_search name."""

    async def handler(arguments, call_api, fetch_url=None, api_post=None, send_file=None, channel_nsfw=None, set_conversation_model=None):
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
            data, _headers = await api_post("/augment/search", json_body=body)
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

    async def handler(arguments, call_api, fetch_url=None, api_post=None, send_file=None, channel_nsfw=None, set_conversation_model=None):
        url = str(arguments.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            return "Error: the url must be an http(s) URL."
        try:
            data, _headers = await api_post("/augment/scrape", json_body={"url": url})
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

    async def handler(arguments, call_api, fetch_url=None, api_post=None, send_file=None, channel_nsfw=None, set_conversation_model=None):
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
            data, _headers = await api_post("/augment/text-parser", data=form)
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
    age-restricted channel. The tool result appends the moderation flags of
    the answer as a [venice] status line, and only a flag that reads yes
    appears: a clean answer carries no line.
    A refusal needs both marks, a tiny image and the violation flag: only
    then the tool answers with an error and posts nothing. Without the
    flag, a small image posts as content."""

    async def handler(arguments, call_api, fetch_url=None, api_post=None, send_file=None, channel_nsfw=None, set_conversation_model=None):
        prompt = str(arguments.get("prompt") or "").strip()
        if not prompt:
            return "Error: the prompt must be a non-empty string."
        model = str(arguments.get("model") or "").strip() or next(iter(VENICE_IMAGE_MODELS))
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
            data, headers = await api_post("/image/generate", json_body=body)
        except ChatError as e:
            return f"Error: the image generation failed: {e}"
        # The moderation signals of the endpoint: a content violation is the
        # documented face of the silent refusal (a blank image with no error).
        violation, status = _moderation_status(headers)
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
        if len(raw) <= VENICE_IMAGE_BLANK_MAX_BYTES and violation == "yes":
            # A refusal needs both marks: the tiny image and the violation
            # flag. Without the flag, a small image posts: the agent assumes
            # it holds content.
            return (
                f"Error: the generation was refused: the answer is a blank image of {len(raw)} bytes "
                f"and the endpoint flags a content violation. {status}. Nothing was posted. "
                "Describe the subject instead of naming it, or pick a model without the refusal flag."
            )
        # The format stays the endpoint default (webp). The id names the file:
        # each image of a conversation lands under its own name.
        name = re.sub(r"[\s/\\]+", "-", str(data.get("id") or "venice-image"))[:100]
        sent = await send_file(f"{name}.webp", raw)
        if status:
            return f"{sent}\n{status}"
        return sent

    model_notes = []
    for image_model, entry in VENICE_IMAGE_MODELS.items():
        labels = []
        dialect = VENICE_IMAGE_DIALECTS.get(image_model)
        if dialect:
            labels.append(dialect)
        labels.extend(VENICE_IMAGE_TRAIT_LABELS[key] for key in entry["traits"])
        model_notes.append(f"{image_model} ({', '.join(labels)})" if labels else image_model)
    return {
        "name": "generate_image"
        , "description": (
            "Generate one image from a text prompt through Venice and post it to the conversation. "
            f"Known models: {', '.join(model_notes)}. "
            "Every model sizes through aspect_ratio. A pixel model also takes cfg_scale, "
            "and the tool maps the ratio to pixels for it. "
            "A resolution model renders at a fixed resolution and quality preset. "
            "A model marked as refusing copyrighted material does exactly that: "
            "describe the subject instead of naming it, or pick another model."
        )
        , "parameters": {
            "type": "object"
            , "properties": {
                "prompt": {"type": "string", "description": "What to draw."}
                , "model": {"type": "string", "description": f"The image model. Default {next(iter(VENICE_IMAGE_MODELS))}."}
                , "aspect_ratio": {"type": "string", "description": "The aspect ratio of the image, for example 1:1, 16:9, or 9:16."}
                , "negative_prompt": {"type": "string", "description": "What to keep out of the image."}
                , "cfg_scale": {"type": "number", "description": "How strictly the pixel model follows the prompt, over 0 up to 20. Default of the endpoint."}
            }
            , "required": ["prompt"]
        }
        , "handler": handler
    }


def _edit_tool() -> dict:
    """The image edit endpoint as a native tool: one edited image, posted to
    the conversation. The endpoint always answers in binary (the JSON mode
    of generate does not exist here), so the call rides provider_post with
    binary=True and the answer format names the file extension. The input
    image comes as an http(s) URL: an attachment of the conversation or
    the URL the posting result of generate_image names. A Discord file
    host URL downloads with the bot token and rides the JSON body as
    base64; a foreign URL passes to the endpoint as-is. The agent
    controls the image, the prompt, the model, and the aspect ratio.
    safe_mode drops only on an age-restricted channel. The moderation
    flags report like generate_image: only a flag that reads yes appears,
    and a blank refusal needs both marks."""

    async def handler(arguments, call_api, fetch_url=None, api_post=None, send_file=None, channel_nsfw=None, set_conversation_model=None):
        image = str(arguments.get("image") or "").strip()
        if not image.startswith(("http://", "https://")):
            return "Error: the image must be the http(s) URL of the picture to edit."
        prompt = str(arguments.get("prompt") or "").strip()
        if not prompt:
            return "Error: the prompt must say what to change."
        model = str(arguments.get("model") or "").strip() or next(iter(VENICE_EDIT_MODELS))
        prompt_limit = VENICE_IMAGE_PROMPT_LIMITS.get(model, VENICE_PROMPT_MAX_CHARS)
        if len(prompt) > prompt_limit:
            return f"Error: the prompt is over the {prompt_limit}-character limit of the model {model}."
        body = {"model": model, "prompt": prompt}
        if urlparse(image).netloc.lower() in DISCORD_FILE_HOSTS:
            # The Discord file hosts need an authorized download: the image
            # rides the body as base64 instead of the URL.
            if fetch_url is None:
                return "Error: the Discord download is not available here."
            fetched = await fetch_url(image)
            if fetched is None:
                return "Error: the download of the Discord file failed."
            body["image"] = base64.b64encode(fetched[0]).decode("ascii")
        else:
            body["image"] = image
        aspect_ratio = str(arguments.get("aspect_ratio") or "").strip()
        if aspect_ratio and aspect_ratio != "auto":
            # auto is the endpoint default: the edit keeps the input shape.
            body["aspect_ratio"] = aspect_ratio
        if model in VENICE_EDIT_TIER_MODELS:
            # The tier models render at the 2K preset, the quality models
            # at medium — the same convention as generate, so the catalog
            # cost matches the bill (gpt-image-2-edit defaults to high,
            # which bills far over medium).
            body["resolution"] = VENICE_IMAGE_RESOLUTION
            if model in VENICE_EDIT_QUALITY_MODELS:
                body["quality"] = VENICE_IMAGE_QUALITY
        if channel_nsfw is not None and await channel_nsfw():
            # The endpoint default true blurs adult content: it drops only
            # where Discord itself gates the channel behind 18+.
            body["safe_mode"] = False
        try:
            data, headers = await api_post("/image/edit", json_body=body, binary=True)
        except ChatError as e:
            return f"Error: the image edit failed: {e}"
        if not isinstance(data, (bytes, bytearray)) or not data:
            return "Error: the image edit returned no image."
        violation, status = _moderation_status(headers)
        if len(data) <= VENICE_IMAGE_BLANK_MAX_BYTES and violation == "yes":
            # A refusal needs both marks: the tiny image and the violation
            # flag, the same gate as generate.
            return (
                f"Error: the edit was refused: the answer is a blank image of {len(data)} bytes "
                f"and the endpoint flags a content violation. {status}. Nothing was posted. "
                "Describe the subject instead of naming it, or pick a model without the refusal flag."
            )
        if send_file is None:
            return "Error: the image posting is not available here."
        # The binary answer carries no generation id: the answer format
        # (a content-type header) names the extension, the model and a
        # random token name the file.
        content_type = str(headers.get("content-type") or "").split(";")[0].strip().lower()
        extension = VENICE_EDIT_FORMATS.get(content_type, "png")
        name = f"{model}-{random.randint(0, 0xFFFF):04x}.{extension}"
        sent = await send_file(name, bytes(data))
        if status:
            return f"{sent}\n{status}"
        return sent

    model_notes = []
    for edit_model, entry in VENICE_EDIT_MODELS.items():
        labels = [VENICE_IMAGE_TRAIT_LABELS[key] for key in entry["traits"]]
        model_notes.append(f"{edit_model} ({', '.join(labels)})" if labels else edit_model)
    return {
        "name": "edit_image"
        , "description": (
            "Edit one image through Venice and post the result to the conversation. "
            "The image is the http(s) URL of the picture: an attachment of a message, "
            "or the URL the generate_image result names. "
            f"Known models: {', '.join(model_notes)}. "
            "Every model takes the prompt and the aspect ratio; auto (the default) keeps the input shape. "
            "A model marked as refusing copyrighted material does exactly that."
        )
        , "parameters": {
            "type": "object"
            , "properties": {
                "image": {"type": "string", "description": "The http(s) URL of the picture to edit."}
                , "prompt": {"type": "string", "description": "What to change in the picture."}
                , "model": {"type": "string", "description": f"The edit model. Default {next(iter(VENICE_EDIT_MODELS))}."}
                , "aspect_ratio": {"type": "string", "description": "The aspect ratio of the result, for example 1:1, 16:9, or 9:16. Default auto."}
            }
            , "required": ["image", "prompt"]
        }
        , "handler": handler
    }


def _environment_tool() -> dict:
    """The environment tool: the agent states the capabilities the current
    task needs, the tool picks the chat preset that provides them and
    switches the conversation to it. A task that asks for the uncensored
    capability loads the NSFW variant of the preset, and only in a
    conversation behind the 18+ gate. The agent never sees a model id: the
    short preset name is its only handle."""

    async def handler(arguments, call_api, fetch_url=None, api_post=None, send_file=None, channel_nsfw=None, set_conversation_model=None):
        if set_conversation_model is None:
            return "Error: the environment configuration is not available here."
        raw = arguments.get("capabilities")
        items = [raw] if isinstance(raw, str) else list(raw) if isinstance(raw, list) else []
        requested = []
        for item in items:
            text = str(item).strip().lower()
            if text and text not in requested:
                requested.append(text)
        if not requested:
            return 'Error: state at least one capability, or "default" to restore the default environment.'
        if "default" in requested:
            if len(requested) > 1:
                return 'Error: "default" accepts no other capability.'
            await set_conversation_model(None)
            return "The environment is back to the default configuration. The change answers the next message."
        unknown = [item for item in requested if item not in VENICE_CHAT_CAPABILITIES]
        if unknown:
            return (
                f"Error: unknown capability: {', '.join(unknown)}. "
                f"Known capabilities: {', '.join(VENICE_CHAT_CAPABILITIES)}."
            )
        adult = "uncensored" in requested
        if adult and (channel_nsfw is None or not await channel_nsfw()):
            return "Error: the uncensored capability needs a conversation behind the 18+ gate."
        for name, preset in VENICE_CHAT_PRESETS.items():
            remaining = [trait for trait in requested if trait != "uncensored"]
            if adult:
                # The 18+ variant is the uncensored build of the preset; the
                # other capabilities must hold for the variant itself. A
                # preset without a variant can still be uncensored by
                # itself, its normal id is the uncensored build. The stored
                # value is the preset name: the request-time resolution
                # picks the variant by the gate of the moment.
                variant = preset.get("nsfw")
                if variant is not None and all(trait in preset.get("nsfw_traits", ()) for trait in remaining):
                    await set_conversation_model(name)
                    granted = ", ".join(sorted(requested))
                    return (
                        f"The environment now provides: {granted}. Active preset: {name} (18+ variant). "
                        "The change answers the next message."
                    )
                if variant is None and "uncensored" in preset.get("traits", ()) and all(trait in preset["traits"] for trait in remaining):
                    await set_conversation_model(name)
                    granted = ", ".join(sorted(requested))
                    return (
                        f"The environment now provides: {granted}. Active preset: {name}. "
                        "The change answers the next message."
                    )
            else:
                traits = preset.get("traits", ())
                if all(trait in traits for trait in requested):
                    await set_conversation_model(name)
                    granted = ", ".join(sorted(requested))
                    return (
                        f"The environment now provides: {granted}. Active preset: {name}. "
                        "The change answers the next message."
                    )
        menu = "; ".join(preset_menu_line(name, preset) for name, preset in VENICE_CHAT_PRESETS.items())
        return f"Error: no environment preset provides: {', '.join(sorted(requested))}. Available: {menu}."

    return {
        "name": "configure_environment"
        , "description": (
            "Configure the environment of this conversation for the current task. "
            "State the capabilities the task needs. The change answers the next message. "
            f"Capabilities: {', '.join(VENICE_CHAT_CAPABILITIES)}, and \"default\" to restore the default environment. "
            "The uncensored capability needs a conversation behind the 18+ gate."
        )
        , "parameters": {
            "type": "object"
            , "properties": {
                "capabilities": {
                    "type": "array"
                    , "items": {"type": "string", "enum": ["default", *VENICE_CHAT_CAPABILITIES]}
                    , "description": "The capabilities the current task needs."
                }
            }
            , "required": ["capabilities"]
        }
        , "handler": handler
    }


def preset_menu_line(name: str, preset: dict) -> str:
    """One catalog entry for the menus of the agent and the user."""
    traits = ", ".join(sorted(preset.get("traits", ()))) or "plain"
    suffix = "; 18+ variant available" if preset.get("nsfw") else ""
    return f"{name} ({traits}{suffix})"


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
      , "qwen3-6-35b-a3b"
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
      , "qwen3-6-35b-a3b": 256_000
      , "qwen-3-8-27b": 262_144
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
      , "google-gemma-4-31b-it"
      , "gemma-4-uncensored"
      , "qwen3-6-35b-a3b"
      , "qwen-3-8-27b"
    }
    # Documented balance-and-limits endpoint, readable with the inference key.
    usage_url = "https://api.venice.ai/api/v1/api_keys/rate_limits"
    # Prompt caches expire after inactivity "typically 5-10 minutes" (docs);
    # the 300 s floor assumes the cache dead no later than it surely is.
    cache_ttl = 300
    # The vision model of analyze_image on this provider.
    vision_model = "venice-uncensored-1-2"

    def native_tools(self) -> list:
        """The provider tools: the vision tool, the augment set, the image
        generation and edit, and the environment tool. web_search takes the
        harness name, so the Venice search replaces the DuckDuckGo default
        while this provider is active. web_scrape, parse_document,
        generate_image, edit_image, and configure_environment join as
        additions; the harness web_fetch keeps its place, it reads the
        Discord file hosts with the bot
        token."""
        return [
            analyze_image_tool(self.vision_model)
          , _search_tool()
          , _scrape_tool()
          , _parse_tool()
          , _image_tool()
          , _edit_tool()
          , _environment_tool()
        ]

    def preset_name(self, model_id: str) -> str | None:
        """The short preset name of a chat model id, None when unknown."""
        for name, preset in VENICE_CHAT_PRESETS.items():
            if model_id in (preset.get("normal"), preset.get("nsfw")):
                return name
        return None

    def request_model(self, name: str, nsfw: bool = False) -> str:
        """Resolve a model string for one request: a preset name (any
        casing) maps to its NSFW variant behind the 18+ gate, to its
        normal id anywhere else. A raw id passes through unchanged."""
        for catalog_name, preset in VENICE_CHAT_PRESETS.items():
            if catalog_name.lower() == str(name).strip().lower():
                if nsfw and preset.get("nsfw"):
                    return preset["nsfw"]
                return preset["normal"]
        return name

    def resolve_model(self, name: str) -> str:
        """Map a short preset name (any casing) to its model id, pass
        anything else through unchanged."""
        return self.request_model(name, nsfw=False)

    def context_length(self, model_name: str) -> int | None:
        """The context size of a model in tokens, None when unknown. A
        preset name resolves to its normal variant first."""
        return self.context_lengths.get(self.request_model(model_name, nsfw=False))

    def cost_of(self, data: dict) -> float:
        """The Venice request cost: the answer carries it split by billing
        currency. The USD part is the metric, the DIEM part fills in when
        the plan bills that way."""
        cost = data.get("cost") or {}
        return float(cost.get("usd") or cost.get("diem") or 0)

    def default_model(self) -> str:
        """The default of a cleared configuration: the preset NAME of the
        first model, so the default resolves per request like any preset —
        the 18+ variant behind the gate, the normal id elsewhere."""
        for name, preset in VENICE_CHAT_PRESETS.items():
            if preset.get("normal") == self.models[0]:
                return name
        return self.models[0]

    def preset_fallback(self, model: str) -> str | None:
        """The preset one step up in cost from the preset holding the
        model (either variant), cycling to the cheapest at the ceiling.
        The cost of a preset is its cost property, 0 when the entry names
        none. None when the model sits in no preset."""
        ranked = []
        current = None
        for name, preset in VENICE_CHAT_PRESETS.items():
            entry = (preset.get("cost", 0.0), name)
            ranked.append(entry)
            if model in (preset.get("normal"), preset.get("nsfw")):
                current = entry
        if current is None:
            return None
        ranked.sort()
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

    async def fetch_usage(self, session, api_key: str):
        """The rate-limits answer plus the billing allowance. /billing/balance
        needs an ADMIN key, so an inference key keeps the plain row without
        an error. The billing answer says the plan currency, and the
        allowance of the plan: the remaining credit and, when the plan
        reports one, the reserve it draws from."""
        rows, error = await super().fetch_usage(session, api_key)
        if error or not rows:
            return rows, error
        try:
            async with session.get(
                "https://api.venice.ai/api/v1/billing/balance"
                , headers={"Authorization": f"Bearer {api_key}"}
                , timeout=aiohttp.ClientTimeout(total=15),
            ) as response:
                if response.status != 200:
                    # An inference key cannot read the billing endpoint.
                    return rows, error
                try:
                    billing = await response.json(content_type=None)
                except Exception:
                    return rows, error
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return rows, error
        if not isinstance(billing, dict):
            return rows, error
        parts = []
        currency = billing.get("consumptionCurrency")
        if currency:
            parts.append(f"plan currency {currency}")
        balances = billing.get("balances") or {}
        remaining = balances.get("diem")
        reserve = billing.get("diemEpochAllocation")
        if isinstance(remaining, (int, float)) and isinstance(reserve, (int, float)) and reserve:
            parts.append(f"allowance {remaining:g} of {reserve:g} points left this epoch")
        elif isinstance(balances.get("usd"), (int, float)):
            parts.append(f"credit balance ${balances['usd']:g}")
        if billing.get("canConsume") is False:
            rows[0]["exhausted"] = True
        if parts:
            rows[0]["text"] += f"; {'; '.join(parts)}"
        return rows, error

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
