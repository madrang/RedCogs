# The native tools of the Venice provider:
#    the augment set
#  , the image generation, the image edit, the background removal, the song generation
#  , and the environment tool
#  , plus the shared model-parameter builders and the preset menu line.

import base64
import random
import re
from urllib.parse import urlparse

import aiohttp

from ...llm_chat import ChatError
from ...tools.base import DISCORD_FILE_HOSTS, _cap
from .audio import quote_song
from .decisions import select_preset
from .resources import EDITS_URI, IMAGES_URI, MUSIC_URI
from .catalog import (
    VENICE_QUERY_MAX_CHARS
  , VENICE_SEARCH_MAX_LIMIT
  , VENICE_PROMPT_MAX_CHARS
  , VENICE_IMAGE_MODELS
  , VENICE_IMAGE_PROMPT_LIMITS
  , VENICE_IMAGE_DIALECTS
  , VENICE_IMAGE_RESOLUTION
  , VENICE_IMAGE_QUALITY
  , VENICE_RENDER_TIMEOUT
  , VENICE_PIXEL_RATIOS
  , VENICE_SEED_MAX
  , VENICE_BACKGROUND_COST
  , VENICE_EDIT_MODELS
  , VENICE_EDIT_TIER_MODELS
  , VENICE_EDIT_MODEL_RESOLUTIONS
  , VENICE_EDIT_QUALITY_MODELS
  , VENICE_EDIT_FORMATS
  , VENICE_IMAGE_TRAIT_LABELS
  , VENICE_MUSIC_MODELS
  , VENICE_MUSIC_CONSTRAINTS
  , VENICE_CHAT_CAPABILITIES
  , VENICE_CHAT_PRESETS
  , VENICE_CREDIT_ROUTING_FLOOR
  , VENICE_ROUTING_TRAIT_AT
)
from .moderation import _moderation_status, _refusal_error


def _search_tool() -> dict:
    """The augment search as a native tool, under the harness web_search name."""

    async def handler(arguments, engine):
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
            data, _headers = await engine.api_post("/augment/search", json_body=body)
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
                , "limit": {"type": "integer", "default": 10, "description": f"How many results to return, 1 to {VENICE_SEARCH_MAX_LIMIT}."}
            }
            , "required": ["query"]
        }
        , "handler": handler
    }


def _scrape_tool() -> dict:
    """The augment scrape as a native tool: one page as markdown."""

    async def handler(arguments, engine):
        url = str(arguments.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            return "Error: the url must be an http(s) URL."
        try:
            data, _headers = await engine.api_post("/augment/scrape", json_body={"url": url})
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

    async def handler(arguments, engine):
        url = str(arguments.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            return "Error: the url must be the http(s) URL of a document."
        if engine.fetch_url is None:
            return "Error: the document download is not available here."
        fetched = await engine.fetch_url(url)
        if fetched is None:
            return "Error: the download of the document failed."
        body, content_type = fetched
        name = urlparse(url).path.rsplit("/", 1)[-1][:100] or "document"
        form = aiohttp.FormData()
        form.add_field("file", body, filename=name, content_type=content_type)
        try:
            data, _headers = await engine.api_post("/augment/text-parser", data=form)
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


async def _report_cost(engine, cost: float) -> None:
    """Report one render price to the engine cost counter of the reply, when
    the context carries the reporter (a stand-in engine may lack it). The
    report rides every answered render: a refusal bills the same."""
    reporter = getattr(engine, "add_media_cost", None)
    if reporter is not None:
        await reporter(cost)


def _model_property(catalog: dict) -> dict:
    """The model parameter of an image tool: the enum of the preset names (the fixed set of accepted values, the JSON Schema way — the handler
       refuses anything outside it) and a description that names the dialect groups and the trait groups of the catalog.
       The default model rides the schema default field: the first catalog entry, the one a blank ask takes.
    """
    groups: dict = {}
    dialect_groups: dict = {}
    for preset_name, entry in catalog.items():
        for trait in entry["traits"]:
            groups.setdefault(trait, []).append(preset_name)
        dialect = VENICE_IMAGE_DIALECTS.get(entry["model"])
        if dialect and dialect != "ratio":
            dialect_groups.setdefault(dialect, []).append(preset_name)
    notes = [f"{dialect.capitalize()} models: {', '.join(names)}." for dialect, names in dialect_groups.items()]
    notes.extend(f"{VENICE_IMAGE_TRAIT_LABELS[trait].capitalize()}: {', '.join(names)}." for trait, names in groups.items())
    compositing = [preset_name for preset_name, entry in catalog.items() if "max_images" in entry]
    if compositing:
        # Only the edit catalog carries the key: a model without it composites
        # no extra image, so the group names the models that take them.
        notes.append(f"Composites extra images: {', '.join(compositing)}.")
    description = "The model preset name."
    if notes:
        description += " " + " ".join(notes)
    return {"type": "string", "enum": list(catalog), "default": next(iter(catalog)), "description": description}


def _image_tool(ceiling: float | None = None) -> dict:
    """The image generation endpoint as a native tool: one image, posted to the conversation.
       The endpoint answers in base64 JSON (return_binary stays false): the binary mode does not fit provider_post.
       The agent controls the prompt, the model, the aspect ratio, the negative prompt, and the cfg scale.
       The resolution, the quality, the watermark, the exif metadata, and the seed are preset here; safe_mode drops only on an age-restricted channel.
       The credit ratio bounds the offered models at the context start: the priciest entries leave the descriptors first as the balance runs down, and the list stays frozen for the context lifetime so the prompt cache holds.
       The handler checks the live ceiling at call time: a preset over it answers that it still exists but sits temporarily disabled for the low credit allowance, and a remembered preset outside the frozen list resolves through the full catalog.
       Every answered render reports the catalog price of its preset into the cost total of the reply: a refusal bills the same, an HTTP failure bills nothing.
       The tool result appends the moderation flags of the answer as a [venice] status line, and only a flag that reads yes appears: a clean answer carries no line.
       A refusal needs both marks, a blank image and the violation flag: only then the tool answers with an error and posts nothing.
       Without the flag, a small image posts as content.
    """
    # The ceiling of the credit ratio bounds the catalog: the entries over
    # it leave the enum and the handler walk. None serves the full catalog.
    catalog = VENICE_IMAGE_MODELS
    if ceiling is not None:
        catalog = {name: entry for name, entry in VENICE_IMAGE_MODELS.items() if entry["cost"] <= ceiling}

    async def handler(arguments, engine):
        prompt = str(arguments.get("prompt") or "").strip()
        if not prompt:
            return "Error: the prompt must be a non-empty string."
        asked = str(arguments.get("model") or "").strip()
        matched = None
        for preset_name, entry in catalog.items():
            # The preset name is the only handle, in any casing; a blank
            # ask takes the first entry (the default).
            if not asked or preset_name.lower() == asked.lower():
                matched = (preset_name, entry)
                break
        if matched is None and asked:
            # A remembered preset may sit outside the frozen descriptors:
            # the full catalog resolves it, the live ceiling decides.
            for preset_name, entry in VENICE_IMAGE_MODELS.items():
                if preset_name.lower() == asked.lower():
                    matched = (preset_name, entry)
                    break
        if matched is None:
            # Only the curated catalog, by its preset names: the dialect,
            # the prompt limit, and the cost of an unknown model are all
            # unverified (a raw model id is not a name).
            return (
                f"Error: unknown image model {asked}. "
                f"Valid models: {', '.join(catalog)}. "
                f"The price and the live enable state of every model sit in the harness resource {IMAGES_URI} (read_resource, server \"harness\")."
            )
        preset_name, entry = matched
        # The live ceiling disables a preset without touching the frozen
        # descriptors: the prompt cache holds, the refusal carries the news.
        getter = getattr(engine, "media_ceiling", None)
        live = await getter("image") if getter is not None else None
        if live is not None and entry["cost"] > live:
            allowed = ", ".join(name for name, other in VENICE_IMAGE_MODELS.items() if other["cost"] <= live)
            return (
                f"Error: the image preset {preset_name} still exists but sits temporarily disabled: "
                f"the bundled credit allowance runs low. Valid models: {allowed}. "
                f"The live ceiling and the price of every model sit in the harness resource {IMAGES_URI} (read_resource, server \"harness\")."
            )
        model = entry["model"]
        prompt_limit = VENICE_IMAGE_PROMPT_LIMITS.get(model, VENICE_PROMPT_MAX_CHARS)
        if len(prompt) > prompt_limit:
            return f"Error: the prompt is over the {prompt_limit}-character limit of the model {preset_name}."
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
            if model == "gpt-image-2-5-sunburst":
                # gpt-image-2-5-sunburst is the only curated model with a
                # quality field, and its default high bills high.
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
        if engine.channel_nsfw is not None and await engine.channel_nsfw():
            # The endpoint default true blurs adult content: it drops only
            # where Discord itself gates the channel behind 18+.
            body["safe_mode"] = False
        try:
            data, headers = await engine.api_post("/image/generate", json_body=body, timeout=VENICE_RENDER_TIMEOUT)
        except ChatError as e:
            return f"Error: the image generation failed: {e}"
        # The endpoint answered, so the render billed the catalog price of
        # its preset: a refusal bills the same, the report precedes every
        # check of the answer.
        await _report_cost(engine, entry["cost"])
        # The moderation signals of the endpoint: a content violation is the
        # documented face of the silent refusal (a blank image with no error).
        violation, status = _moderation_status(headers)
        images = data.get("images") or []
        if not images:
            return "Error: the image generation returned no image."
        if engine.send_file is None:
            return "Error: the image posting is not available here."
        try:
            raw = base64.b64decode(images[0])
        except (ValueError, TypeError):
            return "Error: the image generation returned unreadable image data."
        if not raw:
            return "Error: the image generation returned an empty image."
        refused = _refusal_error("generation", raw, violation, status)
        if refused is not None:
            return refused
        # The format stays the endpoint default (webp). The id names the file:
        # each image of a conversation lands under its own name.
        name = re.sub(r"[\s/\\]+", "-", str(data.get("id") or "venice-image"))[:100]
        sent = await engine.send_file(f"{name}.webp", raw)
        if status:
            return f"{sent}\n{status}"
        return sent

    return {
        "name": "generate_image"
        # The usage counter a successful generation increments (the engine
        # counts the call into the scope stats).
      , "media": "images"
        , "description": (
            "Generate one image from a text prompt through Venice. "
            "The result joins the current message as an attachment, and the tool answer names the posted file. "
            "A resolution model renders at a fixed resolution and quality preset. "
            "A model that refuses copyrighted material says so in the model field: describe the subject instead of naming it, or pick another model."
        )
        , "parameters": {
            "type": "object"
            , "properties": {
                "prompt": {"type": "string", "description": "What to draw."}
                , "model": _model_property(catalog)
                , "aspect_ratio": {"type": "string", "description": "The aspect ratio of the image, for example 1:1, 16:9, or 9:16. The tool maps it to pixels for a pixel model."}
                , "negative_prompt": {"type": "string", "description": "What to keep out of the image."}
                , "cfg_scale": {"type": "number", "description": "How strictly a pixel model follows the prompt. A number over 0 and at most 20. Omit it for the endpoint default."}
            }
            , "required": ["prompt"]
        }
        , "handler": handler
    }


def _edit_tool(ceiling: float | None = None) -> dict:
    """The image edit endpoints as a native tool: one edited image, posted to the conversation.
       The endpoints always answer in binary (the JSON mode of generate does not exist here),
       so the call rides provider_post with binary=True and the answer format names the file extension.
       The input images come as http(s) URLs: attachments of the conversation or the URLs the posting result of generate_image names.
       A Discord file host URL downloads with the bot token and rides the JSON body as base64; a foreign URL passes to the endpoint as-is.
       One input image rides /image/edit. Extra images ride /image/multi-edit: the first image stays the base, the rest work as layers or masks.
       The multi-edit endpoint names its model field modelId, and only a compositing model takes extra images (the catalog max_images key).
       The agent controls the image, the extra images, the prompt, the model, and the aspect ratio.
       The credit ratio bounds the offered models at the context start like generate_image, the handler checks the live ceiling at call time.
       Every answered edit reports the catalog price of its preset into the cost total of the reply like generate_image: the extra images of a multi-edit bill beyond the price.
       safe_mode drops only on an age-restricted channel.
       The moderation flags report like generate_image: only a flag that reads yes appears, and a blank refusal needs both marks.
    """
    # The ceiling of the credit ratio bounds the catalog: the entries over
    # it leave the enum and the handler walk. None serves the full catalog.
    catalog = VENICE_EDIT_MODELS
    if ceiling is not None:
        catalog = {name: entry for name, entry in VENICE_EDIT_MODELS.items() if entry["cost"] <= ceiling}

    async def handler(arguments, engine):
        image = str(arguments.get("image") or "").strip()
        if not image.startswith(("http://", "https://")):
            return "Error: the image must be the http(s) URL of the picture to edit."
        extras: list[str] = []
        raw_extras = arguments.get("extra_images")
        if raw_extras is not None:
            if not isinstance(raw_extras, list):
                return "Error: extra_images must be a list of http(s) URLs."
            for item in raw_extras:
                extra = str(item or "").strip()
                if not extra.startswith(("http://", "https://")):
                    return "Error: every extra image must be an http(s) URL."
                extras.append(extra)
        prompt = str(arguments.get("prompt") or "").strip()
        if not prompt:
            return "Error: the prompt must say what to change."
        asked = str(arguments.get("model") or "").strip()
        matched = None
        for preset_name, entry in catalog.items():
            # The preset name is the only handle, in any casing; a blank
            # ask takes the first entry (the default).
            if not asked or preset_name.lower() == asked.lower():
                matched = (preset_name, entry)
                break
        if matched is None and asked:
            # A remembered preset may sit outside the frozen descriptors:
            # the full catalog resolves it, the live ceiling decides.
            for preset_name, entry in VENICE_EDIT_MODELS.items():
                if preset_name.lower() == asked.lower():
                    matched = (preset_name, entry)
                    break
        if matched is None:
            # Only the curated catalog, by its preset names: a generate
            # model here edits through an untested path at an unknown
            # price (a raw model id is not a name).
            return (
                f"Error: unknown edit model {asked}. "
                f"Valid models: {', '.join(catalog)}. "
                f"The price and the live enable state of every model sit in the harness resource {EDITS_URI} (read_resource, server \"harness\")."
            )
        preset_name, entry = matched
        # The live ceiling disables a preset without touching the frozen
        # descriptors: the prompt cache holds, the refusal carries the news.
        getter = getattr(engine, "media_ceiling", None)
        live = await getter("edit") if getter is not None else None
        if live is not None and entry["cost"] > live:
            allowed = ", ".join(name for name, other in VENICE_EDIT_MODELS.items() if other["cost"] <= live)
            return (
                f"Error: the edit preset {preset_name} still exists but sits temporarily disabled: "
                f"the bundled credit allowance runs low. Valid models: {allowed}. "
                f"The live ceiling and the price of every model sit in the harness resource {EDITS_URI} (read_resource, server \"harness\")."
            )
        model = entry["model"]
        prompt_limit = VENICE_IMAGE_PROMPT_LIMITS.get(model, VENICE_PROMPT_MAX_CHARS)
        if len(prompt) > prompt_limit:
            return f"Error: the prompt is over the {prompt_limit}-character limit of the model {preset_name}."
        if extras:
            if "max_images" not in entry:
                compositing = ", ".join(name for name, other in catalog.items() if "max_images" in other)
                return (
                    f"Error: the model {preset_name} composites no extra image. "
                    f"Models that composite: {compositing}."
                )
            max_images = entry["max_images"]
            if max_images is not None and 1 + len(extras) > max_images:
                return f"Error: the model {preset_name} takes at most {max_images} input images ({1 + len(extras)} asked)."

        async def slot(url):
            # Every image slot takes the same form: a Discord download as
            # base64, a foreign URL as-is.
            if urlparse(url).netloc.lower() not in DISCORD_FILE_HOSTS:
                return url, None
            # The Discord file hosts need an authorized download.
            if engine.fetch_url is None:
                return None, "Error: the Discord download is not available here."
            fetched = await engine.fetch_url(url)
            if fetched is None:
                return None, f"Error: the download of the Discord file failed: {url}"
            return base64.b64encode(fetched[0]).decode("ascii"), None

        if not extras:
            resolved, failure = await slot(image)
            if failure:
                return failure
            body = {"model": model, "prompt": prompt, "image": resolved}
            path = "/image/edit"
        else:
            images = []
            for url in [image, *extras]:
                resolved, failure = await slot(url)
                if failure:
                    return failure
                images.append(resolved)
            body = {"modelId": model, "prompt": prompt, "images": images}
            path = "/image/multi-edit"
        aspect_ratio = str(arguments.get("aspect_ratio") or "").strip()
        if aspect_ratio and aspect_ratio != "auto":
            # auto is the endpoint default: the edit keeps the input shape,
            # the multi-edit flow reads the shape of the first image.
            body["aspect_ratio"] = aspect_ratio
        if model in VENICE_EDIT_TIER_MODELS:
            # The tier models render at the 2K preset (the override map
            # drops a model below it), the quality models at medium —
            # the same convention as generate, so the catalog cost
            # matches the bill. A model out of the quality set renders
            # at its default quality (gpt-image-2-5-sunburst-edit, high).
            body["resolution"] = VENICE_EDIT_MODEL_RESOLUTIONS.get(model, VENICE_IMAGE_RESOLUTION)
            if model in VENICE_EDIT_QUALITY_MODELS:
                body["quality"] = VENICE_IMAGE_QUALITY
        if engine.channel_nsfw is not None and await engine.channel_nsfw():
            # The endpoint default true blurs adult content: it drops only
            # where Discord itself gates the channel behind 18+.
            body["safe_mode"] = False
        try:
            data, headers = await engine.api_post(path, json_body=body, binary=True, timeout=VENICE_RENDER_TIMEOUT)
        except ChatError as e:
            return f"Error: the image edit failed: {e}"
        # The endpoint answered, so the edit billed the catalog price of its
        # preset: a refusal bills the same. The extra images of a multi-edit
        # bill beyond the catalog price, the report carries the base alone.
        await _report_cost(engine, entry["cost"])
        if not isinstance(data, (bytes, bytearray)) or not data:
            return "Error: the image edit returned no image."
        violation, status = _moderation_status(headers)
        refused = _refusal_error("edit", bytes(data), violation, status)
        if refused is not None:
            return refused
        if engine.send_file is None:
            return "Error: the image posting is not available here."
        # The binary answer carries no generation id: the answer format
        # (a content-type header) names the extension, the model and a
        # random token name the file.
        content_type = str(headers.get("content-type") or "").split(";")[0].strip().lower()
        extension = VENICE_EDIT_FORMATS.get(content_type, "png")
        name = f"{model}-{random.randint(0, 0xFFFF):04x}.{extension}"
        sent = await engine.send_file(name, bytes(data))
        if status:
            return f"{sent}\n{status}"
        return sent

    return {
        "name": "edit_image"
        # The usage counter a successful edit increments (the engine counts
        # the call into the scope stats).
      , "media": "inpaints"
      , "description": (
            "Edit images through Venice: one picture, or several pictures composited into one result. "
            "The result joins the current message as an attachment, and the tool answer names the posted file. "
            "A model that refuses copyrighted material says so in the model field: "
            "describe the subject instead of naming it, or pick another model."
        )
      , "parameters": {
            "type": "object"
          , "properties": {
                "image": {
                    "type": "string"
                  , "description": "The http(s) URL of the base picture to edit. Use an attachment of the conversation, or the URL a generate_image answer names."
                }
              , "extra_images": {
                    "type": "array"
                  , "items": {"type": "string"}
                  , "description": "More http(s) URLs the model composites with the base picture, as layers or masks. The model property names the models that take them."
                }
                , "prompt": {"type": "string", "description": "What to change in the picture."}
                , "model": _model_property(catalog)
                , "aspect_ratio": {
                    "type": "string"
                    , "default": "auto"
                    , "description": "The aspect ratio of the result, for example 1:1, 16:9, or 9:16. Auto keeps the shape of the base image."
                }
            }
            , "required": ["image", "prompt"]
        }
        , "handler": handler
    }


def _background_remove_tool() -> dict:
    """The background removal endpoint as a native tool: one image in, a PNG with a transparent background out, posted to the conversation.
       The endpoint takes no model and no other dial (additionalProperties false): a foreign URL rides the body as image_url, a Discord download as base64 in image.
       The binary answer posts as a file.
       The call bills the flat price of bria-bg-remover (VENICE_BACKGROUND_COST, the entry the live image list carries behind the endpoint, released Feb 25, 2026), and every answered removal reports it into the cost total of the reply.
       The endpoint takes no model, so no catalog entry rides the tool.
    """

    async def handler(arguments, engine):
        image = str(arguments.get("image") or "").strip()
        if not image.startswith(("http://", "https://")):
            return "Error: the image must be the http(s) URL of the picture."
        body = {}
        if urlparse(image).netloc.lower() in DISCORD_FILE_HOSTS:
            # The Discord file hosts need an authorized download: the
            # image rides the body as base64 instead of the URL.
            if engine.fetch_url is None:
                return "Error: the Discord download is not available here."
            fetched = await engine.fetch_url(image)
            if fetched is None:
                return "Error: the download of the Discord file failed."
            body["image"] = base64.b64encode(fetched[0]).decode("ascii")
        else:
            body["image_url"] = image
        try:
            data, headers = await engine.api_post("/image/background-remove", json_body=body, binary=True)
        except ChatError as e:
            return f"Error: the background removal failed: {e}"
        # The endpoint answered, so the removal billed its flat price: a
        # refusal bills the same.
        await _report_cost(engine, VENICE_BACKGROUND_COST)
        if not isinstance(data, (bytes, bytearray)) or not data:
            return "Error: the background removal returned no image."
        violation, status = _moderation_status(headers)
        refused = _refusal_error("background removal", bytes(data), violation, status)
        if refused is not None:
            return refused
        if engine.send_file is None:
            return "Error: the image posting is not available here."
        # The binary answer carries no id: the answer format (a
        # content-type header) names the extension, the endpoint and a
        # random token name the file.
        content_type = str(headers.get("content-type") or "").split(";")[0].strip().lower()
        extension = VENICE_EDIT_FORMATS.get(content_type, "png")
        name = f"background-removed-{random.randint(0, 0xFFFF):04x}.{extension}"
        sent = await engine.send_file(name, bytes(data))
        if status:
            return f"{sent}\n{status}"
        return sent

    return {
        "name": "remove_background"
        # The usage counter a successful removal increments: it produces
        # an image post, so it rides the generate windows.
      , "media": "images"
        , "description": (
            "Remove the background of one image through Venice. "
            "The result joins the current message as an attachment, and the tool answer names the posted file."
        )
        , "parameters": {
            "type": "object"
            , "properties": {
                "image": {
                    "type": "string"
                    , "description": "The http(s) URL of the picture. Use an attachment of the conversation, or the URL a generate_image answer names."
                }
            }
            , "required": ["image"]
        }
        , "handler": handler
    }


def _music_model_property() -> dict:
    """The model parameter of the song tool: the enum of the preset names and a description that names the lyrics models, the instrumental models, and the duration models."""
    lyrics = [name for name, entry in VENICE_MUSIC_MODELS.items() if "lyrics" in entry["traits"]]
    instrumental = [name for name, entry in VENICE_MUSIC_MODELS.items() if "instrumental" in entry["traits"]]
    duration = [
        name for name, entry in VENICE_MUSIC_MODELS.items()
        if "duration" in VENICE_MUSIC_CONSTRAINTS.get(entry["model"], {})
    ]
    notes = [
        f"Lyrics models: {', '.join(lyrics)}."
        , f"Instrumental models (no vocals): {', '.join(instrumental)}."
        , f"Duration models: {', '.join(duration)}."
    ]
    description = "The model preset name. " + " ".join(notes)
    return {"type": "string", "enum": list(VENICE_MUSIC_MODELS), "default": next(iter(VENICE_MUSIC_MODELS)), "description": description}


def _music_tool() -> dict:
    """The song generation flow as a native tool: one request, posted as an approval embed.
       The tool validates the request against the constraints table and prices it with the live quote endpoint.
       It then hands the request to the approval flow of the engine: the embed with the Approve and Reject buttons and the price posts to the conversation,
       and the generation runs only after the approval.
       The tool returns before the vote: the outcome arrives as a later harness turn."""

    async def handler(arguments, engine):
        prompt = str(arguments.get("prompt") or "").strip()
        if not prompt:
            return "Error: the prompt must be a non-empty string."
        asked = str(arguments.get("model") or "").strip()
        matched = None
        for preset_name, entry in VENICE_MUSIC_MODELS.items():
            # The preset name is the only handle, in any casing; a blank
            # ask takes the first entry (the default).
            if not asked or preset_name.lower() == asked.lower():
                matched = (preset_name, entry)
                break
        if matched is None:
            # Only the curated catalog, by its preset names (a raw model id is not a name).
            return (
                f"Error: unknown song model {asked}. "
                f"Valid models: {', '.join(VENICE_MUSIC_MODELS)}. "
                f"The price and the dials of every model sit in the harness resource {MUSIC_URI} (read_resource, server \"harness\")."
            )
        preset_name, entry = matched
        model = entry["model"]
        constraints = VENICE_MUSIC_CONSTRAINTS.get(model, {})
        prompt_limit = constraints.get("prompt")
        if prompt_limit is not None and len(prompt) > prompt_limit:
            return f"Error: the prompt is over the {prompt_limit}-character limit of the model {preset_name}."
        body = {"model": model, "prompt": prompt}
        lyrics = str(arguments.get("lyrics") or "").strip()
        lyrics_limit = constraints.get("lyrics")
        if lyrics and lyrics_limit is None:
            return f"Error: the model {preset_name} takes no lyrics. Pick a lyrics model."
        if lyrics and len(lyrics) > lyrics_limit:
            return f"Error: the lyrics are over the {lyrics_limit}-character limit of the model {preset_name}."
        if lyrics:
            body["lyrics_prompt"] = lyrics
        dial = constraints.get("duration")
        duration = arguments.get("duration_seconds")
        if dial is None:
            if duration is not None:
                return f"Error: the model {preset_name} takes no duration dial."
        else:
            low, high, default = dial
            if duration is None:
                # The model default prices and renders the bare request.
                duration = default
            try:
                duration = int(duration)
            except (TypeError, ValueError):
                return "Error: the duration_seconds must be a whole number of seconds."
            if not low <= duration <= high:
                return f"Error: the duration_seconds of {preset_name} must stay between {low} and {high}."
            body["duration_seconds"] = duration
        instrumental = arguments.get("instrumental")
        if instrumental:
            if not constraints.get("instrumental"):
                return f"Error: the model {preset_name} takes no force_instrumental dial."
            body["force_instrumental"] = True
        try:
            cost = await quote_song(engine.api_post, body)
        except ChatError as e:
            return f"Error: the song quote failed: {e}"
        if engine.request_song is None:
            return "Error: the song approval is not available here."
        request = {
            "preset": preset_name
            , "model": model
            , "prompt": prompt
            , "lyrics": lyrics
            , "duration": body.get("duration_seconds")
            , "instrumental": bool(body.get("force_instrumental"))
            , "cost": cost
            , "body": body
        }
        return await engine.request_song(request)

    return {
        "name": "generate_song"
        # The usage counter a successful request increments (the engine
        # counts the call into the scope stats).
      , "media": "music"
        # The availability flags of the engine: a direct message carries the
        # tool only for the bot owner, and a guild hides it while the
        # credit ratio sits under the media disable band.
      , "dm_owner_only": True
      , "guild_credit_gate": True
        , "description": (
            "Generate one song from a text prompt through Venice. "
            "The request posts as an approval embed with its price and buttons. "
            "The song generates only after the users approve it, and the result joins the conversation as an audio file. "
            "A lyrics model sings the lyrics you pass. An instrumental model generates no vocals."
        )
        , "parameters": {
            "type": "object"
            , "properties": {
                "prompt": {"type": "string", "description": "What the song sounds like: genre, mood, instruments, tempo."}
                , "model": _music_model_property()
                , "lyrics": {"type": "string", "description": "The lyrics the model sings. Only a lyrics model accepts them."}
                , "duration_seconds": {"type": "integer", "description": "The length of the song in seconds. Only a duration model accepts the dial."}
                , "instrumental": {"type": "boolean", "description": "Force an instrumental track without vocals. Only a model with the dial accepts it."}
            }
            , "required": ["prompt"]
        }
        , "handler": handler
    }


def _activity_tool() -> dict:
    """The activity report as a native tool: the agent names what the
       conversation is doing and the capabilities the work needs. The
       activity label rides the session into the Discord presence of the
       bot. The capabilities resolve through the routing selection: the
       same set that already runs keeps everything in place, and a
       smaller carrier can take the work when it covers the needs — the
       tool never stores the traits that activated a preset, it resolves
       the requested set against the catalog each call. The answer names
       the active preset only on a move, the description tells the agent
       of the presence and the tool optimisation alone. The nsfw
       capability needs a conversation behind the 18+ gate.
    """

    async def handler(arguments, engine):
        activity = str(arguments.get("activity") or "").strip()
        if not activity:
            return "Error: the activity must be a non-empty string."
        raw = arguments.get("capabilities")
        items = [raw] if isinstance(raw, str) else list(raw) if isinstance(raw, list) else []
        requested = []
        for item in items:
            text = str(item).strip().lower()
            if text and text not in requested:
                requested.append(text)
        unknown = [item for item in requested if item not in VENICE_CHAT_CAPABILITIES]
        if unknown:
            return (
                f"Error: unknown capability: {', '.join(unknown)}. "
                f"Known capabilities: {', '.join(VENICE_CHAT_CAPABILITIES)}."
            )
        if not requested:
            return "Error: state at least one capability."
        if engine.set_conversation_model is None:
            return "Error: the environment configuration is not available here."
        if engine.set_activity is not None:
            await engine.set_activity(activity)
        gated = "nsfw" in requested
        if gated and (engine.channel_nsfw is None or not await engine.channel_nsfw()):
            return "Error: the nsfw capability needs a conversation behind the 18+ gate."
        ratio_getter = getattr(engine, "credit_ratio", None)
        ratio = await ratio_getter() if ratio_getter is not None else None
        if ratio is not None and ratio < VENICE_CREDIT_ROUTING_FLOOR:
            # The floor of the session-start routing: the balance cannot
            # cover the cycle rest, every switch stays closed. The report
            # alone lands — the presence is free.
            return (
                f"The activity is {activity!r}. The bundled balance runs low, "
                "the environment stays as it is."
            )
        resolved = select_preset(
            {trait: VENICE_ROUTING_TRAIT_AT for trait in requested if trait != "nsfw"}
          , nsfw_allowed=gated
          , ratio=ratio
        )
        if resolved is None:
            menu = "; ".join(
                preset_menu_line(name, preset)
                for name, preset in VENICE_CHAT_PRESETS.items()
                if not preset.get("disabled")
            )
            return f"Error: no environment preset provides: {', '.join(sorted(requested))}. Available: {menu}."
        current = await engine.conversation_preset() if engine.conversation_preset is not None else None
        if resolved == current:
            # The requested set resolves onto the running preset: nothing
            # moves, the report alone lands.
            return f"The activity is {activity!r}. The current environment already provides: {', '.join(sorted(requested))}."
        error = await engine.set_conversation_model(resolved)
        if error:
            return f"Error: the switch to {resolved} was refused: {error}"
        granted = ", ".join(sorted(requested))
        suffix = " (18+ variant)" if gated else ""
        return (
            f"The activity is {activity!r}. Active preset: {resolved}{suffix}. "
            "The change answers the next message."
        )

    return {
        "name": "set_activity"
        , "description": (
            "Report the current activity of this conversation. Call it once each time a new activity starts. "
            "It drives the presence status of the bot and the tool optimisation of the conversation."
        )
        , "parameters": {
            "type": "object"
            , "properties": {
                "activity": {
                    "type": "string"
                  , "description": "A short name of what you are doing, for example 'debugging a script' or 'telling a story'."
                }
                , "capabilities": {
                    "type": "array"
                  , "items": {"type": "string", "enum": list(VENICE_CHAT_CAPABILITIES)}
                  , "description": (
                        f"The capabilities the activity needs. Values: {', '.join(VENICE_CHAT_CAPABILITIES)}. "
                        "The nsfw capability needs a conversation behind the 18+ gate."
                    )
                }
            }
            , "required": ["activity", "capabilities"]
        }
        , "handler": handler
    }


def preset_menu_line(name: str, preset: dict) -> str:
    """One catalog entry for the menus of the agent and the user."""
    traits = ", ".join(sorted(preset.get("traits", ()))) or "plain"
    suffix = "; 18+ variant available" if preset.get("nsfw") else ""
    return f"{name} ({traits}{suffix})"
