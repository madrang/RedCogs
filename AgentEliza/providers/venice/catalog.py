# The curated tables of the Venice provider:
#   - The model catalogs
#   - The endpoint constants
#   - The pricing dials.
# The file holds pure data and no imports.

# The limit types of the rate-limits endpoint, spelled out for the usage rows.
VENICE_LIMIT_NAMES = {"RPM": "requests/min", "RPD": "requests/day", "TPM": "tokens/min"}
# The bundled credit plan: the monthly allowance — Venice keeps at most three months of allowance banked (300%), the rest is lost.
# The rate-limits endpoint reports the live balance in data.balances.BUNDLED_CREDITS (USD, 100 credits a dollar).
# The cycle day — the day of the month the cycle restarts — paces the credit gate
# (the guild media tools and the chat model routing):
# it lives in Config (`eliza setcycleday`), never in this file.
VENICE_CREDIT_ALLOWANCE = 22500
# The augment endpoints are experimental and billed per request ($0.01 each).
VENICE_QUERY_MAX_CHARS = 400
VENICE_SEARCH_MAX_LIMIT = 20
# Image generation: the endpoint-wide prompt cap.
VENICE_PROMPT_MAX_CHARS = 7500
# The curated image models.
# Order: release date, the most recent first. The first entry is the default model, like the first entry of PROVIDERS is the default provider.
# Keys: the preset names, the only model handle the agent sees, like the chat presets.
# A raw model id is not a name and the tool refuses it.
# Cost: the price in USD of the request the tool sends — the 2K preset on the resolution-tier models (gpt-image-2-5-sunburst at 2K medium),
# the flat generation price on the rest, the default 1K tier when a model prices by tier and the tool sends no resolution.
# Source: the live model list (GET /models?type=image needs no key).
# Image models price per image, not per million tokens.
# The comment on each entry names the release date (the created field of the live list).
VENICE_IMAGE_MODELS = {
    "Muse": {"model": "muse-image", "traits": [], "cost": 0.02}  # released Sep 2, 2026
  , "Grok": {"model": "grok-imagine-image-2-0", "traits": ["copyrighted_material"], "cost": 0.10}  # released Aug 10, 2026
  , "Qwen": {"model": "qwen-image-3-pro", "traits": ["uncensored"], "cost": 0.09}  # released Jul 15, 2026
  , "Seedream": {"model": "seedream-v5-pro", "traits": ["uncensored", "copyrighted_material"], "cost": 0.11}  # released Jul 7, 2026
  , "Luma": {"model": "luma-uni-1-max", "traits": [], "cost": 0.12}  # released Jun 16, 2026
  , "Ideogram": {"model": "ideogram-v4", "traits": [], "cost": 0.06}  # released Jun 2, 2026
  , "Krea": {"model": "krea-v2-large", "traits": [], "cost": 0.07}  # released May 21, 2026
  , "GPT Image": {"model": "gpt-image-2-5-sunburst", "traits": [], "cost": 0.05}  # released Sep 7, 2026
  , "Wan": {"model": "wan-2-7-pro-text-to-image", "traits": [], "cost": 0.09375}  # released Mar 31, 2026
  , "Lustify": {"model": "lustify-v8", "traits": ["uncensored"], "cost": 0.01}  # released Mar 29, 2026
  , "Hunyuan": {"model": "hunyuan-image-v3", "traits": [], "cost": 0.09}  # released Feb 28, 2026
  , "Nano Banana": {"model": "nano-banana-2", "traits": [], "cost": 0.14}  # released Feb 25, 2026
  , "Recraft": {"model": "recraft-v4-pro", "traits": [], "cost": 0.29}  # released Feb 11, 2026
  , "Chroma": {"model": "chroma", "traits": ["uncensored"], "cost": 0.01}  # released Jan 29, 2026
  , "ImagineArt": {"model": "imagineart-1.5-pro", "traits": [], "cost": 0.06}  # released Jan 26, 2026
  , "Z Turbo": {"model": "z-image-turbo", "traits": [], "cost": 0.01}  # released Dec 3, 2025
  , "Flux": {"model": "flux-2-max", "traits": [], "cost": 0.09}  # released Nov 25, 2025
  , "Venice SD": {"model": "venice-sd35", "traits": [], "cost": 0.01}  # released Mar 27, 2025
  , "WAI Illustrious": {"model": "wai-Illustrious", "traits": ["uncensored"], "cost": 0.01}  # released Jan 11, 2025
}
# The promptCharacterLimit of each curated model (model_spec.constraints of the live list), the generate catalog first, the edit catalog after.
# An entry may sit above the endpoint-wide VENICE_PROMPT_MAX_CHARS: the model accepts the longer prompt, so its entry lifts the cap.
# An id missing here keeps the endpoint-wide fallback.
VENICE_IMAGE_PROMPT_LIMITS = {
    "muse-image": 10000
  , "grok-imagine-image-2-0": 7500
  , "qwen-image-3-pro": 10000
  , "seedream-v5-pro": 10000
  , "luma-uni-1-max": 6000
  , "ideogram-v4": 10000
  , "krea-v2-large": 5000
  , "gpt-image-2-5-sunburst": 10000
  , "wan-2-7-pro-text-to-image": 3000
  , "lustify-v8": 1500
  , "hunyuan-image-v3": 3000
  , "nano-banana-2": 32768
  , "recraft-v4-pro": 10000
  , "chroma": 7500
  , "imagineart-1.5-pro": 10000
  , "z-image-turbo": 7500
  , "flux-2-max": 3000
  , "venice-sd35": 1500
  , "wai-Illustrious": 1500
  , "muse-image-edit": 10000
  , "grok-imagine-image-2-0-edit": 7500
  , "qwen-image-3-pro-edit": 10000
  , "luma-uni-1-edit": 6000
  , "wan-2-7-pro-edit": 5000
  , "gpt-image-2-5-sunburst-edit": 10000
  , "firered-image-edit": 1500
  , "nano-banana-2-edit": 32768
  , "flux-2-max-edit": 3000
  , "seedream-v4-edit": 10000
}
# The sizing dialect of each curated model. An unknown id takes the ratio dialect.
# pixel models size through width and height (model_spec.constraints carries a widthHeightDivisor and no aspectRatios).
# resolution-tier models carry a fixed resolution (and gpt-image-2-5-sunburst a quality) preset (constraints carries a resolutions array).
# The rest take the aspect ratio as-is.
# The tool description reports the dialect beside the traits of each model, so the agent knows which parameters apply.
VENICE_IMAGE_DIALECTS = {
    "muse-image": "ratio"
  , "grok-imagine-image-2-0": "resolution"
  , "qwen-image-3-pro": "resolution"
  , "seedream-v5-pro": "resolution"
  , "luma-uni-1-max": "ratio"
  , "ideogram-v4": "ratio"
  , "krea-v2-large": "ratio"
  , "gpt-image-2-5-sunburst": "resolution"
  , "wan-2-7-pro-text-to-image": "ratio"
  , "lustify-v8": "pixel"
  , "hunyuan-image-v3": "ratio"
  , "nano-banana-2": "resolution"
  , "recraft-v4-pro": "ratio"
  , "chroma": "pixel"
  , "imagineart-1.5-pro": "ratio"
  , "z-image-turbo": "pixel"
  , "flux-2-max": "ratio"
  , "venice-sd35": "pixel"
  , "wai-Illustrious": "pixel"
}
# The presets of the resolution-tier models. 2K and medium for now, adjust after some use.
VENICE_IMAGE_RESOLUTION = "2K"
VENICE_IMAGE_QUALITY = "medium"
# The total timeout of one image render attempt: a 2K edit render runs past the 120 s augment default of provider_post.
# The value matches the inference cap of chat_request — a server-side render waits like a long generation.
VENICE_RENDER_TIMEOUT = 900
# aspect_ratio to pixels for the pixel-dialect models: sides at most 1280, multiples of 16 — every pixel model's widthHeightDivisor divides it
# (16 on venice-sd35 and wai-Illustrious, 8 on z-image-turbo, lustify-v8, chroma), so one table serves them all. About one megapixel.
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
# The curated edit models of /image/edit.
# Order and keys mirror the generate catalog: release date, the most recent first, the first entry the default model, preset-name keys the agent sees.
# Source: the live list under the inpaint type (GET /models?type=inpaint, which needs no key). The edit models are inpaint models.
# Cost: the inpaint price at the 2K resolution tier, else the flat price.
# A quality table never sets an edit cost — the endpoint refuses the quality body field, so every model renders at its default quality.
# gpt-image-2-5-sunburst-edit defaults to high and bills 2K at $0.15.
# grok-imagine-image-2-0-edit defaults to medium, so its 2K price is $0.10 either way.
# Each entry holds the model id behind its name, its capability traits (plain names), and its real cost in USD per edit.
# The comment on each entry names the release date (the created field of the live list).
# The max_images key holds the input-image ceiling of the compositing models (constraints.maxInputImages of the live list, read 2026-09-19),
# and None marks a model the live list caps not (the grok, qwen, and firered families).
# A model with no max_images key composites no extra image (constraints.combineImages false, luma-uni-1-edit).
# The extra images ride the multi-edit endpoint, one request with the whole image array.
# An extra image can bill on its own (pricing.inputImages of the live list): every image past the included count pays the additional price.
VENICE_EDIT_MODELS = {
    "Muse": {"model": "muse-image-edit", "traits": [], "cost": 0.02, "max_images": 6}  # released Sep 2, 2026
  , "Grok": {"model": "grok-imagine-image-2-0-edit", "traits": [], "cost": 0.10, "max_images": None}  # released Aug 10, 2026
  , "Qwen": {"model": "qwen-image-3-pro-edit", "traits": [], "cost": 0.09, "max_images": None}  # released Jul 15, 2026
  , "Luma": {"model": "luma-uni-1-edit", "traits": [], "cost": 0.06}  # released Jun 16, 2026
  , "Wan": {"model": "wan-2-7-pro-edit", "traits": [], "cost": 0.094, "max_images": 6}  # released Apr 22, 2026
  , "GPT Image": {"model": "gpt-image-2-5-sunburst-edit", "traits": [], "cost": 0.15, "max_images": 6}  # released Sep 7, 2026
  , "FireRed": {"model": "firered-image-edit", "traits": [], "cost": 0.04, "max_images": None}  # released Mar 24, 2026
  , "Nano Banana": {"model": "nano-banana-2-edit", "traits": [], "cost": 0.14, "max_images": 6}  # released Feb 25, 2026
  , "Flux": {"model": "flux-2-max-edit", "traits": [], "cost": 0.12, "max_images": 6}  # released Jan 4, 2026
  , "Seedream": {"model": "seedream-v4-edit", "traits": ["uncensored"], "cost": 0.05, "max_images": 6}  # released Jan 3, 2026
}
# The edit models that price and render by resolution tier: the edit tool sends the same 2K constant as generate, so the catalog cost matches the bill.
# A model outside the quality set renders at its default quality.
# The set holds model ids: the handler resolves a preset name to its entry before it reads the set.
VENICE_EDIT_TIER_MODELS = {
    "gpt-image-2-5-sunburst-edit"
  , "nano-banana-2-edit"
  , "grok-imagine-image-2-0-edit"
  , "qwen-image-3-pro-edit"
}
# The edit models that render below the 2K preset: an id here renders at its mapped resolution instead.
# No model rides the map today. The quality routes stay closed on the edit endpoint: the body field is an unrecognized key, and a model feature suffix answers an invalid model id.
# An id missing here keeps VENICE_IMAGE_RESOLUTION.
VENICE_EDIT_MODEL_RESOLUTIONS = {}
# The edit models that take the quality preset: the tool sends them the medium constant of generate, so the model skips its high default and the catalog cost matches the bill.
# No model accepts the parameter today, so the set is empty.
# TODO: revalidate later — the docs and the live specs still carry the parameter and its quality tables (2K medium $0.06 on gpt-image-2-5-sunburst-edit).
VENICE_EDIT_QUALITY_MODELS: set[str] = set()
# The edit answer formats of the endpoint, mapped to file extensions.
VENICE_EDIT_FORMATS = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}
# The agent-facing label of each trait flag. The flags ride the catalog entries (VENICE_IMAGE_MODELS, VENICE_EDIT_MODELS, VENICE_MUSIC_MODELS).
# copyrighted_material: the model refuses a prompt that names copyrighted material. The answer is a uniform blank image, no error.
# uncensored: the live model list reports the model as applying minimal content-based filtering (model_spec.uncensored true).
# The full trait table lives in the vault note Venice.AI/HTTP API.md.
VENICE_IMAGE_TRAIT_LABELS = {
    "copyrighted_material": "refuses copyrighted material"
  , "uncensored": "uncensored"
}
# A refused render is a uniform blank image, so the pixel check decides blankness — the compressed size spans too wide for a size ceiling.
# A refused render reads uniform black: this per-channel ceiling separates it from dark noise.
VENICE_IMAGE_DARK_MAX = 8
# The curated music models — the full-song models of the live music list.
# Order: release date, the most recent first. The first entry is the default model, like the image catalogs.
# Keys: the preset names, the only model handle the agent sees, like the chat presets. A raw model id is not a name.
# Cost: the price in USD of a bare generation at the model defaults, validated against the live quote endpoint
# (POST /audio/quote needs no key, its answers round to cents): the flat generation price on the flat models,
# the default-duration price on the duration-priced rest (60 s on ACE-Step and ElevenLabs, 90 s on Sonilo).
# Seed Audio takes no duration dial (the quote body refuses the field): it prices per second of output, the
# length rides the prompt, and its entry holds the bare-request quote.
# The generation flow is asynchronous — quote, queue (POST /audio/queue), poll (POST /audio/retrieve), then
# complete (POST /audio/complete) after the download. No tool rides the catalog yet.
# The lyrics trait: the model takes custom lyrics (lyrics_prompt of the queue body).
# The instrumental trait: the model generates no vocals.
# Source: the live music list (GET /models?type=music needs no key).
# The comment on each entry names the release date (the created field of the live list).
VENICE_MUSIC_MODELS = {
    "Sonilo": {"model": "sonilo-v1-1-music", "traits": ["instrumental"], "cost": 0.25875}  # released Jul 20, 2026
  , "Seed Audio": {"model": "seed-audio-1-0", "traits": [], "cost": 0.35}  # released Jun 28, 2026
  , "Lyria": {"model": "lyria-3-pro", "traits": [], "cost": 0.10}  # released May 21, 2026
  , "MiniMax": {"model": "minimax-music-v26", "traits": ["lyrics"], "cost": 0.18}  # released Apr 11, 2026
  , "ACE-Step": {"model": "ace-step-15", "traits": ["lyrics", "uncensored"], "cost": 0.03}  # released Feb 22, 2026
  , "ElevenLabs": {"model": "elevenlabs-music", "traits": ["instrumental", "uncensored"], "cost": 0.69}  # released Feb 21, 2026
}
# The request dials and text limits of each curated music model (model_spec of the live list).
# Keys per model id:
#   - prompt: the promptCharacterLimit.
#   - lyrics: the lyricsCharacterLimit, present only on a model that takes lyrics_prompt.
#   - duration: the (minimum, maximum, default) of duration_seconds, present only on a model that takes the dial.
#   - instrumental: True on a model that takes force_instrumental.
# A model with no entry for a dial refuses the body field (HTTP 400).
VENICE_MUSIC_CONSTRAINTS = {
    "sonilo-v1-1-music": {"prompt": 4096, "duration": (1, 600, 90)}
  , "seed-audio-1-0": {"prompt": 3000}
  , "lyria-3-pro": {"prompt": 5000}
  , "minimax-music-v26": {"prompt": 300, "lyrics": 1000, "instrumental": True}
  , "ace-step-15": {"prompt": 512, "lyrics": 4096, "duration": (60, 210, 60)}
  , "elevenlabs-music": {"prompt": 4096, "duration": (3, 600, 60), "instrumental": True}
}
# The answer formats of the retrieve endpoint, mapped to file extensions.
VENICE_MUSIC_FORMATS = {"audio/mpeg": "mp3", "audio/mp3": "mp3", "audio/wav": "wav", "audio/flac": "flac", "audio/mp4": "m4a", "audio/x-m4a": "m4a"}
# The total budget of one song generation: the queue call plus every retrieve poll.
# The value matches the inference cap of chat_request — a server-side render waits like a long generation.
VENICE_MUSIC_TIMEOUT = 900
# The healthy-balance mark of the credit ratio: the cost pressure of the
# selection tiebreak reads its minimum at and above it, its maximum at the
# routing floor.
VENICE_CREDIT_GATE_BUFFER = 1.5
# The routing floor of the credit ratio: under it the chat routing stops and
# the configured model answers.
VENICE_CREDIT_ROUTING_FLOOR = 1.0
# The cost ceiling of the lite tier of the chat routing. No caller rides it.
VENICE_ROUTING_LITE_COST = 0.10
# The strength a capability must read before the chat routing counts it as needed.
VENICE_ROUTING_TRAIT_AT = 0.5
# The cost pressure of the selection tiebreak: the pressure names the cost
# step one extra preset trait must stay under. A healthy balance reads the
# min, the routing floor reads the max, and the ratio between them scales
# linearly.
VENICE_ROUTING_PRESSURE_MIN = 1.0
VENICE_ROUTING_PRESSURE_MAX = 8.0
# The decision model of the chat routing: a fresh session asks it which chat
# preset fits its opening message. Source: the live decision list
# (GET /models?type=decision, read 2026-09-20), the only id it publishes.
# The input price rides at $0.042 per 1M tokens, the answers stay free.
JEV_MODEL_ID = "jev-latest"
# The capabilities the agent can ask of the environment tool. One settled vocabulary: the same words in the schema enum, the catalog traits, and the tool answers, in the language a user types — no spec terms.
# The coding trait mirrors the optimizedForCode flag of the live model list. The roleplay and storytelling traits ride the Aion pair.
# The concise answers and thorough answers traits mark the size pairs: the lite and mini presets answer concisely, the regular and pro presets thoroughly.
# The decision flow judges the pair as one score axis: a conversation cannot need both ends.
# The reasoning trait marks a model that solves hard problems far above its class. The writing trait marks a model whose prose stands close to the leaders.
# The imagination trait names a model that brings more original ideas than its class: the Google entries carry it on the benchmarks and the owner's word
# of 2026-09-21, distinct from the roleplay and storytelling of the Aion pair.
# The Qwen presets carry no such trait on the community word: the 3.8 answers read stiff and filtered for creative work.
VENICE_CHAT_CAPABILITIES = (
    "vision", "nsfw", "long context", "coding", "reasoning", "writing", "imagination"
  , "roleplay", "storytelling", "concise answers", "thorough answers"
)
# The fixed tool list of the presets whose models overuse the wider tool
# surface (the Qwen presets and Gemma): the choice poll, the environment
# switch, and the media endpoints whose results stay small. The MCP servers
# of the Config and every other harness or provider tool stay off these
# presets.
VENICE_FIXED_TOOLS = (
    "propose_choices", "configure_environment"
  , "generate_image", "edit_image", "remove_background", "generate_song"
)
# Chat presets: a short display name for the agent and the user, the model id behind it, an optional NSFW variant id for conversations behind the 18+ gate, the capability names the preset provides, and the cost of the preset.
# The variant carries the same capabilities as the normal id: a model whose capabilities differ joins the catalog under its own preset.
# Cost: the operating price of the model on a 0..1 scale. The operating price is 10x the input price plus 1x the output and cache-read prices, each per 1M tokens (the operating_usd_per_m column of scripts/list_models.py).
# The scale anchors DeepSeek Lite at 0 and the priciest catalog model at 1. A negative cost marks a preset cheaper than the default.
# The catalog order is preference order: the first preset that satisfies a request wins. The short names are the only model handle the agent ever sees.
# An entry may carry disabled True: the environment tool and the overload fallback skip it, so the agent cannot reach it, while the user commands (setmodel, the providers menu) keep it.
# An entry may carry tool_filter, the names of the only tools its models may carry: the engine drops every other tool, the MCP set included.
# An entry may carry mcp False: the model takes no MCP server tools, the harness and provider tools stay (the Gemini backend rejects the schema keywords of the user servers).
# The comment on each entry names the release date of its model ids (the created field of the live list).
VENICE_CHAT_PRESETS = {
    "DeepSeek Lite": {
        # No coding trait (the operator's call): a coding request upgrades to DeepSeek Pro, the next preset that carries it.
        "normal": "deepseek-v4-flash-0731"  # released Jul 31, 2026
        # "normal": "deepseek-v4-1-flash" # Sep 9, 2026 - cost: 0.06
      , "traits": ["long context", "concise answers"]
      , "cost": 0.0
    }
  , "DeepSeek Pro": {
        "normal": "deepseek-v4-pro"  # released Apr 24, 2026
        # "normal": "deepseek-v4-pro-0813" # Aug 13, 2026 - cost: 0.36
      , "traits": ["long context", "coding", "thorough answers", "writing"]
      , "cost": 0.33
    }

    # Google
  , "Gemma": {
        # The fixed tool list stays under the model cap of 20 tool definitions.
        "normal": "google-gemma-4-31b-it"  # released Apr 3, 2026
      , "traits": ["vision", "reasoning", "nsfw", "imagination"]
      , "cost": -0.01
      , "nsfw": "gemma-4-uncensored"  # released Apr 13, 2026
      , "tool_filter": VENICE_FIXED_TOOLS
    }
    , "Gemini": {
        "normal": "gemini-3-8-flash"  # released Sep 2, 2026
      , "traits": ["vision", "coding", "long context", "imagination", "concise answers"]
      , "cost": 0.22
      , "mcp": False
    }

    # Meta
  , "Llama Lite": {
        # Disabled, untested - 128K Ctx.
        "normal": "llama-3.2-3b"  # released Oct 3, 2024
      , "traits": ["concise answers"]
      , "cost": 0.0
      , "disabled": True
    }
  , "Llama": {
        # Disabled, untested - 128K Ctx.
        "normal": "llama-3.3-70b"  # released Apr 6, 2025
      , "traits": ["thorough answers"]
      , "cost": 0.14
      , "disabled": True
    }

    # Z.AI
  , "GLM Lite": {
        # Disabled for the agent: Low quality output.
        # The negative-cost preset stays a manual choice of the operator.
        "normal": "zai-org-glm-4.7-flash"  # released Jan 29, 2026
      , "traits": ["concise answers"]
      , "cost": -0.02
      , "nsfw": "olafangensan-glm-4.7-flash-heretic"  # released Feb 4, 2026
      , "disabled": True
    }
  , "GLM Vision": {
        "normal": "z-ai-glm-5-3-flash"  # released Aug 21, 2026
      , "traits": ["long context", "vision", "coding"]
      , "cost": 0.0
    }
  , "GLM 1M": {
        "normal": "z-ai-glm-5-3"  # released Aug 18, 2026
      , "traits": ["long context", "coding", "thorough answers"]
      , "cost": 0.39
    }

    # Thinking Machines
  , "Inkling": {
        "normal": "inkling"  # released Jul 16, 2026
      , "traits": ["vision", "concise answers"]
      , "cost": 0.29
    }

    # MoonshotAI
  , "Kimi": {
        "normal": "kimi-k3"  # released Jul 16, 2026
      , "traits": ["long context", "vision", "coding", "thorough answers", "writing"]
      , "cost": 1.0
    }

    # Alibaba
  , "Qwen Lite": {
        # No coding trait (the operator's call): a coding request upgrades to Qwen Max.
        "normal": "qwen-3-8-flash"  # 1M Ctx, released Sep 9, 2026
      , "traits": ["long context", "vision", "reasoning", "concise answers"]
      , "cost": 0.0
      , "tool_filter": VENICE_FIXED_TOOLS
    }
  , "Qwen": {
        # No coding trait (the operator's call): a coding request upgrades to Qwen Max.
        "normal": "qwen-3-8-27b"  # 262K Ctx, released Aug 17, 2026
      , "traits": ["vision", "nsfw", "reasoning", "concise answers"]
      , "cost": 0.10
      , "tool_filter": VENICE_FIXED_TOOLS
    }
  , "Qwen Max": {
        "normal": "qwen-3-8-2-4t-a95b"  # 262K Ctx, released Aug 12, 2026
      , "traits": ["coding", "nsfw", "reasoning", "thorough answers"]
      , "cost": 0.56
      , "tool_filter": VENICE_FIXED_TOOLS
    }

    # xAI
  , "Grok": {
        # Disabled, untested - 500K Ctx.
        "normal": "grok-4-7"  # released Sep 15, 2026
      , "traits": ["vision", "coding", "reasoning"]
      , "cost": 0.51
      , "disabled": True
    }

    # MiniMaxAI
  , "MiniMax": {
        # Disabled, untested - 512K Ctx.
        "normal": "minimax-m3-preview"  # released Jun 11, 2026
      , "traits": ["vision", "coding", "reasoning"]
      , "cost": 0.04
      , "disabled": True
    }

    # Xiaomi
  , "MiMo": {
        # Disabled, untested - 1M Ctx.
        "normal": "xiaomi-mimo-v2-5"  # released Jun 10, 2026
      , "traits": ["long context", "vision", "coding", "reasoning"]
      , "cost": 0.07
      , "disabled": True
    }

    # Inception
  , "Mercury": {
        # Disabled, untested - 260K Ctx.
        "normal": "mercury-2-5"  # released Sep 7, 2026
      , "traits": ["reasoning"]
      , "cost": -0.03
      , "disabled": True
    }

    # ByteDance/BytePlus
  , "Dola-Seed": {
        # Disabled, untested - 256K Ctx.
        "normal": "seed-2-1-turbo"  # released Jun 27, 2026
      , "traits": ["vision", "coding", "reasoning"]
      , "cost": 0.14
      , "disabled": True
    }

    # Mistral
  , "Mistral": {
        # Disabled, untested - 256K Ctx.
        "normal": "mistral-small-3-2-24b-instruct"  # released Jan 14, 2026
      , "traits": ["vision"]
      , "cost": -0.02
      , "disabled": True
    }

    # NVIDIA
  , "Nemotron Nano": {
        # Disabled, untested - 128K Ctx.
        "normal": "nvidia-nemotron-3-nano-30b-a3b"  # released Jan 26, 2026
      , "traits": []
      , "cost": -0.02
      , "disabled": True
    }
  , "Nemotron Ultra": {
        # Disabled, untested - 256K Ctx.
        "normal": "nvidia-nemotron-3-ultra-550b-a55b"  # released Jun 3, 2026
      , "traits": ["reasoning"]
      , "cost": 0.14
      , "disabled": True
    }

  , "Aion Mini": {
        # Model based on DeepSeek
        "normal": "aion-labs-aion-3-0-mini"  # released Jul 8, 2026
      , "traits": ["nsfw", "roleplay", "storytelling", "concise answers"]
      , "cost": 0.16
    }
  , "Aion": {
        # Model based on GLM-5.1
        "normal": "aion-labs-aion-3-0"  # released Jul 8, 2026
      , "traits": ["nsfw", "reasoning", "roleplay", "storytelling", "thorough answers"]
      , "cost": 0.80
    }

  , "Venice Uncensored": {
        "normal": "venice-uncensored-1-2"  # released Apr 1, 2026
      , "traits": ["vision", "nsfw", "roleplay"]
      , "cost": 0.01
      , "disabled": True # Spaz outs and repeat in loops the same 3 words. Broken!
    }
}
# The output token ceiling of each curated chat model (maxCompletionTokens of the live model list, read 2026-09-23).
# The request carries it as max_tokens: an absent cap lets the backend reserve an output default that can exceed
# the context of the model (minimax-m3-preview reserved 512000 of its 524288), so a small prompt already trips
# the context check. The cap equals the true ceiling of the model, so it cuts no generation short.
# An id missing here sends no cap and keeps the backend default.
VENICE_CHAT_COMPLETION_TOKENS = {
    "deepseek-v4-flash-0731": 32768
  , "deepseek-v4-pro": 32768
  , "google-gemma-4-31b-it": 8192
  , "gemma-4-uncensored": 8192
  , "gemini-3-8-flash": 65536
  , "z-ai-glm-5-3-flash": 131072
  , "z-ai-glm-5-3": 131072
  , "inkling": 65536
  , "kimi-k3": 131072
  , "qwen-3-8-flash": 131072
  , "qwen-3-8-27b": 65536
  , "qwen-3-8-2-4t-a95b": 65536
  , "aion-labs-aion-3-0-mini": 32768
  , "aion-labs-aion-3-0": 32768
  , "llama-3.2-3b": 4096
  , "llama-3.3-70b": 4096
  , "zai-org-glm-4.7-flash": 16384
  , "olafangensan-glm-4.7-flash-heretic": 24000
  , "venice-uncensored-1-2": 8192
  , "grok-4-7": 200000
  , "minimax-m3-preview": 65536
  , "xiaomi-mimo-v2-5": 65536
  , "mercury-2-5": 65536
  , "seed-2-1-turbo": 65536
  , "mistral-small-3-2-24b-instruct": 16384
  , "nvidia-nemotron-3-nano-30b-a3b": 16384
  , "nvidia-nemotron-3-ultra-550b-a55b": 32768
}
