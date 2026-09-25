# The audio generation flow of the Venice API: the price quote and the asynchronous
# queue, poll, complete cycle of one song.

import asyncio
import logging
import random

from ...llm_chat import ChatError
from .catalog import VENICE_MUSIC_FORMATS, VENICE_MUSIC_TIMEOUT

log = logging.getLogger("red.agenteliza")

# The first poll wait after the queue call, and the cap of one poll wait.
MUSIC_POLL_FIRST_SECONDS = 2
MUSIC_POLL_MAX_SECONDS = 30


async def quote_song(api_post, body: dict) -> float:
    """The live USD price of one queue body (POST /audio/quote, its answers round to cents).
       The quote endpoint takes the model and the duration alone: the prompt
       and the other dials left its schema, so they stay out of the call."""
    quote_body = {key: body[key] for key in ("model", "duration_seconds") if key in body}
    data, _headers = await api_post("/audio/quote", json_body=quote_body)
    quote = data.get("quote") if isinstance(data, dict) else None
    if not isinstance(quote, (int, float)) or quote < 0:
        raise ChatError("http", f"The quote endpoint returned no price: {str(data)[:200]}")
    return float(quote)


async def queue_song(api_post, body: dict) -> tuple[str, str]:
    """Start one generation (POST /audio/queue). Return the model and the queue id."""
    data, _headers = await api_post("/audio/queue", json_body=body)
    model = str(data.get("model") or body.get("model") or "song") if isinstance(data, dict) else "song"
    queue_id = data.get("queue_id") if isinstance(data, dict) else None
    if not queue_id:
        raise ChatError("http", f"The queue endpoint returned no queue id: {str(data)[:200]}")
    return model, str(queue_id)


async def retrieve_song(api_post, model: str, queue_id: str) -> tuple[str, bytes]:
    """Poll one queued generation to its audio (POST /audio/retrieve).

    A JSON body means still processing and carries average_execution_time in
    milliseconds, which paces the next poll. A binary body is the finished
    audio: the download posts through the caller, then POST /audio/complete
    closes the generation. The content type names the file extension. The
    whole poll loop runs inside the VENICE_MUSIC_TIMEOUT budget."""
    deadline = asyncio.get_running_loop().time() + VENICE_MUSIC_TIMEOUT
    wait = MUSIC_POLL_FIRST_SECONDS
    while True:
        data, headers = await api_post(
            "/audio/retrieve", json_body={"model": model, "queue_id": queue_id}, binary=True, timeout=60
        )
        if isinstance(data, (bytes, bytearray)) and data:
            content_type = str(headers.get("content-type") or "").split(";")[0].strip().lower()
            extension = VENICE_MUSIC_FORMATS.get(content_type, "mp3")
            try:
                await api_post("/audio/complete", json_body={"model": model, "queue_id": queue_id}, timeout=60)
            except ChatError as e:
                # The audio is in hand: the cleanup call failed only.
                log.warning("The complete call of queue %s failed: %s", queue_id, e)
            name = f"{model}-{random.randint(0, 0xFFFF):04x}.{extension}"
            return name, bytes(data)
        average = data.get("average_execution_time") if isinstance(data, dict) else None
        if isinstance(average, (int, float)) and average > 0:
            wait = min(MUSIC_POLL_MAX_SECONDS, max(MUSIC_POLL_FIRST_SECONDS, average / 1000 / 2))
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise ChatError(
                "connection"
                , f"The song generation ran past its {VENICE_MUSIC_TIMEOUT} s time budget without audio."
            )
        await asyncio.sleep(min(wait, remaining))


async def run_song(api_post, body: dict) -> tuple[str, bytes]:
    """The one-shot flow of one queue body: queue, poll, complete."""
    model, queue_id = await queue_song(api_post, body)
    return await retrieve_song(api_post, model, queue_id)
