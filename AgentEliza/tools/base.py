"""Shared constants and helpers of the harness tools."""

import re

import discord

# Cap of one tool result, uniform across the harness tools and the MCP
# results. 64K characters holds a full read_history page of 64 messages.
TOOL_RESULT_MAX_CHARS = 64_000
# The message time format everywhere: turn stamps in the context, read_history
# output, read_history after/before input. ISO 8601, UTC, minute precision.
MESSAGE_TIME_FORMAT = "%Y-%m-%dT%H:%MZ"
# The Discord file hosts. Their downloads take the bot token in the
# Authorization header. Send the token to these hosts only.
DISCORD_FILE_HOSTS = {"cdn.discordapp.com", "media.discordapp.net"}
# The audio the speech transcription reads: the Discord content types carry
# the prefix, an unknown type falls back to the file name suffix (the
# formats the transcription endpoint accepts).
AUDIO_SUFFIXES = (".wav", ".wave", ".flac", ".m4a", ".aac", ".mp4", ".mp3", ".ogg", ".oga", ".webm")
# The transcription endpoint answers 413 past this size. An audio file over
# it stays a plain attachment.
TRANSCRIBE_MAX_BYTES = 25 * 1024 * 1024


def attachments_text(attachments) -> str:
    """The attachments line of a message: name, content type, and URL per file."""
    if not attachments:
        return ""
    items = ", ".join(f"{name} ({kind or 'unknown type'}) <{url}>" for name, kind, url in attachments)
    return f"\n[attachments: {items}]"


def is_audio_attachment(attachment) -> bool:
    """True when the attachment is an audio file the transcription reads."""
    kind = (attachment.content_type or "").lower()
    if kind.startswith("audio/"):
        return True
    return kind in ("", "application/octet-stream") and attachment.filename.lower().endswith(AUDIO_SUFFIXES)


def poll_result_suffix(message) -> str:
    """The results line of a poll result notification: the embed holds the outcome."""
    for embed in message.embeds:
        if embed.type != "poll_result":
            continue
        fields = {field.name: field.value for field in embed.fields}
        if "poll_question_text" not in fields:
            continue
        # A tie has no victor fields.
        victor = fields.get("victor_answer_text")
        outcome = (
            f"winner: {victor} ({fields.get('victor_answer_votes', '?')} votes)"
            if victor else "no winner: a tie"
        )
        return f"\nPoll results for {fields['poll_question_text']!r}: {outcome}, total votes: {fields.get('total_votes', '?')}."
    return ""


# The title that marks the speech-transcription embed: the listener answers
# an audio message with a bot reply that carries it.
SPEECH_EMBED_TITLE = "Speech transcription"
# The embed description holds 4096 characters at most. Room stays for the
# truncation note.
SPEECH_EMBED_MAX_CHARS = 4000


def speech_embed(speaker, audio_time, text: str, duration: float, model: str) -> discord.Embed:
    """The speech-transcription embed the listener answers an audio message with.

    Every part the history views read back rides the embed: the speaker
    name and the user id (speech_embed_line below), and the text in the
    description. The author field and the timestamp carry the speaker for
    a human reader.
    """
    if len(text) > SPEECH_EMBED_MAX_CHARS:
        text = text[:SPEECH_EMBED_MAX_CHARS] + f"\n[truncated: {len(text) - SPEECH_EMBED_MAX_CHARS} characters dropped]"
    embed = discord.Embed(
        title=SPEECH_EMBED_TITLE
        , description=text or "(no speech detected)"
        , timestamp=audio_time
    )
    embed.set_author(name=speaker.display_name, icon_url=speaker.display_avatar.url)
    embed.add_field(name="User", value=str(speaker.id))
    seconds = f"{duration:.0f} s of audio, " if duration > 0 else ""
    embed.set_footer(text=f"{seconds}transcribed by {model}")
    return embed


def speech_embed_line(message) -> str | None:
    """The attributed line of one speech-transcription embed, or None.

    The embed is the only carrier of the speech in the history views: the
    bot reply it rides drops the audio message it answers (a bot reply and
    its target read as a harness notice), so the embed itself must hold
    the speaker. The line keeps the shape of a user history line minus the
    stamp: `name <@id>: text`. None when the message carries no such
    embed, or the embed misses its parts."""
    for embed in message.embeds:
        if embed.title != SPEECH_EMBED_TITLE:
            continue
        author = embed.author
        speaker = (getattr(author, "name", "") or "").strip() if author is not None else ""
        user_id = next((field.value.strip() for field in embed.fields if field.name == "User"), "")
        text = (embed.description or "").strip()
        if speaker and user_id.isdigit() and text:
            return f"{speaker} <@{user_id}>: {text}"
        return None
    return None


async def read_limited(response, limit: int) -> bytes:
    """Read the response body up to limit bytes.

    response.content.read(n) can return after the first network chunk, long
    before n bytes. The chunked loop reads the stream to the cap.
    """
    body = bytearray()
    async for chunk in response.content.iter_chunked(65536):
        body += chunk
        if len(body) > limit:
            return bytes(body[:limit])
    return bytes(body)


def _cap(text: str) -> str:
    """Truncate a tool result to the context-friendly cap."""
    if len(text) > TOOL_RESULT_MAX_CHARS:
        dropped = len(text) - TOOL_RESULT_MAX_CHARS
        text = text[:TOOL_RESULT_MAX_CHARS] + f"\n[truncated: {dropped} characters dropped]"
    return text


def expected_count(arguments: dict):
    """The validated expected count of an edit tool, or an error text."""
    raw = arguments.get("expected")
    if raw is None:
        return 1, None
    if isinstance(raw, float) and raw.is_integer():
        raw = int(raw)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        return None, "Error: expected must be a whole number of at least 1."
    return raw, None


# Quote-class canonicalization for the match pass of guarded_replace. Text
# written by a person carries typographic quotes and dashes while the agent
# emits ASCII, so an exact old_text misses by invisible characters. Every
# rule maps one character to one character: an index in the canonical form
# is the same index in the original, and the write can splice the original.
_QUOTE_RULES = tuple(
    (re.compile(pattern), replacement)
    for pattern, replacement in (
        ("""[\u2018\u2019\u201a\u201b\u2032\u2035]""", chr(39))
        , ("""[\u201c\u201d\u201e\u201f\u2033\u2036]""", chr(34))
        , ("""[\u2012\u2013\u2014\u2015]""", chr(45))
        , ("""\u00a0""", chr(32))
    )
)


def _canonical(text: str) -> str:
    """The quote-canonical form of a text, for match comparison."""
    for pattern, replacement in _QUOTE_RULES:
        text = pattern.sub(replacement, text)
    return text


def guarded_replace(content: str, old_text: str, new_text: str, expected: int = 1):
    """Replace expected occurrences of old_text in content, or refuse.

    old_text must be non-empty. The count is the guard: a count that
    differs from expected returns (None, count, canonical) and the caller
    writes nothing. An exact miss retries in the quote-canonical form, and
    the replacement splices the original text at the canonical indices.
    Returns (new content, count, canonical).
    """
    count = content.count(old_text)
    canonical = False
    if count == 0:
        canon_content = _canonical(content)
        canon_needle = _canonical(old_text)
        count = canon_content.count(canon_needle) if canon_needle else 0
        canonical = count > 0
    if count != expected:
        return None, count, canonical
    if not canonical:
        replaced = content.replace(old_text, new_text, 1) if expected == 1 else content.replace(old_text, new_text)
        return replaced, count, canonical
    # The canonical mapping is one-to-one, so a canonical index is an
    # original index. Splice from the end so the earlier indices stay valid.
    indices = []
    start = 0
    while True:
        at = canon_content.find(canon_needle, start)
        if at == -1:
            break
        indices.append(at)
        start = at + len(canon_needle)
    out = content
    for at in reversed(indices):
        out = out[:at] + new_text + out[at + len(old_text):]
    return out, count, canonical
