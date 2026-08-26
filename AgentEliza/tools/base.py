"""Shared constants and helpers of the harness tools."""

import re

# Cap of one tool result, uniform across the harness tools and the MCP
# results. 64K characters holds a full read_history page of 64 messages.
TOOL_RESULT_MAX_CHARS = 64_000
# The message time format everywhere: turn stamps in the context, read_history
# output, read_history after/before input. ISO 8601, UTC, minute precision.
MESSAGE_TIME_FORMAT = "%Y-%m-%dT%H:%MZ"
# The Discord file hosts. Their downloads take the bot token in the
# Authorization header. Send the token to these hosts only.
DISCORD_FILE_HOSTS = {"cdn.discordapp.com", "media.discordapp.net"}


def attachments_text(attachments) -> str:
    """The attachments line of a message: name, content type, and URL per file."""
    if not attachments:
        return ""
    items = ", ".join(f"{name} ({kind or 'unknown type'}) <{url}>" for name, kind, url in attachments)
    return f"\n[attachments: {items}]"


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
