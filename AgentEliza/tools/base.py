"""Shared constants and helpers of the harness tools."""

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
