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


def _cap(text: str) -> str:
    """Truncate a tool result to the context-friendly cap."""
    if len(text) > TOOL_RESULT_MAX_CHARS:
        dropped = len(text) - TOOL_RESULT_MAX_CHARS
        text = text[:TOOL_RESULT_MAX_CHARS] + f"\n[truncated: {dropped} characters dropped]"
    return text
