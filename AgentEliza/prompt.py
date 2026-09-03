"""The system message of a context: base prompt, place lines, rules, memory blocks."""

import contextlib
import discord

SYSTEM_PROMPT = (
    "The harness marks the context with simple delimiters:\n"
    "- A user message starts with the UTC time, the sender name, the mention id, and a colon, for example `2026-08-12T14:30Z Madrang <@491487179927978014>: hello`\n"
    "  The id lets you target that user with the memory tools or answer with a mention.\n"
    "- A memory note starts with `[memory NAME]` and ends with `[/memory]`. It shows the stored memory of that user.\n"
    "  The harness adds it before the first message of a user in this context.\n"
    "- A harness request starts with `[harness]` and ends with `[/harness]`. It arrives as a **user** message.\n"
    "  When the request asks to condense the conversation, answer with a summary of the conversation so far.\n"
    "\n"
    "Memory rules:\n"
    "- You have three memory scopes: `server`, `channel`, and `user`. Your memory tools read and write them.\n"
    "- Update the memory **often**. When you learn a durable fact, write it at once.\n"
    "- Keep the memory *organized*. Group the facts by topic, and remove a fact that is no longer true.\n"
    "- The channel and user summaries update through a harness request. The server summary changes only when you update it.\n"
    "\n"
    "The summary and the memory:\n"
    "- The summary condenses the older part of the conversation. You write it when the harness asks.\n"
    "  It keeps the thread of the conversation.\n"
    "- The memory holds the facts that you choose to keep. You write it at any time with the memory tools.\n"
    "  It survives across conversations.\n"
    "- Put a durable fact in the memory. Let the summary keep the flow of the conversation.\n"
    "- Do not copy memory content into a summary. The context already shows the memory notes. A copy wastes context space.\n"
    "- The harness preloads the memory in the system message and the summary as the first exchange after it.\n"
    "\n"
    "Conversation rules:\n"
    "- An empty message is a poke. The user wants your attention and said nothing.\n"
    "- To stay silent, answer with only `[no-reply]`. The harness then sends nothing.\n"
    "- When a user mentions a past event that you do not know, use the read_history tool to find the exchange.\n"
    "- When an answer needs information that you do not have, gather it with your tools before you answer.\n"
    "- One answer can hold many tool calls. Search, validate what you find, and explore what your tools can do.\n"
    "- After 10 tool calls in one answer, finish the answer in text without tools.\n"
    "\n"
    "Discord renders your answers. You can use markdown: **bold**, *italics*, __underline__, ~~strikethrough~~, `code`, code blocks, quotes, lists, and # headers.\n"
    "You can also use the Discord forms: ||spoiler||, -# subtext, [masked links](https://url), and <t:UNIX:R> timestamps.\n"
    "Tables do not work on Discord. Use a list or a code block.\n"
    "A mention pings its target: <@USER_ID> for a user, <@&ROLE_ID> for a role, @here for the online members."
    "List the detailed documentation with **list_resources** and read it with **read_resource**.\n"
    "\n"
    "Now connected to Discord chat! Welcome {name}!\n"
)

def _age_line(*, allowed: bool, reason: str | None = None) -> str:
    """The age-restriction status line of the system message. Each case has
    its own full wording, per the STE rules of the vault."""
    if allowed:
        return (
            f"Age restriction: {reason or 'this conversation permits adult content'}. "
            "You can send adult content in this conversation."
        )
    return (
        "Age restriction: this conversation is not age-restricted. "
        "Do not send adult content."
    )


async def place_block(bot, guild_id, channel_id, is_owner: bool = False) -> str:
    """The location lines of the system message: server and channel, the
    server description on its own line, then the age-restriction status.
    The status mirrors channel_nsfw: the channel flag, the parent channel
    of a thread, an age-restricted guild, or the direct message of the bot
    owner."""
    if guild_id is None:
        # A direct message has no server or channel object to describe.
        lines = ["Direct message: a private conversation with the user."]
        lines.append(_age_line(
            allowed=is_owner
            , reason="this direct message belongs to the bot owner" if is_owner else None
        ))
        return "\n".join(lines)
    lines = []
    guild = bot.get_guild(guild_id)
    if guild is not None:
        lines.append(f"Server: {guild.name}")
    channel = bot.get_channel(channel_id)
    if channel is None:
        # A cache miss must not drop the line: ask the API. A lookup
        # failure only drops the channel line, never the context build.
        with contextlib.suppress(discord.NotFound, discord.Forbidden, discord.HTTPException):
            channel = await bot.fetch_channel(channel_id)
    # A thread is not a GuildChannel: the guild attribute covers both.
    if getattr(channel, "guild", None) is not None:
        line = f"Channel: #{channel.name}"
        topic = getattr(channel, "topic", None)
        if topic:
            line += f" — {' '.join(topic.split())}"
        lines.append(line)
    description = getattr(guild, "description", None) if guild is not None else None
    if description:
        # The description line sits right before the age line, so the two
        # close the block as one unit.
        lines.append(f"Server description: {' '.join(description.split())}")
    parent = getattr(channel, "parent", None)
    gated = bool(
        getattr(channel, "nsfw", False)
        or getattr(parent, "nsfw", False)
        or getattr(getattr(channel, "guild", None) or getattr(parent, "guild", None) or guild, "nsfw_level", None)
        == discord.NSFWLevel.age_restricted
    )
    lines.append(_age_line(
        allowed=gated
        , reason="this channel is age-restricted" if gated else None
    ))
    return "\n".join(lines)


def system_text(bot_name: str, memory_entries: list, rules_block: str = "", place: str = "") -> str:
    """The system message of a context: prompt, place, rules, memory blocks."""
    text = SYSTEM_PROMPT.format(name=bot_name)
    if place:
        text += f"\n\n{place}"
    if rules_block:
        text += f"\n\n{rules_block}"
    for label, memory in memory_entries:
        text += f"\n\n{label} memory:\n{memory}"
    return text
