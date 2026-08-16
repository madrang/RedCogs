"""The send_file tool: an agent-written text file to the current channel."""

import io
import re

import discord

# Base Discord upload limit (25 MiB), for direct messages. A guild channel
# reports its own limit in guild.filesize_limit (boosted servers get more).
FILE_SEND_DM_MAX_BYTES = 26_214_400


class FileTools:
    """The send_file tool."""

    def file_tools(self) -> list:
        """The OpenAI function schema of the file tool."""
        return [
            {
                "type": "function"
                , "function": {
                    "name": "send_file"
                    , "description": (
                        "Send a file to the current conversation. Do not repeat the file content in your answer."
                    )
                    , "parameters": {
                        "type": "object"
                        , "properties": {
                            "filename": {
                                "type": "string"
                                , "description": "The file name with its extension."
                            }
                            , "content": {
                                "type": "string"
                                , "description": "The full text of the file."
                            }
                            , "caption": {
                                "type": "string"
                                , "description": "An optional text message sent with the file."
                            }
                        }
                        , "required": ["filename", "content"]
                    }
                }
            }
        ]

    async def _tool_send_file(self, arguments: dict, *, guild_id, channel_id, user_id) -> str:
        filename = arguments.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            return "Error: the filename must be a non-empty string."
        # The name is cosmetic on Discord: keep one clean path-less part.
        name = re.sub(r"[\\/\x00-\x1f]+", "_", filename).strip(" .")[:100]
        if not name:
            return "Error: the filename has no usable characters."
        content = arguments.get("content")
        if not isinstance(content, str) or not content:
            return "Error: the content must be a non-empty string."
        caption = arguments.get("caption")
        if caption is not None and not isinstance(caption, str):
            return "Error: the caption must be a string."
        if caption and len(caption) > 2000:
            return "Error: the caption is over 2000 characters, the Discord message limit."
        channel = await self.channel_getter(channel_id) if self.channel_getter else None
        if channel is None:
            return "Error: the current channel is unknown."
        guild = getattr(channel, "guild", None)
        if guild is not None and guild.me is not None and not channel.permissions_for(guild.me).attach_files:
            return "Error: I do not have the permission to attach files in this channel."
        data = content.encode("utf-8")
        limit = guild.filesize_limit if guild is not None else FILE_SEND_DM_MAX_BYTES
        if len(data) > limit:
            return (
                f"Error: the file holds {len(data)} bytes, over the Discord upload limit of this channel "
                f"({limit} bytes). Split the content into smaller files."
            )
        try:
            await channel.send(content=caption or None, file=discord.File(io.BytesIO(data), filename=name))
        except (discord.Forbidden, discord.HTTPException) as e:
            return f"Error: the file send failed: {e}"
        return f"The file {name} ({len(data)} bytes) has been sent."
