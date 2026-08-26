"""The send_file tool: an agent-written text file to the current channel."""

import asyncio
import io
import re

import discord

# Base Discord upload limit (25 MiB), for direct messages.
# A guild channel reports its own limit in guild.filesize_limit (boosted servers get more).
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
                                , "description": "The full text of the file. path overrides this value."
                            }
                            , "path": {
                                "type": "string"
                                , "description": "The workspace path of a file written earlier. Overrides content."
                            }
                            , "caption": {
                                "type": "string"
                                , "description": "An optional text message sent with the file."
                            }
                        }
                        , "required": ["filename"]
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
        path = arguments.get("path")
        content = arguments.get("content")
        caption = arguments.get("caption")
        if caption is not None and not isinstance(caption, str):
            return "Error: the caption must be a string."
        if caption and len(caption) > 2000:
            return "Error: the caption is over 2000 characters, the Discord message limit."
        if path is not None:
            # A workspace file: confined to the folder of this conversation.
            if not isinstance(path, str) or not path.strip():
                return "Error: the path must be a non-empty string."
            if self.workspace is None:
                return "Error: the workspace is not available."
            session_id = channel_id if guild_id is not None else user_id
            target = self.workspace.resolve(session_id, path.strip())
            if target is None:
                return "Error: the path escapes the workspace. Use a relative path inside it."
            if not target.is_file():
                return f"Error: the workspace has no file at {path!r}. List the files with file_list."
            try:
                data = await asyncio.to_thread(target.read_bytes)
            except OSError as e:
                return f"Error: the read failed: {e}."
            if not data:
                return "Error: the workspace file is empty."
        else:
            if not isinstance(content, str) or not content:
                return "Error: the content must be a non-empty string, or give the workspace path of a file."
            data = content.encode("utf-8", errors="replace")
        channel = await self.channel_getter(channel_id) if self.channel_getter else None
        if channel is None:
            return "Error: the current channel is unknown."
        guild = getattr(channel, "guild", None)
        if guild is not None and guild.me is not None and not channel.permissions_for(guild.me).attach_files:
            return "Error: I do not have the permission to attach files in this channel."
        limit = guild.filesize_limit if guild is not None else FILE_SEND_DM_MAX_BYTES
        if len(data) > limit:
            return (
                f"Error: the file holds {len(data)} bytes, over the Discord upload limit of this channel "
                f"({limit} bytes). Split the content into smaller files."
            )
        try:
            await channel.send(content=caption or None, file=discord.File(io.BytesIO(data), filename=name))
        except (discord.Forbidden, discord.HTTPException) as e:
            return f"Error: the file send failed: {e}."
        return f"The file {name} ({len(data)} bytes) has been sent."
