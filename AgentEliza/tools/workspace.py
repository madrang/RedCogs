"""The workspace tools: the files of the per-session folder in the OS temp dir."""

import asyncio
import fnmatch
from urllib.parse import urlparse

import aiohttp

from ..workspace import WORKSPACE_FILE_MAX_BYTES, WORKSPACE_SESSION_MAX_BYTES
from .base import DISCORD_FILE_HOSTS, TOOL_RESULT_MAX_CHARS, _cap, expected_count, guarded_replace

# Default and cap of one file_read chunk.
FILE_READ_DEFAULT_BYTES = TOOL_RESULT_MAX_CHARS
# Cap of the matches of one file_search.
FILE_SEARCH_MAX_MATCHES = 50
# One attachment download takes at most this long.
ATTACHMENT_FETCH_TIMEOUT = aiohttp.ClientTimeout(total=60)


class WorkspaceTools:
    """The workspace tools: file_write, file_append, file_edit, file_read, file_search, file_list, attachment_fetch."""

    def workspace_tools(self) -> list:
        """The OpenAI function schemas of the workspace tools."""
        return [
            {
                "type": "function"
                , "function": {
                    "name": "file_write"
                    , "description": (
                        "Write a text file in the workspace of this conversation. Overwrites an existing file. "
                        "The workspace is temporary: the harness deletes the folder after 7 days without use."
                    )
                    , "parameters": {
                        "type": "object"
                        , "properties": {
                            "path": {
                                "type": "string"
                                , "description": "The relative path inside the workspace, for example notes/draft.txt."
                            }
                            , "content": {
                                "type": "string"
                                , "description": "The full text of the file."
                            }
                        }
                        , "required": ["path", "content"]
                    }
                }
            }
            , {
                "type": "function"
                , "function": {
                    "name": "file_append"
                    , "description": "Append text to a file in the workspace."
                    , "parameters": {
                        "type": "object"
                        , "properties": {
                            "path": {
                                "type": "string"
                                , "description": "The relative path inside the workspace."
                            }
                            , "content": {
                                "type": "string"
                                , "description": "The text to append."
                            }
                        }
                        , "required": ["path", "content"]
                    }
                }
            }
            , {
                "type": "function"
                , "function": {
                    "name": "file_edit"
                    , "description": (
                        "Replace one passage of a text file in the workspace. "
                        "The edit counts the matches and refuses a wrong count."
                    )
                    , "parameters": {
                        "type": "object"
                        , "properties": {
                            "path": {
                                "type": "string"
                                , "description": "The relative path inside the workspace."
                            }
                            , "old_text": {
                                "type": "string"
                                , "description": "The exact passage to find."
                            }
                            , "new_text": {
                                "type": "string"
                                , "description": "The replacement text. An empty text deletes the passage."
                            }
                            , "expected": {
                                "type": "integer"
                                , "default": 1
                                , "description": "The number of matches to replace. Give the true count to replace every match."
                            }
                        }
                        , "required": ["path", "old_text", "new_text"]
                    }
                }
            }
            , {
                "type": "function"
                , "function": {
                    "name": "file_read"
                    , "description": (
                        "Read a chunk of a workspace file. Byte offsets: file_list shows the sizes. "
                        "The result says when more bytes remain."
                    )
                    , "parameters": {
                        "type": "object"
                        , "properties": {
                            "path": {
                                "type": "string"
                                , "description": "The relative path inside the workspace."
                            }
                            , "offset": {
                                "type": "integer"
                                , "default": 0
                                , "description": "The byte offset to start at."
                            }
                            , "limit": {
                                "type": "integer"
                                , "default": FILE_READ_DEFAULT_BYTES
                                , "description": f"The maximum bytes to read, at most {FILE_READ_DEFAULT_BYTES}."
                            }
                        }
                        , "required": ["path"]
                    }
                }
            }
            , {
                "type": "function"
                , "function": {
                    "name": "file_search"
                    , "description": (
                        "Search the workspace files for a text, case-insensitive. "
                        "Each match comes with the path, the line number, and the byte offset: "
                        "read around a match with file_read from that offset."
                    )
                    , "parameters": {
                        "type": "object"
                        , "properties": {
                            "query": {
                                "type": "string"
                                , "description": "The text to find."
                            }
                            , "path": {
                                "type": "string"
                                , "description": "Optional. Search only this file or folder of the workspace."
                            }
                        }
                        , "required": ["query"]
                    }
                }
            }
            , {
                "type": "function"
                , "function": {
                    "name": "file_list"
                    , "description": "List the files of the workspace with their sizes in bytes."
                    , "parameters": {
                        "type": "object"
                        , "properties": {
                            "path": {
                                "type": "string"
                                , "description": (
                                    "Optional. A glob pattern for the paths to list, for example *.txt or notes/*.md. "
                                    "A pattern without a folder part matches file names at any depth."
                                )
                            }
                        }
                    }
                }
            }
            , {
                "type": "function"
                , "function": {
                    "name": "attachment_fetch"
                    , "description": (
                        "Download an attachment into the workspace. Then read the file with file_read."
                    )
                    , "parameters": {
                        "type": "object"
                        , "properties": {
                            "url": {
                                "type": "string"
                                , "description": (
                                    "The http(s) URL of the attachment. Only the Discord file hosts work: "
                                    "the URLs from the [attachments] lines."
                                )
                            }
                            , "path": {
                                "type": "string"
                                , "description": "The workspace path to save as. Default: the file name of the URL."
                            }
                        }
                        , "required": ["url"]
                    }
                }
            }
        ]

    def _target(self, session_id: int, path):
        """The confined workspace path of a tool argument, or an error text."""
        if not isinstance(path, str) or not path.strip():
            return None, "Error: the path must be a non-empty string."
        target = self.workspace.resolve(session_id, path.strip())
        if target is None:
            return None, "Error: the path escapes the workspace. Use a relative path inside it."
        return target, None

    def _session_id(self, guild_id, channel_id, user_id) -> int:
        """The workspace of the conversation: the channel, or the user in direct messages."""
        return channel_id if guild_id is not None else user_id

    @staticmethod
    def _encode(content) -> bytes:
        """The content as UTF-8 bytes. JSON-decoded arguments can hold lone surrogates."""
        return content.encode("utf-8", errors="replace")

    def _write(self, session_id: int, target, data: bytes, append: bool) -> None:
        self.workspace.make_folder(session_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        if append:
            with target.open("ab") as file:
                file.write(data)
        else:
            target.write_bytes(data)
        self.workspace.touch(session_id)

    async def _quota_error(self, session_id: int, target, added: int, append: bool) -> str | None:
        """The error text when added bytes break a cap, else None."""
        existing = 0
        if target.is_file():
            stat = await asyncio.to_thread(target.stat)
            existing = stat.st_size
        if existing + added > WORKSPACE_FILE_MAX_BYTES:
            return f"Error: the file would hold {existing + added} bytes, over the {WORKSPACE_FILE_MAX_BYTES}-byte file cap."
        folder_size = await asyncio.to_thread(self.workspace.session_size, session_id)
        # An overwrite frees the bytes of the old file, an append keeps them.
        replaced = 0 if append else existing
        if folder_size - replaced + added > WORKSPACE_SESSION_MAX_BYTES:
            return f"Error: the workspace of this conversation is full ({WORKSPACE_SESSION_MAX_BYTES} bytes)."
        return None

    async def _tool_file_write(self, arguments: dict, *, guild_id, channel_id, user_id) -> str:
        session_id = self._session_id(guild_id, channel_id, user_id)
        target, error = self._target(session_id, arguments.get("path"))
        if error:
            return error
        content = arguments.get("content")
        if not isinstance(content, str):
            return "Error: the content must be a string."
        data = self._encode(content)
        error = await self._quota_error(session_id, target, len(data), False)
        if error:
            return error
        try:
            await asyncio.to_thread(self._write, session_id, target, data, False)
        except OSError as e:
            return f"Error: the write failed: {e}."
        return f"The file {target.name} ({len(data)} bytes) has been written."

    async def _tool_file_append(self, arguments: dict, *, guild_id, channel_id, user_id) -> str:
        session_id = self._session_id(guild_id, channel_id, user_id)
        target, error = self._target(session_id, arguments.get("path"))
        if error:
            return error
        content = arguments.get("content")
        if not isinstance(content, str) or not content:
            return "Error: the content must be a non-empty string."
        data = self._encode(content)
        error = await self._quota_error(session_id, target, len(data), True)
        if error:
            return error
        try:
            await asyncio.to_thread(self._write, session_id, target, data, True)
        except OSError as e:
            return f"Error: the append failed: {e}."
        return f"Appended {len(data)} bytes to {target.name}."

    async def _tool_file_edit(self, arguments: dict, *, guild_id, channel_id, user_id) -> str:
        session_id = self._session_id(guild_id, channel_id, user_id)
        target, error = self._target(session_id, arguments.get("path"))
        if error:
            return error
        old_text = arguments.get("old_text")
        if not isinstance(old_text, str) or not old_text:
            return "Error: the old text must be a non-empty string."
        new_text = arguments.get("new_text")
        if not isinstance(new_text, str):
            return "Error: the new text must be a string."
        expected, error = expected_count(arguments)
        if error:
            return error
        if not target.is_file():
            return f"Error: the workspace has no file at {arguments.get('path')!r}."
        try:
            data = await asyncio.to_thread(target.read_bytes)
        except OSError as e:
            return f"Error: the read failed: {e}."
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return "Error: the file is not UTF-8 text. Only text files can be edited."
        replaced, count, canonical = guarded_replace(text, old_text, new_text, expected)
        if replaced is None:
            return f"Error: found {count} matches, expected {expected}. Nothing was written."
        new_data = self._encode(replaced)
        if len(new_data) > WORKSPACE_FILE_MAX_BYTES:
            return f"Error: the file would hold {len(new_data)} bytes, over the {WORKSPACE_FILE_MAX_BYTES}-byte file cap."
        folder_size = await asyncio.to_thread(self.workspace.session_size, session_id)
        # The edit rewrites the file, so the old bytes leave the folder total.
        if folder_size - len(data) + len(new_data) > WORKSPACE_SESSION_MAX_BYTES:
            return f"Error: the workspace of this conversation is full ({WORKSPACE_SESSION_MAX_BYTES} bytes)."
        try:
            await asyncio.to_thread(self._write, session_id, target, new_data, False)
        except OSError as e:
            return f"Error: the write failed: {e}."
        matches = "the match" if expected == 1 else f"{expected} matches"
        note = " (quote-tolerant)" if canonical else ""
        return f"Replaced {matches}{note} in {target.name}. The file now holds {len(new_data)} bytes."

    async def _tool_file_read(self, arguments: dict, *, guild_id, channel_id, user_id) -> str:
        session_id = self._session_id(guild_id, channel_id, user_id)
        target, error = self._target(session_id, arguments.get("path"))
        if error:
            return error
        try:
            offset = max(0, int(arguments.get("offset") or 0))
            limit = int(arguments.get("limit") or FILE_READ_DEFAULT_BYTES)
        except (TypeError, ValueError):
            return "Error: the offset and the limit must be integers."
        limit = max(1, min(limit, FILE_READ_DEFAULT_BYTES))
        if not target.is_file():
            return f"Error: no file at {arguments.get('path')!r} in the workspace. List the files with file_list."

        def _read() -> bytes:
            with target.open("rb") as file:
                file.seek(offset)
                return file.read(limit)

        try:
            data = await asyncio.to_thread(_read)
            size = await asyncio.to_thread(target.stat)
        except OSError as e:
            return f"Error: the read failed: {e}."
        text = data.decode("utf-8", errors="replace")
        remaining = size.st_size - offset - len(data)
        if remaining > 0:
            text += f"\n[{remaining} bytes remain: read again from offset {offset + len(data)}]"
        return text or f"(the file is empty from offset {offset})"

    @staticmethod
    def _search(folder, scope, query: str) -> list:
        """The matching lines as (relative path, line number, byte offset, text).

        The byte offsets come from the raw split, so a decode for the
        display text never shifts them: file_read from the offset lands on
        the line of the match.
        """
        if scope.is_file():
            files = [scope]
        else:
            files = sorted(p for p in scope.rglob("*") if p.is_file())
        matches = []
        for path in files:
            try:
                body = path.read_bytes()
            except OSError:
                continue
            offset = 0
            for number, line in enumerate(body.split(b"\n"), 1):
                if query.casefold() in line.decode("utf-8", errors="replace").casefold():
                    text = " ".join(line.decode("utf-8", errors="replace").split())[:200]
                    matches.append((str(path.relative_to(folder)), number, offset, text))
                    if len(matches) >= FILE_SEARCH_MAX_MATCHES:
                        return matches
                offset += len(line) + 1
        return matches

    async def _tool_file_search(self, arguments: dict, *, guild_id, channel_id, user_id) -> str:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return "Error: the query must be a non-empty string."
        session_id = self._session_id(guild_id, channel_id, user_id)
        folder = self.workspace.folder(session_id)
        scope = folder
        path = arguments.get("path")
        if path is not None and str(path).strip():
            scope, error = self._target(session_id, path)
            if error:
                return error
            if not scope.exists():
                return f"Error: no file or folder at {path!r} in the workspace."
        matches = await asyncio.to_thread(self._search, folder, scope, query.strip())
        if not matches:
            return "(no matches in the workspace)"
        lines = [f"{path}: line {number} (byte {offset}): {text}" for path, number, offset, text in matches]
        head = f"({len(matches)} match{'es' if len(matches) != 1 else ''}"
        if len(matches) >= FILE_SEARCH_MAX_MATCHES:
            head += f", capped at {FILE_SEARCH_MAX_MATCHES}: refine the query or narrow the path"
        return _cap(head + ")\n" + "\n".join(lines))

    async def _tool_file_list(self, arguments: dict, *, guild_id, channel_id, user_id) -> str:
        session_id = self._session_id(guild_id, channel_id, user_id)
        pattern = arguments.get("path")
        if pattern is not None and (not isinstance(pattern, str) or not pattern.strip()):
            return "Error: the path pattern must be a non-empty string."

        def _list() -> list:
            folder = self.workspace.folder(session_id)
            if not folder.exists():
                return []
            files = sorted(
                (str(p.relative_to(folder)), p.stat().st_size)
                for p in folder.rglob("*") if p.is_file()
            )
            if pattern:
                # The match runs on the confined relative path: a pattern
                # can never reach outside the session folder.
                files = [item for item in files if fnmatch.fnmatch(item[0], pattern.strip())]
            return files

        files = await asyncio.to_thread(_list)
        if not files:
            if pattern:
                return f"(no files match {pattern.strip()!r})"
            return "(the workspace is empty)"
        lines = [f"{path} ({size} bytes)" for path, size in files[:100]]
        if len(files) > 100:
            lines.append(f"[and {len(files) - 100} more]")
        return "\n".join(lines)

    async def _tool_attachment_fetch(self, arguments: dict, *, guild_id, channel_id, user_id) -> str:
        url = arguments.get("url")
        if not isinstance(url, str) or not url.strip():
            return "Error: the url must be a non-empty string."
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            return "Error: only http and https URLs can be fetched."
        if urlparse(url).netloc.lower() not in DISCORD_FILE_HOSTS:
            return "Error: only attachments on the Discord file hosts can be downloaded: the URLs from the [attachments] lines."
        session_id = self._session_id(guild_id, channel_id, user_id)
        name = arguments.get("path")
        if not isinstance(name, str) or not name.strip():
            name = urlparse(url).path.rsplit("/", 1)[-1] or "attachment"
        target, error = self._target(session_id, name)
        if error:
            return error
        # The Discord file hosts need an authorized request. The bot token
        # goes to these hosts only, never to a foreign URL.
        headers = {}
        token = self.bot_token_getter() if self.bot_token_getter else None
        if token:
            headers["Authorization"] = f"Bot {token}"
        folder_size = await asyncio.to_thread(self.workspace.session_size, session_id)
        overflow = False
        try:
            async with self.session_getter().get(url, headers=headers, timeout=ATTACHMENT_FETCH_TIMEOUT) as response:
                if response.status != 200:
                    return f"Error: the download failed (HTTP {response.status}). The URL may have expired."
                await asyncio.to_thread(self.workspace.make_folder, session_id)
                await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)
                handle = await asyncio.to_thread(target.open, "wb")
                written = 0
                try:
                    async for chunk in response.content.iter_chunked(65536):
                        written += len(chunk)
                        if written > WORKSPACE_FILE_MAX_BYTES or folder_size + written > WORKSPACE_SESSION_MAX_BYTES:
                            overflow = True
                            break
                        await asyncio.to_thread(handle.write, chunk)
                finally:
                    await asyncio.to_thread(handle.close)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            return f"Error: the download failed: {e}."
        if overflow:
            # A cap error inside the stream leaves a partial file behind.
            await asyncio.to_thread(target.unlink, missing_ok=True)
            if written > WORKSPACE_FILE_MAX_BYTES:
                return f"Error: the attachment is over the {WORKSPACE_FILE_MAX_BYTES}-byte file cap."
            return f"Error: the workspace of this conversation is full ({WORKSPACE_SESSION_MAX_BYTES} bytes)."
        await asyncio.to_thread(self.workspace.touch, session_id)
        return f"The attachment is saved as {target.name} ({written} bytes). Read it with file_read."
