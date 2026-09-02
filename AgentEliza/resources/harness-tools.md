# Harness tools reference

This file describes every tool of the AgentEliza harness: what it does, its parameters, its caps, and the texts it answers with.

The tool loop: one answer can hold many tool calls. Use a call, read the result, then call again or answer. The loop ends after 16 rounds of calls, so plan to finish within 10.

Two rules hold everywhere:

- Every tool result caps at 64000 characters. A longer result truncates and ends with a marker line that names the dropped count.
- A failed call does not crash anything. The failure returns as the tool result and starts with `Error:`. Read the text, adjust the call, and try again.

## Memory

The long-term memory has three scopes: `server` (one per Discord server), `channel` (one per channel), and `user` (one per person). Each scope holds two texts: the memory and the summary. The `kind` parameter selects: `memory` (the default) or `summary`. A stored text caps at 4000 characters.

| Tool | What it does |
| --- | --- |
| `memory_read` | Reads the memory or the summary of one scope. |
| `memory_write` | Replaces the whole text of one scope. |
| `memory_append` | Adds text at the end of one scope. |
| `memory_edit` | Replaces one passage of one scope, count-guarded. |

Shared parameters:

| Parameter | Applies to | Meaning |
| --- | --- | --- |
| `scope` | all four | `server`, `channel`, or `user`. Required. `server` is not available in a direct message. |
| `target` | all four | Blank keeps the active conversation. See Targets below. |
| `kind` | all four | `memory` (default) or `summary`. |
| `content` | write, append | The text to store. |
| `old_text` | edit | The exact passage to find. Required, non-empty. |
| `new_text` | edit | The replacement. An empty text deletes the passage. |
| `expected` | edit | The match count to replace, default 1. Give the true count to replace every match. |

Details:

- `memory_write` replaces the whole text. Read the scope first and merge the old facts in. An empty `content` erases the scope. Content over 4000 characters truncates. The result warns you when that happened: read the scope back and shorten it.
- `memory_append` warns when the scope is full and part of the text did not fit.
- `memory_edit` counts the matches of `old_text`. A count that differs from `expected` refuses the write and reports the found count. An exact miss retries in a quote-tolerant form, because people write typographic quotes while you write straight ones. The result marks that with `(quote-tolerant)`.

### Targets

- A blank `target` keeps the active conversation: the current channel or the current user. The `server` scope always means the current server, so it takes no target.
- Otherwise give a channel or a user of the current server: a Discord id, a mention (`<@id>`, `<#id>`), or an exact name. A user name matches the server display name, the global name, or the username. A channel name works with or without the `#`.
- Names resolve inside the current server only. No match or several matches returns an error that asks for the id.
- In a direct message the target stays blank. There, a non-owner reaches only the own `user` scope.

### Summaries

The channel and user summaries belong to the harness: the compaction writes them after long conversations. Read them when you need the earlier history. Do not rewrite them without a strong reason. The server summary belongs to you: keep it current yourself.

## read_history

Reads past messages of a channel, a thread, or the direct message. Oldest first.

| Parameter | Meaning |
| --- | --- |
| `target` | Blank reads the current channel. Otherwise a channel or thread of this server: id, `<#id>`, or exact name with or without the `#`. |
| `query` | Optional case-insensitive text filter over the scanned messages. |
| `limit` | 1 to 64 messages, default 20. |
| `after` | Optional UTC date or date-time, for example `2026-08-11` or `2026-08-11T14:30Z`. |
| `before` | Same format as `after`. |

- `after` and `before` together read the window between them. One alone reads the messages around that time. Neither reads the most recent messages.
- Discord has no search endpoint for bots. The `query` filters a bounded scan only: at most 400 raw messages, or 100 around a time point. When the scan hits the bound, the header tells you to narrow the time range.
- The result shows only the conversation with the agent: the messages of the bot, and the user messages that mention it or reply to it. Harness notices (error replies) stay out.
- Each message prints as one line: `2026-08-11T14:30Z Name <@id>: text`. The time is UTC, minute precision. Attachments add an `[attachments: ...]` line. A poll result adds its outcome line.
- The first line of the result is a header with the counts and every limit the read hit.

## send_file

Sends one file to the current conversation. Do not repeat the file content in your answer.

| Parameter | Meaning |
| --- | --- |
| `filename` | Required. The name with the extension. The harness reduces it to one clean part of at most 100 characters. |
| `content` | The full text of the file. |
| `path` | A workspace file written earlier. `path` overrides `content`. |
| `caption` | Optional text message sent with the file, at most 2000 characters. |

The size cap is the upload limit of the channel: 25 MiB in a direct message, more on boosted servers. Sending a workspace file refreshes its age.

## Workspace

One private folder per conversation. Use it for drafts, collected data, and work too large for one message.

Paths are unix-style and rooted at the folder: `/file.txt` is the file `file.txt`, and `notes/draft.txt` works too. A backslash reads as a separator. `..` past the root stays at the root. A path that tries to leave the folder answers `invalid path`.

Caps: 25 MiB per file, 250 MiB per folder. Files live about 3 weeks. The older a file gets, the more likely the harness deletes it. A read or a write of a file refreshes its age.

| Tool | Parameters | What it does |
| --- | --- | --- |
| `file_write` | `path`, `content` | Writes or overwrites a text file. |
| `file_append` | `path`, `content` | Adds text at the end. |
| `file_edit` | `path`, `old_text`, `new_text`, `expected` | The same count-guarded replace as `memory_edit`. UTF-8 text files only. |
| `file_move` | `path`, `destination` | Moves or renames inside the workspace. The destination must not exist. The root cannot move. A path cannot move into itself. |
| `file_read` | `path`, `offset`, `limit` | Reads a byte chunk, from `offset` (default 0), at most `limit` bytes (default 64000). The result names the bytes that remain. |
| `file_search` | `query`, `path` | Case-insensitive search. Each match gives the rooted path, the line number, and the byte offset. At most 50 matches. |
| `file_list` | `path` | Lists the files with sizes in bytes. An optional glob pattern filters, for example `*.txt` or `/notes/*.md`. A pattern without a folder part matches file names at any depth. At most 100 entries. |
| `attachment_fetch` | `url`, `path` | Downloads a message attachment into the workspace. Only the Discord file hosts work: use the URLs from the `[attachments]` lines. They expire about 24 hours after the message. The saved path defaults to the file name of the URL. Read the file after with `file_read`. |

## propose_choices

Posts a question with one button per choice. One open poll per conversation.

| Parameter | Meaning |
| --- | --- |
| `question` | The question to answer, at most 300 characters. |
| `choices` | 2 to 10 strings, each at most 55 characters. |
| `multiple` | Default `false`. Set `true` to allow more than one choice per person. |

The status of the open poll arrives prepended to your next messages in `[harness]` tags. With no vote for 5 minutes, the buttons become a native Discord poll that runs 24 hours. When a majority of the active users voted, the poll completes. In a server the report names who picked what: one line per voter while the voters stay within the number of choices, one line per choice past that, the counts only when the names grow long. In a direct message the report gives the counts: the one voter is the speaker. Do not repeat the choices in your answer.

## Web

- `web_search(query)`: searches with DuckDuckGo. Up to 8 numbered results, each with a title, a URL, and a snippet. Read a result with `web_fetch`.
- `web_fetch(url)`: reads one web page as text. An HTML page strips to its readable text. Plain text, JSON, and XML pass through. Other content types fail. The read caps at 1 MB. Attachment URLs on the Discord file hosts work: the fetch sends the bot authorization.

## MCP resources and server tools

- `list_resources(server)`: lists the resources of the connected MCP servers and of the built-in harness server. A resource is data published as context, for example a file or a schema. A blank `server` lists every server, the built-in set first. Each server prints one section with one line per resource: the uri, the name, the MIME type, and the description.
- `read_resource(server, uri)`: reads one resource. A text resource returns its text under a `# uri (mime)` head. A binary resource reports its type and its size.
- The built-in `harness` server holds your own reference files, this file included, under `harness:///` uris. No MCP connection stands behind it.
- Server tools: every connected MCP server contributes its tools under the name `server__tool`. The tool list of your context shows the names. A failed call drops the connection of that server, and the next call reconnects it.
- A provider can replace a harness tool with its own. The replacement keeps the harness name, so for example `web_search` can run on the provider server.
- `analyze_image(url, question)`: describes an image, reads its text, or answers a question about it. The default question describes the image. The attachment URLs of messages work.

The exact tool list of a conversation can differ from this reference: a provider replaces tools, and without the `mcp` package the MCP tools stay away.
