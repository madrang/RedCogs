# AgentEliza

An AI agent harness for Discord. The agent answers mentions and direct messages through an OpenAI-compatible chat API. MCP servers give it extra tools. Part of [Mads-RedCogs](https://github.com/madrang/RedCogs).

The default provider is Kimi Code (`https://api.kimi.com/coding/v1`, model `k3-256k`). Presets ship for Kimi Code, Kimi API, Z.AI Code, and Z.AI API. The host, model, and key are configurable.

## Features

- One conversation session per channel, one per user in direct messages. At most 3 live sessions run in parallel.
- Long-term memory at server, channel, and user scopes. The agent manages it through tools.
- Automatic compaction: a session that grows too large or idles too long is summarized. The summary persists across restarts, and a fresh session restores recent turns from the channel history.
- MCP tools: remote HTTP MCP servers join the tool list, added by command or shipped by the active provider.
- Built-in tools: web search and page fetch, channel history reading, file sending, and interactive choice polls.
- Usage throttle at the provider and hourly rate limits per scope. Only the bot owner bypasses them.

## Requirements

- `mcp>=2` (Red installs it at cog install time)
- An API key for an OpenAI-compatible chat provider

## Setup

```
[p]load AgentEliza
[p]eliza setkey <your-api-key>
[p]eliza status
```

`[p]eliza seturl` switches the provider by preset name or by raw base URL. `[p]eliza providers` lists the presets.

## Talk to the agent

Mention the bot in a channel or send it a direct message. An empty mention reaches the agent as a poke. `[p]eliza forgetme` deletes your own user memory and your direct-message session.

## Commands

User commands:

| Command | Description |
| --- | --- |
| `[p]eliza forgetme` | Deletes your user memory, your summary, and your direct-message session. |
| `[p]eliza memory show [scope] [member]` | Shows memory and summaries. Non-admins see only their own user scope. |

Admin commands:

| Command | Description |
| --- | --- |
| `[p]eliza setkey <key>` | Stores the provider API key. The invoking message is deleted. |
| `[p]eliza seturl <preset\|url>` | Sets the chat API. `clear` resets to the defaults. |
| `[p]eliza providers` | Lists the provider presets. |
| `[p]eliza setmodel <name>` | Sets the model. Names outside the provider list are allowed with a notice. |
| `[p]eliza status` | Checks the API connection and shows the MCP and session counts. |
| `[p]eliza usage` | Shows the usage windows of the provider. |
| `[p]eliza setthreshold <0-100>` | Sets the usage percent where the cog stops answering. 0 disables the throttle. |
| `[p]eliza setlimit <scope> <count>` | Sets the hourly interaction limit of a scope (user, channel, server). 0 disables the limit. |
| `[p]eliza stats [member]` | Shows token totals and the current rate window per scope. |
| `[p]eliza setrules <text>` | Sets the server rules of the system prompt. |
| `[p]eliza memory clear <scope> [member]` | Clears one memory scope and drops its live session. |
| `[p]eliza sessions list` | Lists the live sessions (server and channel, or the DM user) with the usage stats of each scope. |
| `[p]eliza sessions close` | Compacts and drops all sessions, then answers the messages it would answer with a maintenance notice until a reload. |
| `[p]eliza mcp add <name> <url> [Header: value...]` | Adds a remote HTTP MCP server. Header pairs are stored and sent on every request. A command with headers is deleted. |
| `[p]eliza mcp remove <name>` | Removes a server definition and closes its session. |
| `[p]eliza mcp list` | Lists the servers with their state. Header values never print. |

Owner commands:

| Command | Description |
| --- | --- |
| `[p]eliza setdmrules <text>` | Sets the rules of the direct-message system prompt. |

## End-user data

The cog stores per-user memory, conversation summaries, usage stats, and rate windows. All of it is returned or deleted on an end-user data request of the bot.
