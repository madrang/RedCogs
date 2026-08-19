# Mads-RedCogs

A personal repository of cogs for [Red-DiscordBot](https://github.com/Cog-Creators/Red-DiscordBot). MIT license.

## Cogs

| Cog | Description |
| --- | --- |
| **MadTools** | Utility commands: a custom `ping`, plus `pingsite` and `resolvesite` for network checks. |
| **FeedWatch** | Watches feed URLs and posts new entries to a channel as embeds. |
| **AgentEliza** | AI agent harness. Answers mentions and direct messages through an OpenAI-compatible chat API, with MCP servers as tools. |
| **SdrTools** | RTL-SDR tools: posts pictures of the RF spectrum around a frequency. |

## Requirements

- Red-DiscordBot 3.5.0 or newer
- Python 3.11 or newer
- Some cogs declare extra Python packages. Red installs them at cog install time.

## Install

In Discord, with the prefix of your bot:

```
[p]repo add mads-redcogs https://github.com/madrang/RedCogs
[p]cog install mads-redcogs <CogName>
[p]load <CogName>
```

See the README of each cog for its setup steps and commands.

## License

MIT. See [LICENSE](LICENSE).
