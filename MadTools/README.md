# MadTools

A set of utility commands for Red-DiscordBot. Part of [Mads-RedCogs](https://github.com/madrang/RedCogs).

## Commands

| Command | Permission | Description |
| --- | --- | --- |
| `[p]ping` | everyone | Shows the bot latency. Replaces the core `ping` on load and restores it on unload. |
| `[p]pingsite <url>` | mod | Pings the host of the URL through the `ping` command of the bot host. |
| `[p]resolvesite <url>` | mod | Resolves the host of the URL through `dig`. |

The URL needs a scheme, for example `https://example.com`.

## Requirements

`pingsite` and `resolvesite` shell out: the bot host needs `ping` and `dig` installed. No extra Python packages.

## Install

```
[p]cog install mads-redcogs MadTools
[p]load MadTools
```

This cog stores no user data.
