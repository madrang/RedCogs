# FeedWatch

Watches feed URLs and posts new entries to a channel as embeds. Part of [Mads-RedCogs](https://github.com/madrang/RedCogs).

## How it works

Every 5 minutes the cog polls each watch-listed URL of the guild. A post with an id higher than the last posted id goes to the configured channel. The first poll only seeds the last id, so the backlog is never posted.

Each watch URL must return a JSON array of posts:

- `id` (integer)
- `title`
- `link` or `url`
- `excerpt` or `description`

Values are plain strings or WordPress-style `{"rendered": ...}` objects.

## Commands

| Command | Permission | Description |
| --- | --- | --- |
| `[p]setchannel <channel>` | manage channels | Sets the channel for the automated posts. |
| `[p]addsrc <url>` | everyone | Adds a URL to the watchlist of the guild. |

## Install

```
[p]cog install mads-redcogs FeedWatch
[p]load FeedWatch
[p]setchannel #updates
[p]addsrc https://example.com/feed.json
```

No extra Python packages. This cog stores no user data.
