# SdrTools

RTL-SDR tools. Captures the RF spectrum around a frequency with an RTL-SDR device on the bot host and replies with a rendered picture. Part of [Mads-RedCogs](https://github.com/madrang/RedCogs).

## Commands

| Command | Permission | Description |
| --- | --- | --- |
| `[p]spectrum <freq_mhz>` | everyone | Captures I/Q samples around the frequency in MHz, averages an FFT power spectrum, and posts it as a PNG. Example: `[p]spectrum 96.9`. |

Captures are serialized: one capture runs at a time.

## Requirements

- An RTL-SDR dongle on the bot host
- The Python packages `pyrtlsdr`, `numpy`, and `matplotlib` (Red installs them at cog install time)

## Install

```
[p]cog install mads-redcogs SdrTools
[p]load SdrTools
[p]spectrum 96.9
```

This cog stores no user data.
