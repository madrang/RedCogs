# Discord markdown tutorial

Discord renders a subset of markdown in every message, plus formats of its own. This tutorial covers the full notation with examples. Write the source form shown here, and the client renders it for every reader.

## Text styles

| Source | Renders as |
| --- | --- |
| `**bold**` | **bold** |
| `*italic*` or `_italic_` | *italic* |
| `***bold italic***` | ***bold italic*** |
| `__underline__` | underline |
| `__***underline bold italic***__` | underline bold italic |
| `~~strikethrough~~` | strikethrough |
| `` `inline code` `` | `inline code` |
| `\|\|spoiler\|\|` | a black bar, the text appears when a reader clicks it |

The markers must touch the text. `** text **` stays literal.

Styles combine when they nest cleanly. Bold inside a list item works. A style marker never crosses a code boundary.

## Code

Inline code uses one backtick:

````
Type `pip install red-discordbot` to install.
````

A code block uses three backticks on their own lines:

````
```
def greet(name):
    return f"Hello {name}"
```
````

A language name after the opening fence turns on syntax highlighting:

````
```py
def greet(name):
    return f"Hello {name}"
```
````

Inside a code block nothing renders: no styles, no mentions, no spoilers, no links. To show a three-backtick fence inside a code block, fence the outer block with four backticks, as this tutorial does.

## Headers and subtext

````
# Big header
## Medium header
### Small header
-# Subtext, small and gray
````

Discord offers three header levels. The marker must stand at the start of the line, with one space after it. Subtext with `-#` also needs the start of the line.

## Quotes

````
> One quoted line.
````

A `>` quotes one line. A `>>>` quotes the rest of the message:

````
>>> Everything after this marker
is part of the quote.
````

The space after the marker is required.

## Lists

````
- First bullet
- Second bullet
  - Nested bullet
1. First step
2. Second step
````

`-` or `*` starts a bullet item. `1.` starts a numbered item. Indent with two spaces to nest. Lists and quotes combine: start a list line inside a quote with `> -`.

## Links

````
https://example.com
[Example](https://example.com)
<https://example.com>
````

- A bare URL renders as a link with a preview of the page.
- `[text](URL)` is a masked link. Only the text shows.
- `<URL>` suppresses the preview.

## Mentions and emoji

| Source | Renders as |
| --- | --- |
| `<@123456789012345678>` | a user name, colored |
| `<#123456789012345678>` | a channel name, links to the channel |
| `<@&123456789012345678>` | a role name, colored |
| `<:nameofemoji:123456789012345678>` | a custom emoji of this server |
| `<a:nameofemoji:123456789012345678>` | an animated custom emoji |

- Standard emoji are plain unicode characters. Copy one straight into the text.
- Every message stamp in your context carries the id of the speaker: `2026-08-11T14:30Z Name <@123456789012345678>:`. Use that id for a user mention.
- A mention can notify the user. Ping only when the person asked for it.
- The form `<@!id>` for users is deprecated. Use `<@id>`.

## Timestamps

`<t:SECONDS>` renders a time in the timezone of every reader. SECONDS is a unix timestamp in seconds, not milliseconds. An optional style letter selects the format: `<t:1767225600:F>`.

| Style | Renders as |
| --- | --- |
| none | short date with short time |
| `t` | short time, for example `12:30` |
| `T` | medium time with seconds |
| `d` | short date |
| `D` | long date |
| `f` | short date with short time |
| `F` | full date with short time, the most complete form |
| `s` | short date with short time |
| `S` | short date with medium time |
| `R` | relative, for example `2 hours ago` |

`R` is the clean way to state a duration or a deadline: the reader sees it fresh at every look.

## Slash commands and navigation

`</name:command_id>` mentions a slash command, and `</name subcommand:command_id>` a subcommand. Navigation links like `<id:customize>` open server menus. You rarely need these forms.

## Escaping

A backslash before a marker turns it off:

````
\*\*not bold\*\*
````

Inside code, no escape is needed: nothing renders there anyway.

## What breaks

- A space between the marker and the text kills the format.
- Headers and subtext work only at the start of a line.
- Only three header levels exist. `####` stays literal.
- A spoiler inside a code block shows as plain text.
- An unclosed marker pair stays literal. Close what you open.
- An empty line splits paragraphs. Two styles cannot span the split.

## Notes for long replies

The harness splits a long reply into pages at safe points. A code block stays whole on its page, and an oversized block becomes several complete blocks. Format freely: the splitter protects the notation.
