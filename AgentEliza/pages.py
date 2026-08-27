"""Reply pagination: markdown-aware pages with a streaming diff.

Red's pagify splits at a character position and can cut a code block in
two. This pager keeps a fenced code block on one page when it fits, and
splits an oversized block into several complete code blocks instead: a
page never holds a broken fence.

paginate works on the full text each time and can diff against the
previous text (old_content): a page is flagged updated when it is new or
its content changed. A sealed page never changes when more text arrives,
so only the last page of a growing text is ever edited.
"""

import re
from typing import NamedTuple

PAGE_MAX_CHARS = 2000
# Room kept for split artifacts, like the shorten_by of Red's pagify.
PAGE_MARGIN = 8
# A fence is a line that starts with three backticks or more. The tilde
# form (~~~) is not handled: the chat models emit backticks.
_FENCE_RE = re.compile(r"(?m)^`{3,}[^\n]*")


class Page(NamedTuple):
    content: str
    updated: bool


def _scan(text: str) -> list:
    """Split the text into prose and code-block parts.

    A prose part is ("text", str). A code part is ("code", open_line,
    body, closed): open_line is the opening fence with its language tag,
    body the lines between the fences, closed False when the block has no
    closing fence. An unterminated block ends the scan.
    """
    parts = []
    pos = 0
    while pos < len(text):
        match = _FENCE_RE.search(text, pos)
        if match is None:
            parts.append(("text", text[pos:]))
            return parts
        if match.start() > pos:
            parts.append(("text", text[pos:match.start()]))
        line_end = text.find("\n", match.end())
        if line_end == -1:
            parts.append(("code", text[match.start():], "", False))
            return parts
        open_line = text[match.start():line_end]
        close = _FENCE_RE.search(text, line_end + 1)
        if close is None:
            parts.append(("code", open_line, text[line_end + 1:], False))
            return parts
        parts.append(("code", open_line, text[line_end + 1:close.start()], True))
        pos = close.end()
    return parts


def _seal(pages: list, page: str) -> None:
    """Close the open page. A whitespace-only page is dropped."""
    if page.strip():
        pages.append(page)


def _fill(pages: list, cur: str, text: str, budget: int) -> str:
    """Append prose to the open page, sealing a page at each break. Returns the open page."""
    while text:
        room = budget - len(cur)
        if room <= 0:
            _seal(pages, cur)
            cur = ""
            continue
        if len(text) <= room:
            return cur + text
        cut = text.rfind("\n", 0, room)
        if cut <= 0:
            cut = text.rfind(" ", 0, room)
        if cut <= 0:
            # No usable break in the window: a hard cut.
            piece, text = text[:room], text[room:]
        else:
            # The break character stays at the end of the sealed page.
            piece, text = text[:cut + 1], text[cut + 1:]
        _seal(pages, cur + piece)
        cur = ""
    return cur


def _fill_code(pages: list, cur: str, open_line: str, body: str, closed: bool, budget: int, final: bool) -> str:
    """Append a code block to the open page. Returns the open page.

    A block that fits the open page, or a fresh one, stays whole. A block
    too big for one page splits into several complete blocks: each page
    gets its own opening and closing fence. An unterminated block with
    final=True is closed at the end of the text.
    """
    if not closed and not final:
        # Held back: the caller sees this text when the block closes.
        return cur
    if body and not body.endswith("\n"):
        body += "\n"
    block = f"{open_line}\n{body}```"
    if len(block) <= budget - len(cur):
        return cur + block
    if len(block) <= budget:
        _seal(pages, cur)
        return block
    # Oversized: split the body over several complete blocks, on fresh
    # pages. The room covers the fences and one added newline.
    _seal(pages, cur)
    room = budget - len(open_line) - 5
    while body:
        if len(body) <= room:
            piece, body = body, ""
        else:
            cut = body.rfind("\n", 0, room)
            if cut <= 0:
                piece, body = body[:room], body[room:]
            else:
                piece, body = body[:cut + 1], body[cut + 1:]
        if not piece.endswith("\n"):
            piece += "\n"
        pages.append(f"{open_line}\n{piece}```")
    return ""


def _paginate(text: str, final: bool, budget: int) -> list:
    """The pages of the text. final=False holds back an unterminated code block."""
    pages = []
    cur = ""
    for part in _scan(text):
        if part[0] == "text":
            cur = _fill(pages, cur, part[1], budget)
        else:
            _, open_line, body, closed = part
            cur = _fill_code(pages, cur, open_line, body, closed, budget, final)
            if not closed and not final:
                break
    _seal(pages, cur)
    return pages


def paginate(text: str, old_content: str | None = None, *, final: bool = True, page_length: int = PAGE_MAX_CHARS) -> list:
    """The pages of the text, flagged against the pages of old_content.

    old_content is the accumulated text of the previous call. Its pages
    are rebuilt in the streaming form (final=False), the form they were
    sent in. A page is flagged updated when it is new or its content
    changed. final=False holds back an unterminated code block: the text
    after the opening fence joins the pages when the block closes.
    """
    budget = page_length - PAGE_MARGIN
    pages = _paginate(text, final, budget)
    if not old_content:
        return [Page(content, True) for content in pages]
    old = _paginate(old_content, False, budget)
    return [Page(content, index >= len(old) or old[index] != content) for index, content in enumerate(pages)]
