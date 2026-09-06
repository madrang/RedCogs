"""The engine pure functions."""

import pytest

from AgentEliza.llm_chat import collapse_blank_lines


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # A run of blank lines collapses to one blank line.
        ("Let me check.\n\n\n\n\nDone checking.", "Let me check.\n\nDone checking."),
        ("a\n\n\nb\n\n\nc", "a\n\nb\n\nc"),
        # A single blank line stays.
        ("a\n\nb", "a\n\nb"),
        ("a\nb", "a\nb"),
        # Whitespace-only lines count as blank.
        ("a\n \n \n \nb", "a\n\nb"),
        # Blank runs at both ends drop.
        ("\n\n\nLeading and trailing\n\n\n", "Leading and trailing"),
        ("   \n\n  spaced note  \n\n  ", "spaced note"),
        # A padded no-reply tag still matches the tag.
        ("[no-reply]\n\n\n", "[no-reply]"),
        ("line1\n\n\n\t\n \nline2", "line1\n\nline2"),
    ],
)
def test_collapse_blank_lines(raw: str, expected: str) -> None:
    assert collapse_blank_lines(raw) == expected
