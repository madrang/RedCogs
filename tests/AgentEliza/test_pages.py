"""The reply pager: whole code blocks, streaming flags, held-back fences."""

from AgentEliza.pages import paginate


def test_short_text_is_one_updated_page() -> None:
    assert [(page.content, page.updated) for page in paginate("hello")] == [("hello", True)]


def test_long_prose_splits_within_the_page_length() -> None:
    text = "\n".join(f"line {i} " + "x" * 80 for i in range(40))
    pages = paginate(text, page_length=200)
    assert len(pages) > 1
    assert all(len(page.content) <= 200 for page in pages)


def test_a_code_block_stays_whole() -> None:
    # The block fits a fresh page but not the open one after the intro.
    code = "\n".join(f"print({i})" for i in range(15))
    intro = "i" * 150
    text = f"{intro}\n```python\n{code}\n```"
    pages = paginate(text, page_length=200)
    assert len(pages) == 2
    block_page = pages[1].content
    assert block_page.startswith("```python")
    assert block_page.rstrip().endswith("```")
    for i in range(15):
        assert f"print({i})" in block_page


def test_an_oversized_block_splits_into_complete_blocks() -> None:
    body = "\n".join(f"line {i}" for i in range(60))
    pages = paginate(f"```text\n{body}\n```", page_length=300)
    assert len(pages) > 1
    for page in pages:
        assert page.content.startswith("```text")
        assert page.content.rstrip().endswith("```")


def test_an_unterminated_block_is_held_back_until_final() -> None:
    text = "before\n```python\nprint(1)"
    held = paginate(text, final=False)
    # The newline before the fence rides the held page.
    assert [page.content for page in held] == ["before\n"]
    closed = paginate(text)
    assert any("print(1)" in page.content for page in closed)


def test_a_sealed_page_is_not_flagged_again() -> None:
    head = "A" * 90 + "\n"
    pages = paginate(head + "B" * 90, head, page_length=100)
    assert [page.content for page in pages] == [head, "B" * 90]
    # The sealed first page did not change: the caller must not edit it again.
    assert [page.updated for page in pages] == [False, True]
