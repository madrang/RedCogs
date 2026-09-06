"""The import smoke tests: the tier B modules load under the stub package."""

import AgentEliza.llm_chat as llm_chat
import AgentEliza.providers.venice as venice


def test_engine_imports() -> None:
    assert callable(llm_chat.collapse_blank_lines)


def test_venice_provider_imports() -> None:
    assert callable(venice.walk_credits)
