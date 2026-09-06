"""The test bootstrap of the repo.

The tests import the engine and the provider modules without the cog
package init: `AgentEliza/__init__.py` pulls redbot, and redbot is not a
test dependency. The stub package registered here carries the real
`__path__`, so every submodule (llm_chat, providers, tools) resolves its
relative imports as usual. The init itself never runs.
"""

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).parent


def _stub_cog_package() -> None:
    if "AgentEliza" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        "AgentEliza"
        , REPO / "AgentEliza" / "__init__.py"
        , submodule_search_locations=[str(REPO / "AgentEliza")]
    )
    sys.modules["AgentEliza"] = importlib.util.module_from_spec(spec)


_stub_cog_package()
