from .kimi_api import KimiApiProvider
from .kimi_code import KimiCodeProvider
from .zai import ZaiApiProvider, ZaiCodeProvider

# The first PROVIDERS entry is the default provider.
PROVIDERS = [
    KimiCodeProvider(),
    KimiApiProvider(),
    ZaiCodeProvider(),
    ZaiApiProvider(),
]
DEFAULT_PROVIDER = PROVIDERS[0]


def provider_for(base_url: str):
    """The provider matching a base URL, or None for a custom provider."""
    return next((p for p in PROVIDERS if p.base_url == base_url), None)


def provider_named(name: str):
    """The provider with a preset name, or None."""
    return next((p for p in PROVIDERS if p.name.lower() == name.lower()), None)
