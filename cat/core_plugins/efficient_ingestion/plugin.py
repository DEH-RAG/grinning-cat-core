"""Efficient ingestion — replaceable ingestion engine.

Registers ``EfficientIngestionConfiguration`` through the
``factory_allowed_ingestions`` hook: the core resolves the engine through the
ServiceFactory (``re-ingestion`` category) when the embedder changes and, when
this plugin is present, prefers our efficient implementation.

The plugin is system-level: factory entries and the engine selection live in
the global ``system:agent`` store under the ``re-ingestion`` category (see
``configs.py`` and the plugin's settings endpoints).
"""

from cat import hook
from cat.core_plugins.efficient_ingestion.configs import EfficientIngestionConfiguration


@hook(priority=0)
def factory_allowed_ingestions(allowed, lizard):
    """Register the efficient ingestion engine as a factory option."""
    return list(allowed) + [EfficientIngestionConfiguration]
