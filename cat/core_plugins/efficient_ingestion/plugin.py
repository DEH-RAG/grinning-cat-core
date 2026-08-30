"""Efficient ingestion — re-embed engine trigger.

Implements the ``after_embedder_settings_update`` hook: when the configured
embedder changes, every agent's stored sources and procedures are re-embedded
through ``efficient_ingestion.reembed.reembed_all``.

The plugin is system-level: its settings live in the global ``system:agent``
store under the ``re-ingestion`` category (see ``settings.py``).
"""

from cat import hook


@hook(priority=0)
async def after_embedder_settings_update(embedder_name: str, embedder_size: int, lizard) -> None:
    """Re-embed every agent's stored sources and procedures on embedder change.

    The core only fires the hook: when this plugin is loaded the re-embed runs
    on every embedder change, and nothing happens otherwise.
    """
    # lazy import: keeps module load order simple and side-effect free
    from cat.core_plugins.efficient_ingestion.reembed import reembed_all

    if lizard is None:
        return
    await reembed_all(lizard, embedder_name, embedder_size)
