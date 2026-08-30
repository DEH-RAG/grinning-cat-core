"""Hook to handle operations after embedder settings are updated."""

from cat import CheshireCat, hook


@hook(priority=0)
def after_embedder_settings_update(embedder_name: str, embedder_size: int, lizard: CheshireCat) -> None:
    """
    Hook triggered after the embedder settings have been updated.

    This function is executed after the configured embedder changed (the PUT
    /embedder/settings route then notifies the plugins instead of re-embedding
    directly). The ``efficient_ingestion`` core plugin implements this hook to run
    the re-embed of stored sources and procedures; without it the hook is a
    no-op and nothing is re-embedded.

    Args:
        embedder_name: str
            The newly configured embedder name.
        embedder_size: int
            The vector size of the newly configured embedder.
        lizard: CheshireCat
            The system instance (BillTheLizard) passed by the hook caller under
            the ``lizard`` keyword (system-context hook execution). Plugins use
            it to reach every agent.
    """
