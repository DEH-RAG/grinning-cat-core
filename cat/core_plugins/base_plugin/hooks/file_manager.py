"""Hook to handle operations after file-manager deletions."""

from cat import hook


@hook(priority=0)
def after_file_manager_file_deleted(filename: str, scope: str, cat) -> None:
    """
    Hook triggered after a stored file (and its memory points) is deleted.

    Fired by the DELETE /file_manager/... routes after the file and its points
    are removed; the ``ingestion_status`` plugin drops its per-source status
    row here so no stale row lingers. No-op default.
    """
