from typing import Any

from cat import endpoint, log
from cat.db.cruds import plugins as crud_plugins
from cat.db.database import DEFAULT_SYSTEM_KEY, get_sync_db

# plugin id (folder name)
_PLUGIN_ID = "mgmt_message"


@endpoint.get("/global_message", prefix="/mgmt_message", tags=["Management Message"])
async def get_global_message() -> dict[str, Any]:
    """Public, unauthenticated read of the plugin's global settings.

    Returns the raw 4-field settings dict stored under the global
    ``system:plugins:mgmt_message`` key (the default plugin-settings
    mechanism for the system agent). No authentication is required so that
    external consumers (e.g. the RITA widget) can show the global banner
    without holding SYSTEM/admin credentials.

    Example: ``{"management_message": "", "management_active": false,
    "global_message": "...", "show_global_msg": true}``
    """
    try:
        # prefer the async CRUD read; fall back to a direct sync read on failure
        setting = await crud_plugins.get_setting(DEFAULT_SYSTEM_KEY, _PLUGIN_ID)
        if setting is not None:
            return setting

        # direct sync read (e.g. Redis client not fully usable via CRUD)
        db = get_sync_db()
        val = db.json().get(f"{DEFAULT_SYSTEM_KEY}:plugins:{_PLUGIN_ID}")
        return val if isinstance(val, dict) else {}
    except Exception as e:  # noqa: BLE001 - endpoint must never 500 on a banner read
        log.error(f"mgmt_message global_message read failed: {e}")
        return {}
