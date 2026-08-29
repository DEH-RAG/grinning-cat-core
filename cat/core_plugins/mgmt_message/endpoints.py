from typing import Any

from cat import endpoint, log
from cat.core_plugins.mgmt_message.settings import _MGMT_SETTING_NAME
from cat.db.cruds import settings as crud_settings
from cat.db.database import DEFAULT_SYSTEM_KEY


@endpoint.get("/global_message", prefix="/mgmt_message", tags=["Management Message"])
async def get_global_message() -> dict[str, Any]:
    """Public, unauthenticated read of the plugin's global settings.

    Returns the 4-field settings dict stored inside the global ``system:agent``
    settings list under the ``mgmt_message`` entry — the same storage and the
    same ``crud_settings`` interface used for the system-level embedder
    configuration. No authentication is required so that external consumers
    (e.g. the RITA widget) can show the global banner without holding
    SYSTEM/admin credentials.

    Example: ``{"management_message": "", "management_active": false,
    "global_message": "...", "show_global_msg": true}``

    Note: the authenticated ``GET /plugins/system/settings/mgmt_message``
    cannot be used here because it requires SYSTEM READ permission, which a
    public consumer (RITA dashboards for teachers/students) does not hold.
    Writing/updating the settings goes through the standard
    ``PUT /plugins/system/settings/mgmt_message`` route (embedder pattern).
    """
    try:
        setting = await crud_settings.get_setting_by_name(DEFAULT_SYSTEM_KEY, _MGMT_SETTING_NAME)
        value = (setting or {}).get("value")
        return value if isinstance(value, dict) else {}
    except Exception as e:  # noqa: BLE001 - endpoint must never 500 on a banner read
        log.error(f"mgmt_message global_message read failed: {e}")
        return {}