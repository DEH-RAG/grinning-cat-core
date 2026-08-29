from typing import Any

from cat import endpoint, log
from cat.core_plugins.mgmt_message.settings import _MGMT_SETTING_NAME
from cat.db.cruds import settings as crud_settings
from cat.db.database import DEFAULT_SYSTEM_KEY


@endpoint.get("/global_message", prefix="/mgmt_message", tags=["Management Message"])
async def get_global_message() -> dict[str, Any]:
    """Public, unauthenticated read of the plugin's global settings.

    Returns the 4-field settings dict stored inside the global ``system:agent``
    settings list under the ``mgmt_message`` entry (like the system embedder
    configuration). No authentication is required so that external consumers
    (e.g. the RITA widget) can show the global banner without holding
    SYSTEM/admin credentials.

    Example: ``{"management_message": "", "management_active": false,
    "global_message": "...", "show_global_msg": true}``
    """
    try:
        setting = await crud_settings.get_setting_by_name(DEFAULT_SYSTEM_KEY, _MGMT_SETTING_NAME)
        value = (setting or {}).get("value")
        return value if isinstance(value, dict) else {}
    except Exception as e:  # noqa: BLE001 - endpoint must never 500 on a banner read
        log.error(f"mgmt_message global_message read failed: {e}")
        return {}
