from cat import hook
from cat.auth.permissions import AuthResource
from cat.db.cruds import settings as crud_settings
from cat.db.database import DEFAULT_SYSTEM_KEY
from cat import log

@hook(priority=1)
async def auth_request(local_user, agent_id, connection, **kwargs):
    # read global settings
    setting = await crud_settings.get_setting_by_name(DEFAULT_SYSTEM_KEY, "mgmt_message")
    value = (setting or {}).get("value") or {}
    if not value.get("management_active", False):
        return None  # allow

    # allowed iff the principal has SYSTEM permission (admin system user or valid API-KEY)
    permissions = getattr(local_user, "permissions", None) or {}
    if str(AuthResource.SYSTEM) in permissions:
        return None  # allow

    log.info(f"MANAGEMENT MODE: Access negated to user {local_user}.")
    return value.get("management_message", "Access denied")
