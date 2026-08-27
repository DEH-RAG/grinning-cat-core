from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel

from cat import plugin
from cat.db.database import DEFAULT_AGENT_KEY, DEFAULT_SYSTEM_KEY, get_sync_db


class PluginSettings(BaseModel):
    management_message: str = ""
    management_active: bool = False
    global_message: str = ""
    show_global_msg: bool = False


# the global Redis key where this plugin's settings live
_MGMT_SETTING_NAME = "mgmt-message"


@plugin
def settings_schema() -> dict:
    return PluginSettings.model_json_schema()


@plugin
def settings_model():
    return PluginSettings


@plugin
def load_settings(plugin_id: str, agent_id: str) -> dict:
    db = get_sync_db()
    key = f"{DEFAULT_SYSTEM_KEY}:{DEFAULT_AGENT_KEY}"
    # read the setting by name from the JSON list
    found = db.json().get(key, f'$[?(@.name=="{_MGMT_SETTING_NAME}")]')
    if not found:
        return {}
    entry = found[0]
    if not isinstance(entry, dict):
        return {}
    value = entry.get("value")
    return value if isinstance(value, dict) else {}


@plugin
def save_settings(plugin_id: str, settings: dict, agent_id: str) -> dict:
    # validate against the model
    validated = PluginSettings(**settings).model_dump()
    db = get_sync_db()
    key = f"{DEFAULT_SYSTEM_KEY}:{DEFAULT_AGENT_KEY}"
    # read the full list
    full = db.json().get(key, "$")
    settings_list: list = []
    if isinstance(full, list) and full and isinstance(full[0], list):
        settings_list = full[0]
    # find existing entry by name
    new_entry = {
        "name": _MGMT_SETTING_NAME,
        "value": validated,
        "category": None,
        "setting_id": str(uuid4()),
        "updated_at": datetime.now(UTC).timestamp(),
    }
    replaced = False
    for i, entry in enumerate(settings_list):
        if entry.get("name") == _MGMT_SETTING_NAME:
            settings_list[i] = new_entry
            replaced = True
            break
    if not replaced:
        settings_list.append(new_entry)
    db.json().set(key, "$", settings_list)
    return validated