from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel

from cat import plugin
from cat.db.database import DEFAULT_AGENT_KEY, DEFAULT_SYSTEM_KEY, get_sync_db

# entry name under which this plugin persists its settings in the global
# ``system:agent`` settings list, exactly like the system-level embedder
# configuration (same ``{name, value, category, setting_id, updated_at}`` shape)
_MGMT_SETTING_NAME = "mgmt_message"
_MGMT_SETTING_CATEGORY = "mgmt_message"

# legacy key written by the previous default-plugin-settings mechanism
_LEGACY_KEY = f"{DEFAULT_SYSTEM_KEY}:plugins:{_MGMT_SETTING_NAME}"


class PluginSettings(BaseModel):
    management_message: str = ""
    management_active: bool = False
    global_message: str = ""
    show_global_msg: bool = False


@plugin
def settings_schema() -> dict:
    return PluginSettings.model_json_schema()


@plugin
def settings_model():
    return PluginSettings


def _system_agent_key() -> str:
    return f"{DEFAULT_SYSTEM_KEY}:{DEFAULT_AGENT_KEY}"


def _write_entry(db, validated: dict) -> None:
    """Upsert the plugin entry in the ``system:agent`` settings list."""
    key = _system_agent_key()

    # read the full list
    full = db.json().get(key, "$")
    settings_list: list = []
    if isinstance(full, list) and full and isinstance(full[0], list):
        settings_list = full[0]

    new_entry = {
        "name": _MGMT_SETTING_NAME,
        "value": validated,
        "category": _MGMT_SETTING_CATEGORY,
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


def _migrate_legacy(db) -> dict:
    """One-off migration: move the old ``system:plugins:mgmt_message`` dict into ``system:agent``."""
    legacy = db.json().get(_LEGACY_KEY)
    if not isinstance(legacy, dict) or not legacy:
        return {}

    try:
        validated = PluginSettings(**legacy).model_dump()
        _write_entry(db, validated)
        db.delete(_LEGACY_KEY)
        return validated
    except Exception:  # noqa: BLE001 - migration must never block the settings load
        # if the legacy value does not validate, leave it in place
        return {}


@plugin
def load_settings(plugin_id: str, agent_id: str) -> dict:
    # the plugin settings live inside the global ``system:agent`` list under a
    # single entry (like the system embedder configuration), NOT in
    # ``system:plugins:*``
    db = get_sync_db()
    found = db.json().get(_system_agent_key(), f'$[?(@.name=="{_MGMT_SETTING_NAME}")]')
    if not found:
        return _migrate_legacy(db)

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
    _write_entry(db, validated)
    # no longer used: drop the legacy key if it still exists
    if db.exists(_LEGACY_KEY):
        db.delete(_LEGACY_KEY)
    return validated


@plugin
def activated(plugin):
    # Plugin.activate_settings() writes a system:plugins:<id> key on every
    # activation; this plugin persists in system:agent (like the embedder
    # config), so drop that leftover key to keep a single source of truth.
    db = get_sync_db()
    if db.exists(_LEGACY_KEY):
        db.delete(_LEGACY_KEY)