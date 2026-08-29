from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel

from cat import log, plugin
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


def _upsert_entry(db, validated: dict) -> None:
    """Upsert the plugin entry in the ``system:agent`` settings list.

    Mirrors ``crud_settings.upsert_setting_by_category`` (same storage, same
    ``category`` semantics) so the plugin's system:agent entry is byte-identical
    in shape to the one the embedder configuration produces.
    """
    key = _system_agent_key()

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


@plugin
def load_settings(plugin_id: str, agent_id: str) -> dict:
    """Read the plugin settings from the global ``system:agent`` settings list.

    NOTE: the Plugin base invokes the ``load_settings`` override synchronously,
    so this must stay a plain function (no async / no crud_settings).
    """
    db = get_sync_db()
    found = db.json().get(_system_agent_key(), f'$[?(@.name=="{_MGMT_SETTING_NAME}")]')
    if found and isinstance(found[0], dict):
        value = found[0].get("value")
        if isinstance(value, dict):
            return value

    # not stored: one-off migration of the legacy key, otherwise model defaults
    migrated = _migrate_legacy(db)
    if migrated is not None:
        return migrated
    return PluginSettings().model_dump()


@plugin
def save_settings(plugin_id: str, settings: dict, agent_id: str) -> dict:
    """Upsert the plugin settings into the global ``system:agent`` list.

    Sync override (the Plugin base calls it without await).
    """
    validated = PluginSettings(**settings).model_dump()
    db = get_sync_db()
    _upsert_entry(db, validated)
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


def _migrate_legacy(db) -> dict | None:
    """One-off migration: move the old ``system:plugins:mgmt_message`` dict into ``system:agent``."""
    legacy = db.json().get(_LEGACY_KEY)
    if not isinstance(legacy, dict) or not legacy:
        return None

    try:
        validated = PluginSettings(**legacy).model_dump()
        _upsert_entry(db, validated)
        db.delete(_LEGACY_KEY)
        return validated
    except Exception as e:  # noqa: BLE001 - migration must never block the settings load
        log.error(f"mgmt_message legacy migration failed: {e}")
        return None