"""Settings for the efficient_ingestion plugin.

System-level plugin: the settings are stored in the global ``system:agent``
settings list under the ``re-ingestion`` category (single entry), so every
agent shares the same configuration. Mirrors the ``mgmt_message`` storage
pattern using the official ``cat.db.crud`` API.
"""

from pydantic import BaseModel

from cat import log, plugin
from cat.db.cruds import settings as crud_settings
from cat.db.database import DEFAULT_SYSTEM_KEY
from cat.db.models import Setting

_SETTING_NAME = "efficient_ingestion"
_SETTING_CATEGORY = "re-ingestion"
_DEFAULT_MAX_CONCURRENCY = 5


class PluginSettings(BaseModel):
    reembed_max_concurrency: int = _DEFAULT_MAX_CONCURRENCY


@plugin
def settings_schema() -> dict:
    return PluginSettings.model_json_schema()


@plugin
def settings_model():
    return PluginSettings


async def get_settings() -> dict:
    """Read the plugin settings (defaults on any failure, never raises)."""
    try:
        setting = await crud_settings.get_setting_by_name(DEFAULT_SYSTEM_KEY, _SETTING_NAME)
        value = (setting or {}).get("value")
        if isinstance(value, dict):
            return value
    except Exception as e:  # noqa: BLE001 - settings must never break the engine
        log.error(f"efficient_ingestion settings read failed: {e}")
    return PluginSettings().model_dump()


@plugin
async def load_settings(plugin_id: str, agent_id: str) -> dict:
    """System-type read: the settings come from the global ``system:agent`` list."""
    return await get_settings()


@plugin
async def save_settings(plugin_id: str, settings: dict, agent_id: str) -> dict:
    """System-type upsert into ``system:agent``, category ``re-ingestion``."""
    validated = PluginSettings(**settings).model_dump()
    await crud_settings.upsert_setting_by_name(
        DEFAULT_SYSTEM_KEY,
        Setting(name=_SETTING_NAME, value=validated, category=_SETTING_CATEGORY),
    )
    return validated
