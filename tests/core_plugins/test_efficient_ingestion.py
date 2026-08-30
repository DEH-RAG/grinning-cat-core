"""Tests for the efficient_ingestion plugin settings (system-type storage).

The plugin stores its settings in the global ``system:agent`` settings list
under the ``re-ingestion`` category, exactly like the system-level embedder
configuration.

Uses ``tests/conftest.py`` fixtures: Redis db=1 (isolated).
"""

from cat.core_plugins.efficient_ingestion.settings import (
    PluginSettings,
    get_settings,
    load_settings,
    save_settings,
)
from cat.db.cruds import settings as crud_settings
from cat.db.database import DEFAULT_SYSTEM_KEY, get_sync_db


async def test_settings_defaults():
    assert PluginSettings().model_dump() == {"reembed_max_concurrency": 5}

    value = await load_settings.function("efficient_ingestion", DEFAULT_SYSTEM_KEY)
    assert value == {"reembed_max_concurrency": 5}


async def test_save_roundtrip_in_system_agent_re_ingestion_category():
    saved = await save_settings.function(
        "efficient_ingestion", {"reembed_max_concurrency": 3}, DEFAULT_SYSTEM_KEY
    )
    assert saved == {"reembed_max_concurrency": 3}

    entry = await crud_settings.get_setting_by_name(DEFAULT_SYSTEM_KEY, "efficient_ingestion")
    assert entry is not None
    assert entry["category"] == "re-ingestion"
    assert entry["value"] == {"reembed_max_concurrency": 3}

    # cleanup: remove the entry from the system:agent list
    db = get_sync_db()
    db.json().delete("system:agent", '$[?(@.name=="efficient_ingestion")]')


async def test_get_settings_falls_back_to_defaults():
    # ensure nothing stored
    db = get_sync_db()
    db.json().delete("system:agent", '$[?(@.name=="efficient_ingestion")]')

    value = await get_settings()
    assert value == {"reembed_max_concurrency": 5}
