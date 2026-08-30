"""Tests for the efficient_ingestion plugin (replaceable ingestion engine).

The engine is exposed through the ServiceFactory (``re-ingestion`` category):
configs are stored in the global ``system:agent`` settings list, the plugin
registers ``EfficientIngestionConfiguration`` via ``factory_allowed_ingestions``
and, when present, it is the preferred engine.

Uses ``tests/conftest.py`` fixtures: Redis db=1 (isolated).
"""

from cat.core_plugins.efficient_ingestion.configs import EfficientIngestionConfiguration
from cat.core_plugins.efficient_ingestion.reembed import EfficientIngestionEngine
from cat.db.cruds import settings as crud_settings
from cat.db.database import DEFAULT_SYSTEM_KEY, get_sync_db
from cat.db.models import Setting
from cat.services.factory.ingestion import (
    resolve_ingestion_engine,
    resolved_config_name,
    store_selection,
)
from tests.utils import get_client_admin_headers


def _cleanup():
    db = get_sync_db()
    db.json().delete("system:agent", '$[?(@.category == "re-ingestion")]')


async def test_configuration_defaults():
    cfg = EfficientIngestionConfiguration()
    assert cfg.model_dump() == {"reembed_max_concurrency": 5}
    assert cfg.pyclass() is EfficientIngestionEngine


async def test_resolved_default_prefers_plugin_engine(lizard):
    _cleanup()
    # efficient_ingestion is a core plugin: its config class is always allowed
    assert await resolved_config_name(lizard) == "EfficientIngestionConfiguration"

    engine = await resolve_ingestion_engine(lizard)
    assert engine.__class__.__name__ == "EfficientIngestionEngine"
    assert engine.reembed_max_concurrency == 5


async def test_explicit_selection_overrides_default(lizard):
    _cleanup()
    await store_selection("BaseIngestionConfiguration")
    assert await resolved_config_name(lizard) == "BaseIngestionConfiguration"

    engine = await resolve_ingestion_engine(lizard)
    assert engine.__class__.__name__ == "CoreIngestionEngine"

    # an invalid stored name falls back to the plugin engine
    await crud_settings.upsert_setting_by_name(
        DEFAULT_SYSTEM_KEY,
        Setting(name="ingestion_engine", value={"engine": "NotAConfig"}, category="re-ingestion"),
    )
    assert await resolved_config_name(lizard) == "EfficientIngestionConfiguration"


async def test_upsert_stores_category_re_ingestion_and_value(lizard):
    _cleanup()
    from cat.services.factory.ingestion import build_factory

    sf = build_factory(lizard)
    result = await sf.upsert_service("EfficientIngestionConfiguration", {"reembed_max_concurrency": 3})

    entry = await crud_settings.get_setting_by_name(DEFAULT_SYSTEM_KEY, "EfficientIngestionConfiguration")
    assert entry is not None
    assert entry["category"] == "re-ingestion"
    assert entry["value"] == {"reembed_max_concurrency": 3}

    engine = await resolve_ingestion_engine(lizard)
    assert engine.__class__.__name__ == "EfficientIngestionEngine"
    assert engine.reembed_max_concurrency == 3
    _cleanup()


async def test_available_configs_include_base_and_efficient(lizard):
    _cleanup()
    from cat.services.factory.ingestion import build_factory

    sf = build_factory(lizard)
    schemas = await sf.get_schemas()
    assert "BaseIngestionConfiguration" in schemas
    assert "EfficientIngestionConfiguration" in schemas


async def test_endpoints_list_and_select(secure_client, secure_client_headers, cheshire_cat, lizard):
    _cleanup()
    admin_headers = await get_client_admin_headers(secure_client)

    listing = await secure_client.get("/ingestion/settings", headers=secure_client_headers)
    assert listing.status_code == 200
    body = listing.json()
    names = [s["name"] for s in body["settings"]]
    assert "BaseIngestionConfiguration" in names
    assert "EfficientIngestionConfiguration" in names
    assert body["selected_configuration"] == "EfficientIngestionConfiguration"

    # select the base engine explicitly
    res = await secure_client.put(
        "/ingestion/settings/BaseIngestionConfiguration",
        headers=admin_headers,
        json={},
    )
    assert res.status_code == 200

    listing2 = await secure_client.get("/ingestion/settings", headers=secure_client_headers)
    assert listing2.json()["selected_configuration"] == "BaseIngestionConfiguration"
    _cleanup()
