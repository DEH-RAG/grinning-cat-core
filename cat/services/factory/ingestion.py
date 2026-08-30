"""Replaceable ingestion engine (factory pattern, fork-policy level 3).

The PUT /embedder/settings route must re-embed every agent's stored sources
with the new embedder. Which engine performs the pass is decided through the
same ServiceFactory mechanism used for embedders: the core provides the base
implementation and plugins can register more efficient ones via the
``factory_allowed_ingestions`` hook (``efficient_ingestion`` registers
``EfficientIngestionConfiguration``).

The engine selection entry lives in the global ``system:agent`` settings list
under the ``re-ingestion`` category (name ``ingestion_engine``, value = config
class name). Without an explicit entry the first non-core configuration
contributed by a plugin wins; otherwise the base one is used.
"""


from cat.db.cruds import settings as crud_settings
from cat.db.database import DEFAULT_SYSTEM_KEY
from cat.db.models import Setting
from cat.log import log
from cat.services.factory.models import BaseFactoryConfigModel

_SELECTION_NAME = "ingestion_engine"


class BaseIngestionEngine:
    """Interface for the re-embed pass run on embedder change."""

    async def run(self, lizard) -> bool:
        raise NotImplementedError


class CoreIngestionEngine(BaseIngestionEngine):
    """Base implementation: the upstream-parity re-embed flow (core methods)."""

    async def run(self, lizard) -> bool:
        try:
            await lizard.embed_all_in_cheshire_cats()
            return True
        except Exception as e:  # noqa: BLE001 - parity with the core error handling
            log.error(f"Error embedding all stored files: {e}")
            return False


class BaseIngestionConfiguration(BaseFactoryConfigModel):
    """Base configuration: the upstream-parity re-embed flow (core)."""

    @classmethod
    def pyclass(cls) -> type:
        return CoreIngestionEngine

    @classmethod
    def base_class(cls) -> type:
        return BaseIngestionEngine


def build_factory(lizard) -> "ServiceFactory":
    """ServiceFactory for the ``re-ingestion`` category (system scope)."""
    # local import: avoids the circular service_factory <-> factory.ingestion
    from cat.services.service_factory import ServiceFactory

    return ServiceFactory(
        agent_key=lizard.agent_key,
        hook_manager=lizard.plugin_manager,
        factory_allowed_handler_name="factory_allowed_ingestions",
        setting_category="re-ingestion",
        schema_name="ingestionConfigurationName",
    )


async def _allowed_classes(lizard) -> list[type[BaseFactoryConfigModel]]:
    return await build_factory(lizard)._get_allowed_classes()


async def resolved_config_name(lizard) -> str:
    """Effective engine selection: explicit entry, else plugin class, else base."""
    classes = await _allowed_classes(lizard)

    try:
        explicit = await crud_settings.get_setting_by_name(DEFAULT_SYSTEM_KEY, _SELECTION_NAME)
        explicit_value = (explicit or {}).get("value")
        explicit_name = explicit_value.get("engine") if isinstance(explicit_value, dict) else None
        if isinstance(explicit_name, str) and any(c.__name__ == explicit_name for c in classes):
            return explicit_name
    except Exception as e:  # noqa: BLE001 - resolution must never break the route
        log.error(f"ingestion engine selection read failed: {e}")

    non_core = [c for c in classes if c is not BaseIngestionConfiguration]
    if non_core:
        return non_core[0].__name__
    return BaseIngestionConfiguration.__name__


async def resolve_ingestion_engine(lizard) -> BaseIngestionEngine | None:
    """Resolve the configured engine instance; None when nothing can be built.

    Unlike ``ServiceFactory.get_from_config_name`` (which falls back to the
    base with a loud error log when the entry was never saved), a missing
    entry here is normal: the configuration model defaults are used.
    """
    name = await resolved_config_name(lizard)
    sf = build_factory(lizard)
    try:
        config_class = await sf._get_factory_class(name)
        if config_class is None:
            return None
        entry = await crud_settings.get_setting_by_name(DEFAULT_SYSTEM_KEY, name)
        value = (entry or {}).get("value") or {}
        engine = config_class.get_from_config(value)
        await sf._set_agent_id(engine)
        return engine
    except Exception as e:  # noqa: BLE001
        log.error(f"Failed to instantiate ingestion engine '{name}': {e!r}")
        return None


async def store_selection(config_name: str) -> None:
    """Save the explicit engine selection (``system:agent``, ``re-ingestion``)."""
    await crud_settings.upsert_setting_by_name(
        DEFAULT_SYSTEM_KEY,
        Setting(name=_SELECTION_NAME, value={"engine": config_name}, category="re-ingestion"),
    )
