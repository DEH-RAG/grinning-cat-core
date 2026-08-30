from typing import Dict
from fastapi import APIRouter, Body, BackgroundTasks

from cat.auth.connection import AuthorizedInfo
from cat.auth.permissions import AuthResource, AuthPermission, check_permissions
from cat.routes.routes_utils import GetSettingsResponse, GetSettingResponse, UpsertSettingResponse, run_background_task
from cat.services.service_factory import ServiceFactory


async def _reembed_on_embedder_change(lizard, embedder_name: str, embedder_size: int) -> None:
    """Notify plugins that the embedder changed (re-embed trigger).

    The actual re-embed engine lives in the ``ingestion_status`` core plugin
    (``after_embedder_settings_update`` hook); the core only fires the hook so
    the fork stays close to upstream (no re-embed logic here).
    """
    await lizard.plugin_manager.execute_hook(
        "after_embedder_settings_update", embedder_name, embedder_size, caller=lizard
    )


router = APIRouter(tags=["Embedder"], prefix="/embedder")


# get configured Embedders and configuration schemas
@router.get("/settings", response_model=GetSettingsResponse)
async def get_embedders_settings(
    info: AuthorizedInfo = check_permissions(AuthResource.EMBEDDER, AuthPermission.READ),
) -> GetSettingsResponse:
    """Get the list of the Embedders"""
    lizard = info.lizard
    sf = ServiceFactory(
        agent_key=lizard.agent_key,  # type: ignore[arg-type]
        hook_manager=lizard.plugin_manager,
        factory_allowed_handler_name="factory_allowed_embedders",
        setting_category="embedder",
        schema_name="languageEmbedderName",
    )
    return await sf.get_factory_settings()


@router.get("/settings/{embedder_name}", response_model=GetSettingResponse)
async def get_embedder_settings(
    embedder_name: str,
    info: AuthorizedInfo = check_permissions(AuthResource.EMBEDDER, AuthPermission.READ),
) -> GetSettingResponse:
    """Get settings and scheme of the specified Embedder"""
    lizard = info.lizard
    sf = ServiceFactory(
        agent_key=lizard.agent_key,  # type: ignore[arg-type]
        hook_manager=lizard.plugin_manager,
        factory_allowed_handler_name="factory_allowed_embedders",
        setting_category="embedder",
        schema_name="languageEmbedderName",
    )
    return await sf.get_factory_setting(embedder_name)


@router.put("/settings/{embedder_name}", response_model=UpsertSettingResponse)
async def upsert_embedder_setting(
    background_tasks: BackgroundTasks,
    embedder_name: str,
    payload: Dict = Body(default={}),
    info: AuthorizedInfo = check_permissions(AuthResource.EMBEDDER, AuthPermission.WRITE),
) -> UpsertSettingResponse:
    """Upsert the Embedder setting"""
    lizard = info.lizard
    previous_embedder = await lizard.embedder()
    sf = ServiceFactory(
        agent_key=lizard.agent_key,  # type: ignore[arg-type]
        hook_manager=lizard.plugin_manager,
        factory_allowed_handler_name="factory_allowed_embedders",
        setting_category="embedder",
        schema_name="languageEmbedderName",
    )

    result = await sf.upsert_service(embedder_name, payload)

    current_embedder = await lizard.embedder()

    # a characterizing feature of the embedder has been updated: inform the plugins,
    # which own the re-embed engine (ingestion_status core plugin hook)
    if previous_embedder != current_embedder:
        run_background_task(
            background_tasks,
            _reembed_on_embedder_change,
            info.lizard,
            current_embedder.name,
            current_embedder.size,
        )

    return UpsertSettingResponse(**result)
