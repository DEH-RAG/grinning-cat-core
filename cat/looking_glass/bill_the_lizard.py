import asyncio
import json
from typing import List, Dict
from fastapi import FastAPI

from cat.auth.auth_utils import DEFAULT_ADMIN_USERNAME, hash_password
from cat.auth.permissions import get_full_permissions
from cat.db import crud
from cat.db.cruds import settings as crud_settings, plugins as crud_plugins, users as crud_users
from cat.db.database import DEFAULT_SYSTEM_KEY, DEFAULT_CONVERSATIONS_KEY, get_async_db
from cat.db.models import Setting
from cat.env import get_env
from cat.log import log
from cat.looking_glass.cheshire_cat import CheshireCat
from cat.looking_glass.mad_hatter.mad_hatter import MadHatter
from cat.looking_glass.mad_hatter.registry import PluginRegistry
from cat.mixins import OrchestratorMixin, NonCopyableMixin
from cat.rabbit_hole import RabbitHole
from cat.services.factory.auth_handler import CoreAuthHandler
from cat.services.websocket_manager import WebSocketManager
from cat.utils import singleton, safe_deepcopy, sanitize_permissions


@singleton
class BillTheLizard(OrchestratorMixin, NonCopyableMixin):
    """
    Singleton class that manages the Cheshire Cats and their strays.

    The Cheshire Cats are the agents that are currently active and have users to attend.
    The strays are the users that are waiting for an agent to attend them.

    The Bill The Lizard Manager is responsible for:
    - Creating and deleting Cheshire Cats
    - Adding and removing strays from Cheshire Cats
    - Getting the Cheshire Cat of a stray
    - Getting the strays of a Cheshire Cat
    """
    def __init__(self):
        """
        Bill the Lizard initialization.

        The constructor is intentionally kept **synchronous and side-effect-free**: it only initialises the plugin manager
        so that the singleton exists immediately.
        All async bootstrap work (plugin discovery, hook execution, service setup) is deferred to :meth:`bootstrap`,
        which is called by `startup_app` inside uvicorn's event loop.
        """
        self._plugin_registry = None
        self._fastapi_app = None
        self._pending_endpoints = []

        # Minimal sync setup — plugin manager is created but NOT yet discovered.
        self.plugin_manager = MadHatter(self.agent_key)  # type: ignore[arg-type]

        # These are populated in bootstrap(); set to None so callers can detect
        # that bootstrap hasn't run yet.
        self.websocket_manager = None
        self.rabbit_hole = None
        self.core_auth_handler = None

    async def bootstrap(self):
        """
        Fully initialise the lizard inside uvicorn's running event loop.

        Must be awaited from an async context (e.g. the lifespan coroutine) so that:
        - `discover_plugins` can await Redis calls.
        - Hook functions that call `asyncio.ensure_future` schedule tasks on the
          **correct** (uvicorn) event loop instead of a transient side-thread loop.
        """
        # Discover and load all plugins (async: reads active_plugins from Redis)
        await self.plugin_manager.discover_plugins()

        # Store endpoints for later activation (after fastapi_app is set)
        self._pending_endpoints = safe_deepcopy(self.plugin_manager.endpoints)
        if self._fastapi_app is not None:
            self._activate_pending_endpoints()

        # Allow plugins to act before the remaining cat components are created
        await self.plugin_manager.execute_hook("before_lizard_bootstrap", caller=self)

        self.websocket_manager = WebSocketManager()
        self.rabbit_hole = RabbitHole()
        self.core_auth_handler = CoreAuthHandler()

        await self.service_provider.bootstrap_services(self.agent_key, self.plugin_manager)

        # Initialize the default admin if not present
        if not await crud_users.get_users(DEFAULT_SYSTEM_KEY, limit=1):
            permissions = sanitize_permissions(get_full_permissions(), DEFAULT_SYSTEM_KEY)

            await crud_users.create_user(DEFAULT_SYSTEM_KEY, {
                "username": DEFAULT_ADMIN_USERNAME,
                "password": hash_password(get_env("CAT_ADMIN_DEFAULT_PASSWORD")),
                "permissions": permissions,  # base admin has all permissions, but CHAT
            })

        # Start Redis Pub/Sub listener so WebSocket messages are delivered
        # cross-replica in Docker Swarm deployments.  Degrades gracefully to
        # local-only mode if Redis pub/sub is unavailable.
        await self.websocket_manager.start()

        await self.plugin_manager.execute_hook("after_lizard_bootstrap", caller=self)

    async def create_cheshire_cat(self, agent_id: str, metadata: Dict | None = None) -> CheshireCat:
        """
        Create the Cheshire Cat with the given id, directly from db.

        Args:
            agent_id: The id of the agent to get
            metadata: The metadata of the agent to create

        Returns:
            The Cheshire Cat with the given id, or None if it doesn't exist
        """
        if agent_id == DEFAULT_SYSTEM_KEY:
            raise ValueError(f"{agent_id} is not allowed as name for agents")

        if agent_id in await crud_settings.get_agents_main_keys():
            return await self.get_cheshire_cat(agent_id)  # type: ignore[return-value]

        ccat = None
        try:
            ccat = await CheshireCat.create(agent_id)
            if metadata is not None:
                await crud_settings.upsert_setting_by_name(
                    ccat.agent_key,
                    Setting(name="metadata", value=metadata),
                )

            embedder = await self.embedder()
            await ccat.vector_memory_handler.initialize(embedder.name, embedder.size)
            await ccat.embed_procedures()

            await self.plugin_manager.execute_hook("after_cheshire_cat_creation", ccat, caller=self)

            return ccat
        except Exception as e:
            log.error(f"Error creating Cheshire Cat `{agent_id}`: {e}")
            await self.rollback_cheshire_cat_creation(agent_id, ccat)

            raise

    async def rollback_cheshire_cat_creation(self, agent_id: str, cat: CheshireCat | None) -> None:
        """
        Rollback the creation of a Cheshire Cat with the given id.

        Args:
            agent_id: The id of the agent to rollback
            cat: The Cheshire Cat to rollback

        Returns:
            None
        """
        # rollback
        if cat:
            await cat.destroy()
            return

        await crud.delete(agent_id)

    async def _get_cheshire_cat_on_plugin_event(self, agent_id: str, plugin_id: str) -> CheshireCat | None:
        """
        Determines and retrieves the CheshireCat object associated with a specific plugin event for a given agent if the
        plugin is active.

        Args:
            agent_id (str): The unique identifier for the agent.
            plugin_id (str): The unique identifier for the plugin.

        Returns:
            CheshireCat | None: The CheshireCat object if the plugin is active, otherwise None.
        """
        active_plugins = await crud_plugins.get_active_plugins_from_db(agent_id)
        if not active_plugins or plugin_id not in active_plugins:
            return None

        return await self.get_cheshire_cat(agent_id)

    @staticmethod
    async def get_cheshire_cat(agent_id: str) -> CheshireCat | None:
        """
        Gets the Cheshire Cat with the given id, directly from db.

        Args:
            agent_id: The id of the agent to get

        Returns:
            The Cheshire Cat with the given id, or None if it doesn't exist
        """
        if agent_id == DEFAULT_SYSTEM_KEY:
            log.debug("The system agent has been requested: returning null value.")
            return None

        if agent_id not in await crud_settings.get_agents_main_keys():
            log.debug(f"Requested not existing `{agent_id}`")
            raise ValueError("Bad Request")

        agent_settings = await crud_settings.get_settings(agent_id)
        if not agent_settings:
            log.debug(f"Agent `{agent_id}` has no settings")
            return None

        return await CheshireCat.create(agent_id)

    async def clone_cheshire_cat(self, ccat: CheshireCat, new_agent_id: str) -> CheshireCat:
        """
        Clone a Cheshire Cat into a new one.

        Args:
            ccat: The Cheshire Cat to clone.
            new_agent_id: The new agent id to clone into.

        Returns:
            The cloned Cheshire Cat.
        """
        # clone the settings from the provided agent
        log.info(f"Cloning settings from agent {ccat.agent_key} to agent {new_agent_id}")
        await crud_settings.clone_agent(ccat.agent_key, new_agent_id, [DEFAULT_CONVERSATIONS_KEY])

        # delegate cloning of in-memory data/resources from the source Cheshire Cat to the new one
        cloned_ccat = await self.get_cheshire_cat(new_agent_id)
        await cloned_ccat.clone_from(ccat)  # type: ignore[arg-type]

        return cloned_ccat  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Resilient re-ingestion state helpers (Redis-backed)
    # ------------------------------------------------------------------

    @staticmethod
    def _reingestion_key(agent_id: str, suffix: str) -> str:
        return f"reingesting:{agent_id}:{suffix}"

    async def _save_reingestion_sources(self, agent_id: str, sources: Dict) -> None:
        """Persist the list of sources to Redis so it survives a crash."""
        # Convert StoredSourceWithMetadata to JSON-serializable dicts
        serializable = {}
        for collection_name, srcs in sources.items():
            serializable[str(collection_name)] = [
                {
                    "name": s.name,
                    "path": s.path,
                    "metadata": s.metadata,
                    "has_content": s.content is not None,
                }
                for s in srcs
            ]
        await get_async_db().set(
            self._reingestion_key(agent_id, "sources"),
            json.dumps(serializable),
            ex=3600,  # 1 hour TTL
        )

    async def _set_reingestion_status(self, agent_id: str, status: str, error: str = "") -> None:
        await get_async_db().set(
            self._reingestion_key(agent_id, "status"), status, ex=3600,
        )
        if error:
            await get_async_db().set(
                self._reingestion_key(agent_id, "error"), error, ex=3600,
            )

    async def _get_reingestion_status(self, agent_id: str) -> str | None:
        val = await get_async_db().get(self._reingestion_key(agent_id, "status"))
        return val.decode() if val else None

    async def _update_reingestion_progress(
        self, agent_id: str, collections_done: list, procedures_done: bool,
    ) -> None:
        await get_async_db().set(
            self._reingestion_key(agent_id, "progress"),
            json.dumps({"collections_done": collections_done, "procedures_done": procedures_done}),
            ex=3600,
        )

    async def _get_reingestion_progress(self, agent_id: str) -> dict:
        val = await get_async_db().get(self._reingestion_key(agent_id, "progress"))
        return json.loads(val) if val else {}

    async def _has_incomplete_reingestion(self, agent_id: str) -> bool:
        status = await self._get_reingestion_status(agent_id)
        return status in ("collecting", "embedding")

    async def _clear_reingestion_state(self, agent_id: str) -> None:
        base = f"reingesting:{agent_id}"
        redis = get_async_db()
        for suffix in (":sources", ":status", ":progress", ":error"):
            try:
                await redis.delete(base + suffix)
            except Exception:
                pass
        # Also clear the blocking flag
        try:
            await redis.delete(base)
        except Exception:
            pass

    async def embed_all_in_cheshire_cats(self) -> None:
        """Re-embeds all stored files and procedures in all Cheshire Cats with
        resilient state tracking in Redis (per-agent progress survives crashes).
        """
        success = False
        try:
            embedder = await self.embedder()
            embedder_name = embedder.name
            embedder_size = embedder.size

            # Global flag — blocks recall for ALL agents
            redis = get_async_db()
            await redis.set("reingesting:all", "1", ex=600)

            # --- Phase 1: Collect all stored sources from all agents ---
            ccat_ids = await crud_settings.get_agents_main_keys()
            stored_files_by_ccat: List[Dict] = []
            for ccat_id in ccat_ids:
                if (ccat := await self.get_cheshire_cat(ccat_id)) is None:
                    continue

                agent_sources = await ccat.get_stored_sources_with_metadata()
                stored_files_by_ccat.append({
                    "ccat": ccat,
                    "stored_sources": agent_sources,
                })
                # Persist sources to Redis (survives crash)
                await self._save_reingestion_sources(ccat_id, agent_sources)
                await self._set_reingestion_status(ccat_id, "initializing")

            if not stored_files_by_ccat:
                success = True
                return

            # --- Phase 2: Initialize ALL vector databases ---
            for entry in stored_files_by_ccat:
                await entry["ccat"].vector_memory_handler.initialize(
                    embedder_name, embedder_size,
                )

            # --- Phase 3: Re-embed with per-agent progress ---
            semaphore = asyncio.Semaphore(5)
            reembed_tasks = []
            for entry in stored_files_by_ccat:
                agent_id = entry["ccat"].agent_key  # type: ignore[union-attr]

                async def _reembed_with_progress(entry_, agent_id_, sem=semaphore):
                    async with sem:
                        ccat_ = entry_["ccat"]
                        sources_ = entry_["stored_sources"]
                        collections_done = []

                        await self._set_reingestion_status(agent_id_, "embedding")

                        for col_name, srcs in sources_.items():
                            if not srcs:
                                continue
                            await ccat_.embed_stored_sources(col_name, srcs)
                            collections_done.append(str(col_name))
                            await self._update_reingestion_progress(
                                agent_id_, collections_done, procedures_done=False,
                            )

                        await ccat_.embed_procedures()
                        await self._update_reingestion_progress(
                            agent_id_, collections_done, procedures_done=True,
                        )
                        await self._set_reingestion_status(agent_id_, "done")
                        await self._clear_reingestion_state(agent_id_)

                reembed_tasks.append(_reembed_with_progress(entry, agent_id))

            await asyncio.gather(*reembed_tasks)
            success = True
        except Exception as e:
            log.error(f"Error embedding all stored files: {e}")
        finally:
            try:
                await get_async_db().delete("reingesting:all")
            except Exception:
                pass

        await self.plugin_manager.execute_hook(
            "after_all_cheshire_cats_embedded", success, caller=self,
        )

    async def embed_in_cheshire_cat(self, agent_id: str, resume: bool = False) -> None:
        """Re-embed all stored files and procedures for a single Cheshire Cat.

        Unlike ``embed_all_in_cheshire_cats``, this method does **not** delete
        and recreate the global vector collections — it only clears and
        re-populates the requesting agent's tenant points.

        The re-ingestion state (sources, progress, status) is persisted to
        Redis so that a crash or restart can be recovered from.  If the agent
        has an incomplete re-ingestion, this method resumes from where it left
        off.
        """
        ccat = await self.get_cheshire_cat(agent_id)
        if ccat is None:
            raise ValueError(f"Agent '{agent_id}' not found")

        embedder = await self.embedder()
        embedder_name = embedder.name
        embedder_size = embedder.size

        # Verify embedder compatibility before touching any data
        await ccat.vector_memory_handler.check_embedding_compatibility(
            embedder_name, embedder_size,
        )

        flag_key = f"reingesting:{agent_id}"

        # --- Check for incomplete previous run ---
        if await self._has_incomplete_reingestion(agent_id):
            log.warning(
                f"Agent id: {agent_id}. Found incomplete re-ingestion — resuming"
            )
            await self._resume_reingestion(agent_id, ccat, embedder_name, embedder_size)
            return

        # --- Fresh re-ingestion ---
        await get_async_db().set(flag_key, "1", ex=600)
        await self._set_reingestion_status(agent_id, "collecting")

        try:
            stored_sources = await ccat.get_stored_sources_with_metadata()
            await self._save_reingestion_sources(agent_id, stored_sources)

            await self._run_reingestion(agent_id, ccat, stored_sources)
        except Exception as e:
            log.error(f"Agent id: {agent_id}. Re-ingestion failed: {e}")
            await self._set_reingestion_status(agent_id, "failed", str(e))
            raise
        finally:
            try:
                await get_async_db().delete(flag_key)
            except Exception:
                pass

    async def _run_reingestion(
        self, agent_id: str, ccat: CheshireCat, stored_sources: Dict,
    ) -> None:
        """Execute the actual re-ingestion with progress tracking."""
        await self._set_reingestion_status(agent_id, "embedding")

        collections_done = []
        total_collections = [k for k, v in stored_sources.items() if v]

        for collection_name, sources in stored_sources.items():
            if not sources:
                continue
            await ccat.embed_stored_sources(collection_name, sources)
            collections_done.append(str(collection_name))
            await self._update_reingestion_progress(
                agent_id, collections_done, procedures_done=False,
            )

        await ccat.embed_procedures()
        await self._update_reingestion_progress(
            agent_id, collections_done, procedures_done=True,
        )

        await self._set_reingestion_status(agent_id, "done")
        await self._clear_reingestion_state(agent_id)
        log.info(f"Agent id: {agent_id}. Re-ingestion complete")

    async def _resume_reingestion(
        self, agent_id: str, ccat: CheshireCat,
        embedder_name: str, embedder_size: int,
    ) -> None:
        """Resume an interrupted re-ingestion from the last saved state."""
        flag_key = f"reingesting:{agent_id}"
        await get_async_db().set(flag_key, "1", ex=600)

        try:
            # Re-read stored sources from Redis (survives crash)
            sources_raw = await get_async_db().get(
                self._reingestion_key(agent_id, "sources"),
            )
            if not sources_raw:
                # Sources lost — restart from scratch
                log.warning(f"Agent id: {agent_id}. Saved sources lost, restarting from scratch")
                await self._clear_reingestion_state(agent_id)
                stored_sources = await ccat.get_stored_sources_with_metadata()
                await self._save_reingestion_sources(agent_id, stored_sources)
                await self._run_reingestion(agent_id, ccat, stored_sources)
                return

            saved = json.loads(sources_raw)
            progress = await self._get_reingestion_progress(agent_id)

            # Determine which collections still need work.
            # embed_stored_sources() internally clears the tenant points first,
            # so if a collection was interrupted mid-way, it's safe to re-run.
            collections_done = set(progress.get("collections_done", []))
            all_collections = list(saved.keys())

            pending = [c for c in all_collections if c not in collections_done]
            procedures_done = progress.get("procedures_done", False)

            if pending or not procedures_done:
                log.info(
                    f"Agent id: {agent_id}. Resuming — pending collections: {pending}, "
                    f"procedures: {'done' if procedures_done else 'pending'}"
                )

            # Re-embed pending collections (the sources list was saved, but
            # the actual file contents must be re-read from the file manager).
            stored_sources = await ccat.get_stored_sources_with_metadata()
            await self._save_reingestion_sources(agent_id, stored_sources)
            await self._run_reingestion(agent_id, ccat, stored_sources)
        except Exception as e:
            log.error(f"Agent id: {agent_id}. Resume failed: {e}")
            await self._set_reingestion_status(agent_id, "failed", str(e))
            raise
        finally:
            try:
                await get_async_db().delete(flag_key)
            except Exception:
                pass

    def is_custom_endpoint(self, path: str, methods: List[str] | None = None):
        """
        Check if the given path and methods correspond to a custom endpoint.

        Args:
            path (str): The path of the endpoint to check.
            methods (List[str] | None): The HTTP methods of the endpoint to check. If None, checks all methods.

        Returns:
            bool: True if the endpoint is a custom endpoint, False otherwise.
        """
        return any(
            ep.real_path == path and (methods is None or set(ep.methods) == set(methods))
            for ep in self.plugin_manager.endpoints
        )

    async def install_plugin(self, plugin_path: str) -> str:
        try:
            plugin_id = await self.plugin_manager.install_plugin(plugin_path)

            await self.on_plugin_activate(plugin_id)
            await self.plugin_manager.execute_hook(
                "lizard_notify_plugin_installation", plugin_id, plugin_path, caller=self,
            )

            return plugin_id
        except Exception as e:
            log.error(f"Could not install plugin from {plugin_path}: {e}")
            raise e

    async def uninstall_plugin(self, plugin_id: str, dispatch_event: bool = True):
        try:
            # deactivate plugins in the Cheshire Cats
            if plugin_id in self.plugin_manager.active_plugins:
                await self.on_plugin_deactivate(plugin_id)

            await self.plugin_manager.uninstall_plugin(plugin_id)
        except Exception as e:
            log.error(f"Could not uninstall plugin {plugin_id}: {e}")
            raise e
        finally:
            if dispatch_event:
                await self.plugin_manager.execute_hook(
                    "lizard_notify_plugin_uninstallation", plugin_id, caller=self,
                )

    async def toggle_plugin(self, plugin_id: str):
        # the plugin is active, and evidently I am deactivating it: deactivate it in the Cheshire Cats before
        # deactivating it on a system level
        if plugin_id in self.plugin_manager.active_plugins:
            await self.on_plugin_deactivate(plugin_id)

        # toggle (activate or deactivate) the plugin
        await self.plugin_manager.toggle_plugin(plugin_id)

        # if the plugin is now active, activate it in the Cheshire Cats
        if plugin_id in self.plugin_manager.active_plugins:
            await self.on_plugin_activate(plugin_id)

        await self.plugin_manager.execute_hook("after_plugin_toggling_on_system", plugin_id, caller=self)

    def activate_plugin_endpoints(self, plugin_id: str):
        # Store endpoints for later activation
        self._pending_endpoints = safe_deepcopy(self.plugin_manager.plugins[plugin_id].endpoints)
        self._activate_pending_endpoints()

    async def on_plugin_activate(self, plugin_id: str) -> None:
        self.activate_plugin_endpoints(plugin_id)

        # if I already installed and activated the plugin and I am now re-installing it, then migrate plugin settings in
        # the Cheshire Cats to incrementally apply the new settings
        for ccat_id in await crud_plugins.get_agents_plugin_keys(plugin_id):
            # if the plugin is not active for the Cheshire Cat, then skip it
            if (ccat := await self._get_cheshire_cat_on_plugin_event(ccat_id, plugin_id)) is None:
                continue
            await ccat.plugin_manager.activate_plugin(plugin_id)

    async def on_plugin_deactivate(self, plugin_id: str):
        # deactivate the endpoints from the plugin
        if endpoints := self.plugin_manager.plugins[plugin_id].endpoints:
            for endpoint in endpoints:
                endpoint.deactivate(self.fastapi_app)

        for ccat_id in await crud_plugins.get_agents_plugin_keys(plugin_id):
            # if the plugin is not active for the Cheshire Cat, then skip it
            if (ccat := await self._get_cheshire_cat_on_plugin_event(ccat_id, plugin_id)) is None:
                continue
            await ccat.plugin_manager.deactivate_plugin(plugin_id)

    def _activate_pending_endpoints(self) -> None:
        for endpoint in self._pending_endpoints:
            endpoint.activate(self.fastapi_app)
        self._pending_endpoints.clear()

    async def shutdown(self) -> None:
        """
        Shuts down the Bill The Lizard Manager. It closes all the strays' connections and stops the scheduling system.

        Returns:
            None
        """
        await self.plugin_manager.execute_hook("before_lizard_shutdown", caller=self)
        if self.websocket_manager:
            await self.websocket_manager.close_connections()

        endpoints = [
            endpoint
            for plugin_id in self.plugin_manager.active_plugins
            for endpoint in self.plugin_manager.plugins[plugin_id].endpoints
        ]
        for endpoint in endpoints:
            endpoint.deactivate(self.fastapi_app)

        self.core_auth_handler = None
        self.plugin_manager = None
        self.rabbit_hole = None
        self.websocket_manager = None
        self.fastapi_app = None

    @property
    def fastapi_app(self):
        return self._fastapi_app

    @fastapi_app.setter
    def fastapi_app(self, app: FastAPI | None = None):
        self._fastapi_app = app

    @property
    def plugin_registry(self) -> PluginRegistry:
        return self._plugin_registry

    @plugin_registry.setter
    def plugin_registry(self, registry: PluginRegistry):
        self._plugin_registry = registry

    @property
    def agent_key(self):
        return DEFAULT_SYSTEM_KEY
