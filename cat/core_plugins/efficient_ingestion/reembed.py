"""Efficient re-embed engine (owned by the efficient_ingestion plugin).

Implements ``EfficientIngestionEngine`` — the replaceable, more efficient
implementation of the core ``BaseIngestionEngine``: on embedder changes the
stored sources of every agent are re-embedded reusing the already-stored
chunks (chunk-reuse) or, when no chunk is reusable (e.g. the collection was
just recreated), by re-ingesting the source file/URL. Image points are
re-embedded via ``embed_images`` only for multimodal embedders, preserved
payload-only otherwise. Status is written through ``ingestion_status.registry``.

Import-safe: nothing runs at import time.
"""

import asyncio
import os
import uuid

from langchain_core.documents import Document

from cat.core_plugins.efficient_ingestion.ingestion_executor import (
    run_in_ingestion_executor,
)
from cat.core_plugins.ingestion_status.registry import (
    PHASE_EMBEDDING,
    PHASE_PARSING_CHUNKING,
    IngestionStatus,
    claim_source_for_resume,
    get_status,
    set_phase,
    set_status,
)
from cat.db.cruds import settings as crud_settings
from cat.env import get_env_int
from cat.log import log
from cat.looking_glass.models import StoredSourceWithMetadata
from cat.services.factory.embedder import is_multimodal_embedder
from cat.services.factory.ingestion import BaseIngestionEngine
from cat.services.memory.models import PointStruct, VectorMemoryType
from cat.utils import get_nlp_object_name, guess_file_type, is_url


async def _set_status(ccat, source: str, status: IngestionStatus, error: str | None = None, chat_id: str | None = None) -> None:
    """Best-effort status write via the plugin's own registry."""
    scope = str(chat_id) if chat_id else "agent"
    try:
        await set_status(
            ccat.agent_key,
            scope,
            source,
            type_="url" if is_url(source) else "file",
            status=status,
            chat_id=chat_id,
            error=error,
        )
    except Exception as e:  # noqa: BLE001 - status must never break the re-embed pass
        log.error(f"Agent id: {ccat._id}. Failed to write ingestion status for {source}: {e}")


def _claim_stale_after() -> float:
    """Staleness gate for the per-source claim during the re-embed pass.

    Reuses the resume threshold so a source being actively re-embedded by
    another worker is not re-claimed. ``<=0`` via env disables the gate.
    """
    value = get_env_int("CAT_INGESTION_RESUME_STALE_SECONDS")
    return float(value) if value and value > 0 else 0.0


async def _cleanup_orphan_images(ccat, collection_name, source_name, chat_id) -> None:
    """Remove image points + saved image files of a source before a full re-ingest.

    A full re-ingest (parsing_chunking) regenerates the chunks AND re-extracts
    the images from scratch. The image points and their saved ``image_file``
    from the PREVIOUS (possibly incomplete) parse are therefore stale: delete
    the points in the target first and the files from the agent storage, so the
    re-ingest starts clean and no orphan image file lingers on disk.
    """
    try:
        points, _ = await ccat.vector_memory_handler.get_all_tenant_points(
            str(collection_name), with_vectors=False,
            metadata={"source": source_name, "image": True},
        )
    except Exception:  # noqa: BLE001 - cleanup must never break the pass
        return
    image_files = [
        (p.payload or {}).get("metadata", {}).get("image_file")
        for p in points
        if (p.payload or {}).get("metadata", {}).get("image_file")
    ]
    if not image_files:
        return
    root_dir = ccat.agent_key
    if chat_id:
        root_dir = os.path.join(root_dir, str(chat_id))
    for image_file in image_files:
        try:
            ccat.file_manager.remove_file(os.path.join(root_dir, image_file))
        except Exception:  # noqa: BLE001,S110 - best-effort, cleanup must never break the pass
            pass
    await ccat.vector_memory_handler.delete_tenant_points(
        str(collection_name), metadata={"source": source_name, "image": True}
    )


async def _resolve_callers(ccat, chat_id):
    """Return the RabbitHole/plugin-manager caller and the embedding cat for a source.

    Agent-scoped sources use the CheshireCat itself (scope ``"agent"``); a
    chat-scoped source needs its StrayCat for the correct ``scope``/hook caller.
    Returns ``(cat, scope, chat_id)`` where ``cat`` is what receives the
    ``after_rabbithole_stored_documents`` hook caller, or ``(None, None, None)``
    when the chat does not (or no longer) exist.
    """
    if not chat_id:
        return ccat, "agent", None
    stray_cat = await ccat._find_stray_cat(str(chat_id))
    if stray_cat is None:
        return None, str(chat_id), chat_id
    return stray_cat, str(chat_id), chat_id


async def reembed_sources(ccat, collection_name: VectorMemoryType, stored_sources: list[StoredSourceWithMetadata]) -> None:
    """
    Re-embed stored sources into a vector memory collection (phase machine).

    The re-embed pass is crash-recoverable per source via the ``ingestion_status``
    doc: for each source it decides the start phase from ``status`` +
    ``embedder_name``/``chunker_name`` + the current embedder/chunker, claims the
    per-source work (so two workers never process the SAME source), records the
    phase, executes it and, on the chunk-reuse path, fires the public
    ``after_rabbithole_stored_documents`` hook so analytics/webhooks are informed.

    Phase decision (per source):
      - ``completed`` + embedder == active + chunker == active  -> skip (nothing to do)
      - ``completed`` + chunker mismatch                         -> ``parsing_chunking`` (full re-ingest, chunks are stale)
      - ``completed`` + embedder mismatch (chunks valid)         -> ``embedding`` (chunk-reuse)
      - in-flight / stale row (``uploaded``/``processing``/``error``/``downloading``/``downloaded``):
          resumes from the phase recorded in ``doc["phase"]`` (``embedding`` -> chunk-reuse,
          otherwise ``parsing_chunking``), with a conservative fallback to full re-ingest.
      - no status doc / old row                                -> ``embedding`` if reusable points exist, else full re-ingest.
    """
    log.info(f"Agent id: {ccat._id}. Embedding stored files to the vector memory")

    # Collect the existing points once, up-front. We do NOT clear the collection
    # here: on an embedder size/alias change ``initialize`` already recreated it
    # (empty -> no chunks to reuse -> full re-ingest fallback below), while on a
    # same-size re-embed the points persist and are reused (only vectors change).
    existing_points, _ = await ccat.vector_memory_handler.get_all_tenant_points(
        str(collection_name), with_vectors=False
    )

    rabbit_hole = ccat.rabbit_hole
    embedder = await ccat.embedder()
    active_embedder_name = getattr(embedder, "name", None) or get_nlp_object_name(embedder, "default_embedder")
    chunker = getattr(ccat, "chunker", None)
    active_chunker_name = (
        str(chunker.name) if chunker is not None and getattr(chunker, "name", None) else
        get_nlp_object_name(chunker, "default_chunker")
    )
    owner = f"reembed-{os.getpid()}"
    counter = 0

    for source in stored_sources:
        source_name = source.name
        chat_id = source.metadata.get("chat_id")
        cat, scope, chat_id = await _resolve_callers(ccat, chat_id)
        if cat is None:
            log.warning(
                f"Stray cat with id {chat_id} not found. Skipping file {source.path}/{source.name}"
            )
            continue

        # ---- decide the start phase from the status doc ----
        doc = await get_status(ccat.agent_key, scope, source_name)
        doc_status = (doc or {}).get("status")
        doc_embedder = (doc or {}).get("embedder_name")
        doc_chunker = (doc or {}).get("chunker_name")
        doc_phase = (doc or {}).get("phase")

        if (
            doc_status == IngestionStatus.COMPLETED.value
            and doc_embedder == active_embedder_name
            and doc_chunker == active_chunker_name
        ):
            # already up to date for the current embedder AND chunker
            log.debug(
                f"Agent id: {ccat._id}. Source {source_name}: already completed with the "
                f"active embedder/chunker ({active_embedder_name!r}/{active_chunker_name!r}), skipping"
            )
            continue

        if doc_status == IngestionStatus.COMPLETED.value:
            if doc_chunker != active_chunker_name:
                start_phase = PHASE_PARSING_CHUNKING
            else:
                start_phase = PHASE_EMBEDDING
        elif doc_status in (
            IngestionStatus.UPLOADED.value,
            IngestionStatus.PROCESSING.value,
            IngestionStatus.ERROR.value,
            IngestionStatus.DOWNLOADING.value,
            IngestionStatus.DOWNLOADED.value,
        ):
            # in-flight / stale: resume from the recorded phase (conservative).
            # A chunker change invalidates even an embedding-phase row: the stored
            # chunks were produced by the OLD chunker, so a chunk-reuse would keep
            # stale chunks -> full re-ingest (parsing_chunking) instead.
            if doc_chunker != active_chunker_name:
                start_phase = PHASE_PARSING_CHUNKING
            else:
                start_phase = doc_phase if doc_phase in (PHASE_EMBEDDING, PHASE_PARSING_CHUNKING) else PHASE_PARSING_CHUNKING
        else:
            # no doc or unknown: embedding if reusable points exist, else full re-ingest
            has_points = any(
                (p.payload or {}).get("metadata", {}).get("source") == source_name
                for p in existing_points
            )
            start_phase = PHASE_EMBEDDING if has_points else PHASE_PARSING_CHUNKING

        # ---- log the phase transition (debug) ----
        log.debug(
            f"Agent id: {ccat._id}. Source {source_name}: ingestion phase "
            f"{doc_phase or '(none)'} -> {start_phase} (status {doc_status or '(none)'} -> "
            f"{IngestionStatus.PROCESSING.value}, embedder {doc_embedder!r} -> {active_embedder_name!r}, "
            f"chunker {doc_chunker!r} -> {active_chunker_name!r})"
        )

        # ---- claim the per-source work (skip if another worker holds it) ----
        # Only rows with an existing status doc need (and support) a claim: a
        # missing doc means no worker has registered this source yet, so there
        # is nothing to double-process and no lock to contend on.
        if doc is not None:
            claimed = await claim_source_for_resume(
                ccat.agent_key,
                scope,
                source_name,
                stale_after=_claim_stale_after(),
                owner=owner,
                claim_completed=(doc_status == IngestionStatus.COMPLETED.value),
            )
            if claimed is None:
                # another worker is already (re)processing this source
                continue

        # ---- record the phase, then execute ----
        if start_phase == PHASE_EMBEDDING:
            await set_phase(
                ccat.agent_key, scope, source_name,
                PHASE_EMBEDDING,
                embedder_name=active_embedder_name,
                type_="url" if is_url(source_name) else "file",
                chat_id=chat_id,
            )
        else:
            await set_phase(
                ccat.agent_key, scope, source_name,
                PHASE_PARSING_CHUNKING,
                chunker_name=active_chunker_name,
                type_="url" if is_url(source_name) else "file",
                chat_id=chat_id,
            )

        try:
            # chunks already stored for this source (chunk-reuse candidates)
            source_points = [
                p
                for p in existing_points
                if (p.payload or {}).get("metadata", {}).get("source") == source_name
            ]

            if source_points and start_phase == PHASE_EMBEDDING:
                # For episodic sources, the chat must still exist; otherwise the
                # memory is orphaned and is cleaned up (matching the pre-reuse
                # behavior where the source was skipped after the collection wipe).
                if chat_id and not (await ccat._find_stray_cat(str(chat_id))):
                    log.warning(
                        f"Stray cat with id {chat_id} not found. Cleaning up {source.path}/{source.name}"
                    )
                    await ccat.vector_memory_handler.delete_tenant_points(
                        str(collection_name), metadata={"source": source_name}
                    )
                    await _set_status(ccat, source_name, IngestionStatus.COMPLETED, chat_id=chat_id)
                    continue

                # Rebuild the LangChain documents from the stored chunks, dropping
                # the vectors: only the embedding changed, not the content.
                # Image points (metadata.image == True) are handled separately:
                # they must never be embedded as plain text.
                source_points_text = [
                    p
                    for p in source_points
                    if not (p.payload or {}).get("metadata", {}).get("image")
                ]
                source_points_image = [
                    p
                    for p in source_points
                    if (p.payload or {}).get("metadata", {}).get("image")
                ]

                points = []

                # --- text points: existing path unchanged ---
                if source_points_text:
                    docs = [
                        Document(
                            page_content=(p.payload or {}).get("page_content", ""),
                            metadata=dict((p.payload or {}).get("metadata", {})),
                        )
                        for p in source_points_text
                    ]

                    # Re-chunk only chunks exceeding the NEW embedder's input limit.
                    docs = rabbit_hole._split_oversized(docs, embedder)

                    vectors = await run_in_ingestion_executor(
                        embedder.embed_documents, [d.page_content for d in docs]
                    )
                    points.extend(
                        PointStruct(
                            id=uuid.uuid4().hex,
                            payload=d.model_dump(),
                            vector=vector,
                        )
                        for d, vector in zip(docs, vectors)
                    )

                # --- image points: only re-embed via embed_images when multimodal ---
                if source_points_image:
                    if is_multimodal_embedder(embedder):
                        # Recover the saved image bytes (same agent_key[/chat_id]
                        # layout as get_stored_sources_with_metadata/save_file) and
                        # re-embed them in a single embed_images call.
                        recoverable = []
                        for p in source_points_image:
                            meta = (p.payload or {}).get("metadata", {})
                            image_file = meta.get("image_file")
                            root_dir = ccat.agent_key
                            if metadata_chat := meta.get("chat_id"):
                                root_dir = os.path.join(root_dir, str(metadata_chat))
                            image_bytes = (
                                ccat.file_manager.read_file(image_file, root_dir)
                                if image_file
                                else None
                            )
                            if image_bytes is None:
                                # H2 fallback: the image file is gone; keep the
                                # point payload-only (no vector) instead of failing
                                # the whole source.
                                points.append(
                                    PointStruct(
                                        id=uuid.uuid4().hex,
                                        payload={
                                            "page_content": f"[Image] {source_name}",
                                            "metadata": dict(meta),
                                        },
                                        vector={},
                                    )
                                )
                            else:
                                recoverable.append((p, image_bytes))
                        if recoverable:
                            image_vectors = await run_in_ingestion_executor(
                                embedder.embed_images, [b for _, b in recoverable]
                            )
                            if len(image_vectors) != len(recoverable):
                                # The embedder silently dropped vectors (or the
                                # batch was truncated): abort before the delete
                                # so no point is lost (compute-before-delete) and
                                # the source is marked error, not completed.
                                raise ValueError(
                                    f"embed_images returned {len(image_vectors)} vectors "
                                    f"for {len(recoverable)} images"
                                )
                            for (p, _b), vector in zip(recoverable, image_vectors):
                                meta = dict((p.payload or {}).get("metadata", {}))
                                points.append(
                                    PointStruct(
                                        id=uuid.uuid4().hex,
                                        payload={
                                            "page_content": f"[Image] {source_name}",
                                            "metadata": meta,
                                        },
                                        vector=vector,
                                    )
                                )
                    else:
                        # Non-multimodal embedder image handling: keep the image
                        # points payload-only (vector={}) so they are preserved
                        # and recoverable rather than dropped or mis-embedded as
                        # text. The full original payload is copied over; the
                        # empty vector removes the point from ANN similarity
                        # recall while a scroll still returns it.
                        for p in source_points_image:
                            meta = dict((p.payload or {}).get("metadata", {}))
                            points.append(
                                PointStruct(
                                    id=uuid.uuid4().hex,
                                    payload={
                                        "page_content": f"[Image] {source_name}",
                                        "metadata": meta,
                                    },
                                    vector={},
                                )
                            )

                # Clear the source's existing points first so the re-embedded
                # points replace them instead of duplicating them (the reuse
                # branch assigns fresh ids, so without this delete the old and
                # new points would coexist for the same source). All vectors are
                # computed above BEFORE the delete so a failure leaves the old
                # points intact.
                await ccat.vector_memory_handler.delete_tenant_points(
                    str(collection_name), metadata={"source": source_name}
                )
                await ccat.vector_memory_handler.add_points_to_tenant(
                    collection_name=str(collection_name), points=points
                )
                counter += 1

                # Token accounting + completion signal on the chunk-reuse path:
                # fire the SAME public hook rabbit_hole fires after storing, so
                # analytics (token usage) and any consumer of "source stored"
                # (e.g. ingestion_status -> COMPLETED) are informed. We do NOT
                # fire before_rabbithole_*: the reused chunks are already
                # catalogued and must not be re-split/summarized.
                await ccat.plugin_manager.execute_hook(
                    "after_rabbithole_stored_documents", source_name, points, caller=cat,
                )
                continue

            # Full re-ingest path (parsing_chunking): chunks are missing or stale.
            # First drop any orphan image files/points left by a previous
            # incomplete parse, so the re-ingest starts clean.
            await _cleanup_orphan_images(ccat, collection_name, source_name, chat_id)

            content_type = None
            if source.content:
                content_type, _ = guess_file_type(source.content)

            await rabbit_hole.ingest_file(
                cat=cat,
                file=source.content or source.name,
                filename=source.name,
                metadata=source.metadata or {},
                store_file=False,
                content_type=content_type,
            )
            counter += 1
            # ingest_file already fires after_rabbithole_stored_documents ->
            # ingestion_status sets the row to COMPLETED (status accounting).
        except Exception as e:  # noqa: BLE001 - a failing source must not abort the pass
            log.error(f"Agent id: {ccat._id}. Error re-embedding source {source_name}: {e}")
            await _set_status(ccat, source_name, IngestionStatus.ERROR, error=str(e), chat_id=chat_id)

    log.info(f"Agent id: {ccat._id}. Embedded {counter} files to the vector memory")


class EfficientIngestionEngine(BaseIngestionEngine):
    """Efficient re-embed engine (chunk-reuse, image recovery, status writes).

    Pluggable implementation of the base ingestion engine, registered by the
    plugin through the ``factory_allowed_ingestions`` hook as
    ``EfficientIngestionConfiguration``. Unlike the base (upstream) engine it:
    reuses the stored chunks instead of re-parsing every source, recovers the
    image points via ``embed_images`` when the embedder is multimodal, writes
    the ingestion status through ``ingestion_status.registry``, and honors a
    configurable concurrency cap (settings category ``re-ingestion``).
    """

    def __init__(self, reembed_max_concurrency: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.reembed_max_concurrency = max(1, int(reembed_max_concurrency))

    async def run(self, lizard) -> bool:
        """Resolve the current embedder and run the re-embed pass."""
        success = False
        try:
            embedder = await lizard.embedder()
            embedder_name = embedder.name
            embedder_size = embedder.size

            ccat_ids = await crud_settings.get_agents_main_keys()
            stored_files_by_ccat = []
            # first, get all the stored files from all the Cheshire Cats with the
            # metadata stored within the vector memory; nothing is removed from
            # the latter to avoid any race condition
            for ccat_id in ccat_ids:
                if (ccat := await lizard.get_cheshire_cat(ccat_id)) is None:
                    continue
                stored_files_by_ccat.append({
                    "ccat": ccat,
                    "stored_sources": await ccat.get_stored_sources_with_metadata(),
                })

            # re-initialize all the vector databases in a serialized way, outside
            # threads to avoid race conditions
            for entry in stored_files_by_ccat:
                await entry["ccat"].vector_memory_handler.initialize(embedder_name, embedder_size)

            # then re-embed every stored file/procedure, limiting concurrent
            # embeddings to avoid overwhelming resources (tunable via the plugin
            # settings, category 're-ingestion')
            semaphore = asyncio.Semaphore(self.reembed_max_concurrency)

            async def embed_with_limit(entry_):
                async with semaphore:
                    tasks = [
                        reembed_sources(entry_["ccat"], collection_name, sources)
                        for collection_name, sources in entry_["stored_sources"].items()
                        if sources
                    ] + [entry_["ccat"].embed_procedures()]
                    await asyncio.gather(*tasks)

            await asyncio.gather(*[embed_with_limit(entry) for entry in stored_files_by_ccat])

            success = True
        except Exception as e:  # noqa: BLE001 - surfaced on the hook, never raised
            log.error(f"Error embedding all stored files: {e}")

        await lizard.plugin_manager.execute_hook(
            "after_all_cheshire_cats_embedded", success, caller=lizard,
        )
        return success
