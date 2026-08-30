"""Re-embed engine for embedder changes (owned by the ingestion_status plugin).

This module hosts the re-embed logic that used to live in the core
(``CheshireCat.embed_stored_sources`` + ``BillTheLizard.embed_all_in_cheshire_cats``):
when the configured embedder changes, the stored sources of every agent are
re-embedded reusing the already-stored chunks (chunk-reuse) or, when no chunk
is reusable (e.g. the collection was just recreated), by re-ingesting the
source file/URL. Image points are re-embedded via ``embed_images`` only for
multimodal embedders, preserved payload-only otherwise.

Import-safe: nothing runs at import time.
"""

import asyncio
import os
import uuid

from langchain_core.documents import Document

from cat.core_plugins.ingestion_status.registry import (
    IngestionStatus,
    set_status,
)
from cat.db.cruds import settings as crud_settings
from cat.log import log
from cat.looking_glass.models import StoredSourceWithMetadata
from cat.services.factory.embedder import is_multimodal_embedder
from cat.services.ingestion_executor import run_in_ingestion_executor
from cat.services.memory.models import PointStruct, VectorMemoryType
from cat.utils import guess_file_type, is_url


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


async def reembed_sources(ccat, collection_name: VectorMemoryType, stored_sources: list[StoredSourceWithMetadata]) -> None:
    """
    Re-embed stored sources into a vector memory collection (chunk-reuse).

    When the embedder changed but the collection still holds the stored chunks
    (same-size/same-alias re-embed) the existing points are reused: only the
    vectors are recomputed. When a source has no reusable points (e.g. the
    collection was just recreated by ``initialize``) the source is re-ingested
    from its file/URL as a fallback.

    Ported verbatim from ``CheshireCat.embed_stored_sources``: the only changes
    are ``ccat`` as the parameter instead of ``self`` and the ingestion-status
    writes going through this plugin's ``registry.set_status``.
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
    counter = 0
    for source in stored_sources:
        source_name = source.name
        chat_id = source.metadata.get("chat_id")

        await _set_status(ccat, source_name, IngestionStatus.PROCESSING, chat_id=chat_id)
        try:
            # chunks already stored for this source (chunk-reuse candidates)
            source_points = [
                p
                for p in existing_points
                if (p.payload or {}).get("metadata", {}).get("source") == source_name
            ]

            if source_points:
                # For episodic sources, the chat must still exist; otherwise the
                # memory is orphaned and is cleaned up (matching the pre-reuse
                # behavior where the source was skipped after the collection wipe).
                if chat_id and not (stray_cat := await ccat._find_stray_cat(str(chat_id))):
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
                            if chat_id := meta.get("chat_id"):
                                root_dir = os.path.join(root_dir, str(chat_id))
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
                await _set_status(ccat, source_name, IngestionStatus.COMPLETED, chat_id=chat_id)
                continue

            # No reusable chunks for this source (e.g. the collection was just
            # recreated by initialize): fall back to a full re-ingest from the
            # source file/URL.
            content_type = None
            if source.content:
                content_type, _ = guess_file_type(source.content)

            cat = ccat
            if chat_id:
                if not (stray_cat := await ccat._find_stray_cat(str(chat_id))):
                    log.warning(
                        f"Stray cat with id {chat_id} not found. Skipping file {source.path}/{source.name}"
                    )
                    continue
                cat = stray_cat

            await rabbit_hole.ingest_file(
                cat=cat,
                file=source.content or source.name,
                filename=source.name,
                metadata=source.metadata or {},
                store_file=False,
                content_type=content_type,
            )
            counter += 1
            await _set_status(ccat, source_name, IngestionStatus.COMPLETED, chat_id=chat_id)
        except Exception as e:  # noqa: BLE001 - a failing source must not abort the pass
            log.error(f"Agent id: {ccat._id}. Error re-embedding source {source_name}: {e}")
            await _set_status(ccat, source_name, IngestionStatus.ERROR, error=str(e), chat_id=chat_id)

    log.info(f"Agent id: {ccat._id}. Embedded {counter} files to the vector memory")


async def reembed_all(lizard, embedder_name: str, embedder_size: int) -> bool:
    """Re-embed all the stored files and procedures in all the Cheshire Cats.

    Port of ``BillTheLizard.embed_all_in_cheshire_cats``. Fixes the original
    O(agents) duplication: the per-entry gathering used to run once per entry
    inside the initialization loop (re-embedding every agent N times). Now all
    databases are serialized-initialized first, then the re-embeds run once
    with a concurrency cap.
    """
    success = False
    try:
        ccat_ids = await crud_settings.get_agents_main_keys()
        stored_files_by_ccat = []
        # first, get all the stored files from all the Cheshire Cats with the
        # metadata stored within the vector memory; nothing is removed from the
        # latter to avoid any race condition
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
        # embeddings to avoid overwhelming resources
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent

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
    except Exception as e:  # noqa: BLE001 - re-embed failure is surfaced on the hook, never raised
        log.error(f"Error embedding all stored files: {e}")

    await lizard.plugin_manager.execute_hook(
        "after_all_cheshire_cats_embedded", success, caller=lizard,
    )
    return success
