import asyncio
import base64
import hashlib
import json
import mimetypes
import os
import re
import time
import uuid
from io import BytesIO
from typing import List, Dict, Tuple
from httpx import AsyncClient
from langchain_community.document_loaders.parsers.generic import MimeTypeBasedParser
from langchain_core.documents.base import Document, Blob

from cat.log import log
from cat.services.factory.chunker import BaseChunker
from cat.services.memory.models import VectorMemoryType, PointStruct
from cat.services.service_factory import ServiceFactory
from cat.utils import is_url as fnc_is_url


class RabbitHole:
    def __init__(self):
        self.cat = None
        self.stray = None
        self.embedder = None

    async def setup(self, _cat: "BotMixin"):  # type: ignore[name-defined]
        from cat.looking_glass import CheshireCat, StrayCat

        if isinstance(_cat, CheshireCat):
            self.cat = _cat
            self.stray = None
            return

        if isinstance(_cat, StrayCat):
            self.stray = _cat
            self.cat = await _cat.lizard.get_cheshire_cat(_cat.agent_key)
            return

        raise ValueError("RabbitHole can only be setup with CheshireCat or StrayCat instances.")

    """Manages content ingestion. I'm late... I'm late!"""

    async def ingest_memory(self, cat: "CheshireCat", file: BytesIO, filename: str):  # type: ignore[name-defined]
        """Upload memories to the declarative memory from a JSON file.

        Args:
            cat (CheshireCat): Cheshire Cat instance.
            file (BytesIO): JSON file containing vector and content memories.
            filename (str): Filename of the uploaded file.

        Notes
        -----
        This method allows uploading a JSON file containing vector and content memories directly to the declarative
        memory.
        When doing this, please, make sure the embedder used to export the memories is the same as the one used
        when uploading.
        The method also performs a check on the dimensionality of the embeddings (i.e. length of each vector).
        """
        try:
            await self.setup(cat)
            lizard = self.cat.lizard

            # fire the hook with the source before the memories are stored
            await self.cat.plugin_manager.execute_hook(
                "rabbithole_ingestion_start", filename, {}, False, caller=self.cat,
            )

            # Load fyle byte in a dict
            memories = json.loads(file.read().decode("utf-8"))

            # Check the embedder used for the uploaded memories is the same the Cat is using now
            upload_embedder = memories["embedder"]
            embedder = await lizard.embedder()
            cat_embedder = str(embedder.__class__.__name__)
            if upload_embedder != cat_embedder:
                raise Exception(f"Embedder mismatch for file '{filename}': file embedder {upload_embedder} is different from {cat_embedder}")

            # Get Declarative memories in file
            declarative_memories = memories["collections"][str(VectorMemoryType.DECLARATIVE)]
            if not declarative_memories:
                raise Exception(f"No Declarative memories found in the uploaded file '{filename}'.")

            # Store data to upload the memories in batch
            points = [PointStruct(
                id=m["id"],
                payload={"page_content": m["page_content"], "metadata": m["metadata"]},
                vector=m["vector"],
            ) for m in declarative_memories]

            log.info(f"Agent id: {self.cat.agent_key}. Preparing to load {len(points)} vector memories")

            # Check embedding size is correct
            embedder = await lizard.embedder()
            embedder_size = embedder.size
            len_mismatch = [len(p.vector) == embedder_size for p in points]  # type: ignore[union-attr]

            if not any(len_mismatch):
                raise Exception(f"Embedding size mismatch for file '{filename}': vectors length should be {embedder_size}")

            # Upsert memories in batch mode
            await cat.vector_memory_handler.add_points_to_tenant(
                collection_name=str(VectorMemoryType.DECLARATIVE), points=points,
            )
        except Exception as e:
            log.error(f"Error uploading memories from file '{filename}': {e}")
            # fire the error hook alongside the existing log
            await self.cat.plugin_manager.execute_hook(
                "rabbithole_ingestion_error", filename, str(e), caller=self.cat,
            )

    async def ingest_file(
        self,
        cat: "BotMixin",  # type: ignore[name-defined]
        file: str | BytesIO,
        metadata: Dict,
        filename: str | None = None,
        store_file: bool = True,
        content_type: str | None = None,
    ):
        """
        Load a file in the Cat's declarative memory.

        The method splits and converts the file in Langchain `Document`. Then, it stores the `Document` in the Cat's
        memory.

        Args:
            cat (CheshireCat | StrayCat): Cheshire Cat or Stray Cat instance.
            file (str | BytesIO): The file can be a path passed as a string or a `BytesIO` object if the document is ingested using the `rabbithole` endpoint.
            metadata (Dict): Metadata to be stored with each chunk.
            filename (str): The filename of the file to be ingested, if coming from the `/rabbithole/` endpoint.
            store_file (bool): Whether to store the file in the Cat's file storage.
            content_type (str): The content type of the file. If not provided, it will be guessed based on the file extension.

        See Also:
            before_rabbithole_stores_documents
        """
        source = ""
        points = []

        try:
            await self.setup(cat)

            filename = filename or (file if isinstance(file, str) else None)
            if not filename:
                raise ValueError("No filename provided.")

            # fire the hook with the source (the filename; for URLs the filename IS the URL)
            # before the file is downloaded/parsed, so plugins can observe the full lifecycle
            await self.cat.plugin_manager.execute_hook(
                "rabbithole_ingestion_start", filename, metadata, filename.startswith("http"),
                caller=self.stray or self.cat,
            )

            # split a file into a list of docs
            source, file_bytes, content_type, docs, images, is_url = await self._file_to_docs(
                file=file, filename=filename, content_type=content_type
            )

            if not docs:
                raise Exception(f"No valid chunks found in the file '{filename}'.")

            # fire the hook with the resolved source before the docs are embedded/stored
            await self.cat.plugin_manager.execute_hook(
                "rabbithole_ingestion_processing", source, caller=self.stray or self.cat,
            )

            # store in memory
            sha256 = hashlib.sha256()
            sha256.update(file_bytes)
            points = await self.store_documents(
                docs=docs, source=source, file_hash=sha256.hexdigest(), metadata=metadata, images=images,
                source_bytes=file_bytes,
            )

            # store in file storage
            if store_file and not is_url:
                chat_id = self.stray.id if self.stray else None
                await self.cat.save_file(file_bytes, content_type, source, chat_id)

            # notify client
            images_info = f" and {len(images)} images" if images else ""
            await self._send_notification_message(
                f"Finished reading {source}, I made {len(docs)} thoughts{images_info} on it."
            )

            log.info(f"Agent id: {self.cat.agent_key}. Successfully ingested file: {filename}")
        except Exception as e:
            log.error(f"Error ingesting file {filename}: {e}")
            # Don't raise in background tasks - just log the error
            if self.stray:
                try:
                    await self.stray.notifier.send_error(f"Error processing {filename}: {str(e)}")
                except Exception as notify_error:
                    log.error(f"Failed to send error notification: {notify_error}")
            # fire the error hook alongside the existing log/notify
            await self.cat.plugin_manager.execute_hook(
                "rabbithole_ingestion_error", source or filename, str(e), caller=self.stray or self.cat,
            )
        finally:
            # hook the points after they are stored in the vector memory
            await self.cat.plugin_manager.execute_hook(
                "after_rabbithole_stored_documents", source, points, caller=self.stray or self.cat,
            )

    async def _file_to_docs(
        self, file: str | BytesIO, filename: str, content_type: str | None = None
    ) -> Tuple[str, bytes, str | None, List[Document], List[Dict], bool]:
        """
        Load and convert files to Langchain `Document`.

        This method takes a file either from a Python script, from the `/rabbithole/` or `/rabbithole/web` endpoints.
        Hence, it loads it in memory and splits it in chunks.

        Args:
            file (str | BytesIO): The file can be either a string path if loaded programmatically, a `BytesIO` if coming from the `/rabbithole/` endpoint, or a URL if coming from the `/rabbithole/web` endpoint.
            filename (str): The filename of the file to be ingested.
            content_type (str): The content type of the file. If not provided, it will be guessed based on the file extension.

        Returns:
            (source, file_bytes, content_type, docs, images, is_url): Tuple[str, bytes, str | None, List[Document], List[Dict], bool].
                The file name, the file content in bytes, the content type, the list of chunked Langchain `Document`,
                the list of images extracted by multimodal parsers (empty for text-only embedders) and
                a boolean indicating if the file was loaded from a URL.
        """
        def sanitize_filename(file_name: str) -> str:
            if "." not in file_name:
                return file_name
            # Split on the LAST dot only (if any)
            base, ext = file_name.rsplit(".", 1)
            ext = "." + ext
            # Replace any sequence of dots or spaces in the base name only
            base = re.sub(r"[.\s]+", "_", base)
            return base + ext

        async def parse() -> Tuple[str | None, bytes | None, str | None, bool]:
            if isinstance(file, BytesIO):
                # Get the source of UploadFile, file bytes and whether it's a URL
                return sanitize_filename(filename), file.read(), content_type, False
            if fnc_is_url(file):
                try:
                    # notify plugins that the URL download is about to start
                    await self.cat.plugin_manager.execute_hook(
                        "rabbithole_url_downloading", file, filename, caller=self.stray or self.cat,
                    )
                    # Make a request with a fake browser name - use async httpx
                    async with AsyncClient() as client:
                        response = await client.get(file, headers={"User-Agent": "Magic Browser"})
                        response.raise_for_status()
                        # Define mime type and source of url
                        # Add fallback for empty/None content_type
                        ct = response.headers.get(
                            "Content-Type", "text/html" if file.startswith("http") else "text/plain"
                        ).split(";")[0]
                        # Get binary content of url
                        content = response.content
                    # notify plugins that the URL download completed successfully
                    await self.cat.plugin_manager.execute_hook(
                        "rabbithole_url_download_completed", file, filename, caller=self.stray or self.cat,
                    )
                    return file, content, ct, True
                except Exception as e:
                    log.error(f"Agent id: {self.cat.agent_key}. Error: {e}")
                    return None, None, content_type, True
            # Get file bytes - use async file reading
            fb = await asyncio.to_thread(lambda: open(file, "rb").read())  # type: ignore[union-attr]
            return sanitize_filename(os.path.basename(file)), fb, mimetypes.guess_type(file)[0], False  # type: ignore[return-value]

        if not isinstance(file, BytesIO) and not isinstance(file, str):
            raise ValueError(f"{type(file)} is not a valid type.")

        # Check the characteristics of the incoming file.
        source, file_bytes, content_type, is_url = await parse()
        if not file_bytes:
            raise ValueError(f"Something went wrong with the source '{source}'")

        fh = await self.cat.file_handlers()
        log.debug(f"Attempting to parse source: {source}. Detected MIME type: {content_type}. Available handlers: {list(fh.keys())}")

        # Load the bytes in the Blob schema and parse the content. Parser based on the mime type.
        # The parser is CPU/IO-bound (e.g. PyMuPDF on a large PDF): run it off the event loop.
        await self._send_notification_message("I'm parsing the content. Big content could require some minutes...")
        super_docs = await asyncio.to_thread(
            lambda: MimeTypeBasedParser(handlers=fh).parse(
                Blob(data=file_bytes, mimetype=content_type).from_data(data=file_bytes, mime_type=content_type, path=source)
            )
        )

        # Propagate the source to every parsed document BEFORE chunking, so that
        # hooks such as `before_rabbithole_splits_documents` can rely on it
        # (metadata['source'] is otherwise only added later, in store_documents).
        # setdefault preserves a more specific source set by the parser itself.
        for doc in super_docs:
            if isinstance(doc.metadata, dict):
                doc.metadata.setdefault("source", source)

        # Collect the images extracted by multimodal parsers (if a multimodal embedder is active).
        # This must happen BEFORE chunking: the chunkers may drop or alter the metadata that carries
        # the image payload (e.g. PLUS SemanticChunker discards metadata, _merge_short_chunks keeps
        # only the first chunk's metadata).
        images = self._collect_document_images(super_docs) if await self._is_multimodal_embedder() else []

        # The image payload was consumed above to embed and save the images as files:
        # strip it from the parsed documents so it is neither duplicated into every
        # text chunk's metadata nor stored in the vector DB (where it would bloat the
        # payloads and be forwarded to the LLM on recall).
        if images:
            for doc in super_docs:
                doc.metadata.pop("image_base64", None)

        # Split
        await self._send_notification_message("Parsing completed. Now let's go with reading process...")
        docs = await self._split_text(docs=super_docs)
        return source, file_bytes, content_type, docs, images, is_url  # type: ignore[return-value]

    async def _is_multimodal_embedder(self) -> bool:
        """Check whether the active embedder supports multimodality.

        This mirrors the detection used by the PLUS rabbit_hole hook: the active embedder
        instance is resolved to its settings class and `is_multimodal()` tells whether the
        images extracted by multimodal parsers should be embedded and stored in memory.

        The embedder factory must be resolved with the LIZARD's plugin manager (system
        context): the `factory_allowed_embedders` hooks are declared with a `lizard`
        parameter (core `base_plugin` and PLUS alike), and `MadHatter.context_execute_hook`
        passes the caller under the keyword `lizard` only when the executing manager belongs
        to BillTheLizard. Using an agent plugin manager would pass `cat` instead and make
        those hooks raise `TypeError: unexpected keyword argument 'cat'`.
        """
        if not self.cat:
            return False

        lizard = self.cat.lizard
        sp = ServiceFactory(
            agent_key=lizard.agent_key,
            hook_manager=lizard.plugin_manager,
            factory_allowed_handler_name="factory_allowed_embedders",
            setting_category="embedder",
            schema_name="languageEmbedderName",
        )
        embedder_config = await sp.get_config_class_from_adapter(
            await lizard.embedder()
        )
        return bool(embedder_config) and embedder_config.is_multimodal()

    def _collect_document_images(self, docs: List[Document]) -> List[Dict]:
        """Collect the images extracted by multimodal parsers from the parsed documents.

        Multimodal parsers (e.g. the PLUS `UnstructuredParser` configured with
        `extract_image_block_to_payload=True`) attach each extracted image to the parsed
        `Document` metadata as a base64-encoded payload in `image_base64`, with the mime
        type in `image_mime_type`. This helper walks the parsed documents and returns one
        entry per image, carrying the raw bytes ready to be embedded and the base64 payload
        to be stored in the vector memory metadata.

        Returns:
            List[Dict]: A list of dicts with keys ``image_base64``, ``image_bytes`` and
            ``image_mime_type`` (defaulting to ``image/jpeg``).
        """
        images: List[Dict] = []
        for doc in docs:
            image_base64 = doc.metadata.get("image_base64")
            if not image_base64:
                continue
            images.append({
                "image_base64": image_base64,
                "image_bytes": base64.b64decode(image_base64),
                "image_mime_type": doc.metadata.get("image_mime_type", "image/jpeg"),
            })
        return images

    async def store_documents(
        self,
        docs: List[Document],
        source: str,
        file_hash: str | None = None,
        metadata: Dict | None = None,
        images: List[Dict] | None = None,
        source_bytes: bytes | None = None,
    ) -> List[PointStruct]:
        """Add documents to the Cat's declarative memory.

        This method loops a list of Langchain `Document` and adds some metadata. Namely, the source filename and the
        timestamp of insertion. Once done, the method notifies the client via Websocket connection.

        If a multimodal embedder is active and the multimodal parsers extracted some images, this method also embeds
        the images via ``embed_images``, saves them as files in the agent/chat storage via ``save_file`` and stores
        them in the same collection, keeping the file name in the point metadata (``image_file``, no base64 payload).

        Args:
            docs (List[Document]): List of Langchain `Document` to be inserted in the Cat's declarative memory.
            source (str): Source name to be added as a metadata. It can be a file name or an URL.
            file_hash (str | None): Optional hash of the source to be added as a metadata.
            metadata (Dict | None): Optional metadata to be stored with each chunk.
            images (List[Dict] | None): Optional images extracted by multimodal parsers. Each entry has
                ``image_base64``, ``image_bytes`` and ``image_mime_type`` keys. The images are embedded and saved
                as files via ``save_file``; the point metadata only keeps the file name in ``image_file``.
            source_bytes (bytes | None): Optional raw bytes of the ingested source file. Used when the source
                itself is an image: the file is embedded as a single whole-image point instead of the parser
                sub-crops, and no derived file is created.

        Returns:
            stored_points (List[PointStruct]): List of points stored in the Cat's declarative memory
                (text chunks and, if any, image points).

        See Also:
            before_rabbithole_stores_documents
            after_rabbithole_stored_documents

        Notes
        -------
        At this point, it is possible to customize the Cat's behavior using the `before_rabbithole_stores_documents`
        hook to edit the memories before they are inserted in the vector database.
        The hook `after_rabbithole_stored_documents` could be used to track the end of the process, indeed.
        """
        log.info(f"Agent id: {self.cat.agent_key}. Preparing to memorize {len(docs)} vectors for {source}.")

        embedder = await self.cat.lizard.embedder()
        plugin_manager = self.cat.plugin_manager

        # add custom metadata (sent via endpoint) and default metadata (source and when and eventual chat_id)
        for doc in docs:
            # Drop the transient parser image payload, if any: images are embedded and
            # saved as files separately, so their content must never reach the vector
            # DB metadata (and from there the LLM context on recall). This also covers
            # direct callers that pass documents still carrying image_base64.
            doc.metadata.pop("image_base64", None)
            doc.metadata = (
                    doc.metadata
                    | metadata
                    | {"source": source, "when": time.time(), "hash": file_hash}
                    | ({"chat_id": self.stray.id} if self.stray else {})
            )

# hook the docs before they are stored in the vector memory
        docs = await plugin_manager.execute_hook("before_rabbithole_stores_documents", docs, caller=self.stray or self.cat)

        # Store-time sizing guard for documents added by post-chunking hooks
        # (e.g. CAT_ALOG's catalogue card). These are split into budget-compliant
        # in-place sub-chunks (metadata inherited, ordering preserved), so the
        # full content is always stored with no loss. Atomic plugin units that
        # wrap their full text into metadata (e.g. CAT_ALOG's full_card) keep the
        # whole unit retrievable from any of their sub-chunks.
        docs = self._split_oversized(docs, embedder)

        # hook the points before they are stored in the vector memory
        valid_documents = list(filter(lambda doc_: doc_.page_content.strip(), docs))
        storing_vectors = await asyncio.to_thread(
            lambda: embedder.embed_documents([doc_.page_content for doc_ in valid_documents])
        )
        points = [PointStruct(
            id=uuid.uuid4().hex,
            payload=doc.model_dump(),
            vector=vector,
        ) for doc, vector in zip(valid_documents, storing_vectors)]

        # If a multimodal embedder is active and the multimodal parsers extracted some images,
        # embed them and add them to the same collection as the text chunks.
        if images and await self._is_multimodal_embedder():
            chat_id = self.stray.id if self.stray else None
            is_image_source = (mimetypes.guess_type(source)[0] or "").startswith("image/")

            if is_image_source:
                # Uploaded image files: the parser (hi_res) can split the file into
                # sub-crops. Embed the source file itself as a single whole-image
                # point (image_file = the source, no derived file) and ignore crops.
                whole_image = source_bytes if source_bytes is not None else (images[0]["image_bytes"] if images else None)
                embeds = await asyncio.to_thread(lambda: embedder.embed_images([whole_image])) if whole_image is not None else []
                files_and_vectors = [(source, embeds[0])] if embeds else []
            else:
                image_vectors = await asyncio.to_thread(
                    lambda: embedder.embed_images([img["image_bytes"] for img in images])
                )
                files_and_vectors = []
                for idx, (img, vector) in enumerate(zip(images, image_vectors)):
                    image_file = self._image_file_name(source, idx, img["image_mime_type"], img["image_bytes"])
                    await self.cat.save_file(img["image_bytes"], img["image_mime_type"], image_file, chat_id)
                    files_and_vectors.append((image_file, vector))

            image_points = [
                PointStruct(
                    id=uuid.uuid4().hex,
                    vector=vector,
                    payload={
                        "page_content": f"[Image] {source}",
                        "metadata": {
                            **(metadata or {}),
                            "source": source,
                            "when": time.time(),
                            "hash": file_hash,
                            "image": True,
                            "image_file": image_file,
                            **({"chat_id": self.stray.id} if self.stray else {}),
                        },
                    },
                )
                for image_file, vector in files_and_vectors
            ]
            points.extend(image_points)

        collection_name = str(VectorMemoryType.DECLARATIVE if not self.stray else VectorMemoryType.EPISODIC)
        await self.cat.vector_memory_handler.add_points_to_tenant(collection_name=collection_name, points=points)

        return points

    def _image_file_name(self, source: str, index: int, mime_type: str, image_bytes: bytes) -> str:
        """Deterministic, unique file name for an extracted image.

        The stem comes from the source file name so the association between an
        image and the document it was extracted from is recoverable by name.
        """
        stem = os.path.splitext(os.path.basename(source))[0]
        stem = re.sub(r"[^a-zA-Z0-9._-]", "_", stem).strip("._") or "image"
        ext = mimetypes.guess_extension(mime_type or "") or ".png"
        digest = hashlib.sha256(image_bytes).hexdigest()[:8]
        return f"{stem}_img_{index}_{digest}{ext}"

    async def _split_text(self, docs: List[Document]):
        """Split LangChain documents in chunks.

        This method splits the incoming documents in chunks. Other two hooks are available to edit the
        documents before and after the split step.

        Args:
            docs (List[Document]): Content of the loaded file.

        Returns:
            docs (List[Document]): List of split Langchain `Document`.

        See Also:
            before_rabbithole_splits_documents

        Notes
        -----
        The default behavior splits the content and executes the hooks, before the splitting.
        `before_rabbithole_splits_documents` hook returns the original input without any modification.
        """
        plugin_manager = self.cat.plugin_manager

        # do something on the docs before they are split
        docs = await plugin_manager.execute_hook("before_rabbithole_splits_documents", docs, caller=self.stray or self.cat)

        # split docs
        docs = await self.cat.chunker.split_documents(docs)

        # join each short chunk with previous one, instead of deleting them
        try:
            docs = self._merge_short_chunks(docs, self.cat.chunker)
        except Exception as e:
            # Log error but don't fail the entire process
            log.warning(f"Error merging short chunks: {e}. Proceeding with original chunks.")

        # Finalize the document list so no chunk exceeds the active embedder's
        # max_input_tokens. This is a build-phase step, fully separate from the
        # embedding loop in store_documents: it produces the correct final list
        # (splitting oversized chunks into in-place sub-chunks) so the later
        # 1:1 embed/store pairing is never broken by re-chunking mid-loop.
        try:
            embedder = await self.cat.lizard.embedder()
            docs = self._split_oversized(docs, embedder)
        except Exception as e:
            # Log error but don't fail the entire process
            log.warning(f"Failed to finalize oversized chunks: {e}. Proceeding with original chunks.")

        return docs

    def _split_oversized(self, docs: List[Document], embedder) -> List[Document]:
        """Finalize a document list so no chunk exceeds the embedder's input limit.

        This is a *pure fold* over ``docs``: it reads the input list and returns a
        brand-new list, never mutating ``docs`` while scanning it. Any oversized
        chunk is split into budget-compliant sub-chunks that replace it at its own
        index, so the relative order of all chunks is preserved and each sub-chunk
        becomes its own stored point (with its own payload) in the vector database.

        Args:
            docs: The chunked documents produced by the configured chunker.
            embedder: The active embedder, whose ``max_input_tokens`` ceiling is
                enforced. ``None`` or a missing limit disables the split.

        Returns:
            A new list of documents where every chunk is at or under the limit.
        """
        max_tokens = getattr(embedder, "max_input_tokens", None)
        if max_tokens is None or max_tokens <= 0:
            return docs

        sized: List[Document] = []
        for doc in docs:
            token_count = self._doc_tokens(doc, embedder)
            if token_count <= max_tokens:
                sized.append(doc)
                continue
            sub_chunks = self._split_to_budget(doc, embedder)
            source = doc.metadata.get("source") if isinstance(doc.metadata, dict) else None
            log.debug(
                f"OVERSIZED_SPLIT src={source} estimated_tokens={token_count} "
                f"max_input_tokens={max_tokens} split_into={len(sub_chunks)}"
            )
            sized.extend(sub_chunks)
        return sized

    def _doc_tokens(self, doc: Document, embedder) -> int:
        """Conservative token count for a document chunk.

        Prefers the active embedder's own ``_estimate_tokens`` when available so
        the count matches the model that will embed the chunk; otherwise falls
        back to a ~3 chars/token heuristic (never an undercount).
        """
        if embedder is not None and hasattr(embedder, "_estimate_tokens"):
            return embedder._estimate_tokens(doc.page_content)
        return max(1, len(doc.page_content) // 3)

    def _split_to_budget(self, doc: Document, embedder) -> List[Document]:
        """Split one oversized document into budget-compliant sub-chunks.

        Word-based, linear split around the token budget, carrying the original
        metadata (source/payload) forward to every sub-chunk so nothing stored
        in the vector DB is lost.

        Sub-chunks are sized by MEASURING candidates with the embedder's own
        ``_estimate_tokens`` (real tokenizer when available); a coarse
        estimate-then-refine pass keeps it linear even for very long documents.
        """
        max_tokens = getattr(embedder, "max_input_tokens", None)
        if max_tokens is None or max_tokens <= 0:
            return [doc]

        words = doc.page_content.split()
        if not words:
            return [doc]

        def doc_tokens(text: str) -> int:
            if embedder is not None and hasattr(embedder, "_estimate_tokens"):
                return max(1, embedder._estimate_tokens(text))
            return max(1, len(text) // 3)

        # avg tokens per word for THIS document (sampled on 200 words), used
        # to guess chunk boundaries in the coarse pass
        sample = words[:200]
        sample_tokens = max(1, doc_tokens(" ".join(sample)))
        approx_tpw = sample_tokens / max(1, len(sample))
        per_chunk_words = max(1, int(max_tokens / approx_tpw))

        sub_docs: List[Document] = []
        start = 0
        n = len(words)
        while start < n:
            part = words[start:start + per_chunk_words]
            if doc_tokens(" ".join(part)) <= max_tokens:
                take = len(part)
            else:
                # refine: binary-search the largest prefix within budget
                lo, hi = 0, len(part)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if doc_tokens(" ".join(part[:mid])) <= max_tokens:
                        lo = mid
                    else:
                        hi = mid - 1
                take = max(1, lo)  # pathological single word: still emit it
            sub_docs.append(Document(
                page_content=" ".join(words[start:start + take]),
                metadata=doc.metadata,
            ))
            start += take
        return sub_docs

    def _merge_short_chunks(self, docs: List[Document], chunker: BaseChunker) -> List[Document]:
        """Safely merge short chunks with adjacent ones.

        Args:
            docs: List of documents to process
            chunker: The chunker instance for configuration

        Returns:
            List of documents with short chunks merged
        """
        def should_merge_chunk() -> bool:
            """Determine if a chunk should be merged."""
            return (
                    min_chunk_size > len(current_content) > 0 and  # Don't merge empty content
                    len(merged_docs) > 0  # Need previous chunk to merge with
            )

        def can_safely_merge(prev_doc: Document) -> bool:
            """Check if two documents can be safely merged."""
            potential_size = len(prev_doc.page_content) + len(current_doc.page_content) + 2
            return potential_size <= max_merge_size

        if not docs:
            return docs

        # Get configuration with safe defaults
        chunk_size = getattr(chunker.splitter, "chunk_size", getattr(chunker.splitter, "max_chunk_size", 1000))
        chunk_overlap = getattr(chunker.splitter, "chunk_overlap", 100)

        # Conservative thresholds
        min_chunk_size = max(50, chunk_size // 20)  # At least 50 chars
        max_merge_size = chunk_size + chunk_overlap  # Respect splitter's intended size

        merged_docs: list = []  # type: ignore[var-annotated]
        i = 0

        while i < len(docs):
            current_doc = docs[i]
            current_content = current_doc.page_content.strip()

            # Check if this chunk should be merged
            if should_merge_chunk() and can_safely_merge(merged_docs[-1]):
                try:
                    merged_docs[-1] = self._create_merged_document(merged_docs[-1], current_doc)
                except Exception:
                    # If merge fails, keep both documents separate
                    merged_docs.append(current_doc)
            else:
                merged_docs.append(current_doc)

            i += 1

        return merged_docs

    def _create_merged_document(self, prev_doc: Document, current_doc: Document) -> Document:
        """Create a new merged document safely."""
        # Merge content with clear separator
        merged_content = prev_doc.page_content.rstrip() + "\n\n" + current_doc.page_content.lstrip()

        # Merge metadata - since source is the same, we can safely combine
        merged_metadata = prev_doc.metadata.copy()

        # Add all metadata from current doc, handling conflicts intelligently
        for key, value in current_doc.metadata.items():
            if key in merged_metadata and merged_metadata[key] != value:
                # For numeric values (like page numbers), take the range or sum
                if isinstance(merged_metadata[key], (int, float)) and isinstance(value, (int, float)):
                    if key in ["page", "page_number", "chunk_index"]:
                        # For page/chunk numbers, keep the starting one
                        pass  # Keep the previous value
                    else:
                        # For other numeric values, might want to sum or take max
                        merged_metadata[key] = max(merged_metadata[key], value)
                else:
                    # For other conflicts, keep the first value
                    pass
            else:
                merged_metadata[key] = value

        # Add merge tracking
        merge_count = merged_metadata.get("_merge_count", 1) + 1
        merged_metadata["_merge_count"] = merge_count
        merged_metadata["_is_merged"] = True

        return Document(page_content=merged_content, metadata=merged_metadata)

    async def _send_notification_message(self, message: str):
        if self.stray and self.stray.notifier.has_ws_connection():
            await self.stray.notifier.send_notification(message)
