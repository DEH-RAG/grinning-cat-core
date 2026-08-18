import base64
import types

from langchain_core.documents import Document

from cat.db.database import DEFAULT_SYSTEM_KEY
from cat.rabbit_hole import RabbitHole
from cat.services.factory.embedder import MultimodalEmbeddings
from cat.services.memory.models import VectorMemoryType

from tests.utils import agent_id


class FakeMultimodalEmbedder(MultimodalEmbeddings):
    """Minimal multimodal embedder used to exercise the image-ingestion branch."""

    def embed_documents(self, texts):
        return [[0.1] * 4 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 4

    def embed_image(self, image):
        return [0.1] * 4

    def embed_images(self, images):
        return [[0.1] * 4 for _ in images]



async def _multimodal(self):  # noqa: ANN001
    return True


async def _text_only(self):  # noqa: ANN001
    return False


def _image_payload(data: bytes, mime: str = "image/jpeg") -> dict:
    return {
        "image_base64": base64.b64encode(data).decode(),
        "image_bytes": data,
        "image_mime_type": mime,
    }


def test_collect_document_images():
    rabbit_hole = RabbitHole()

    docs = [
        Document(page_content="plain text, no images"),
        Document(page_content="image element", metadata={"image_base64": "aGVsbG8=", "image_mime_type": "image/png"}),
    ]

    images = rabbit_hole._collect_document_images(docs)

    assert len(images) == 1
    assert images[0]["image_bytes"] == b"hello"
    assert images[0]["image_mime_type"] == "image/png"


def test_collect_document_images_defaults_mime():
    rabbit_hole = RabbitHole()

    docs = [Document(page_content="image", metadata={"image_base64": "aGVsbG8="})]
    images = rabbit_hole._collect_document_images(docs)

    assert images[0]["image_mime_type"] == "image/jpeg"


async def test_store_documents_multimodal_embeds_and_stores_images(cheshire_cat, monkeypatch):
    stored: dict = {}

    async def fake_add_points(collection_name, points):
        stored["collection"] = collection_name
        stored["points"] = points

    async def fake_embedder():
        return FakeMultimodalEmbedder()

    # Force multimodal detection and swap the embedder + the vector memory storage.
    monkeypatch.setattr(RabbitHole, "_is_multimodal_embedder", _multimodal)
    monkeypatch.setattr(cheshire_cat.lizard, "embedder", fake_embedder)
    monkeypatch.setattr(cheshire_cat.vector_memory_handler, "add_points_to_tenant", fake_add_points)

    # Record save_file calls instead of writing the image files to the real storage.
    saved_files: list = []

    async def fake_save_file(file_bytes, content_type, source, chat_id=None):
        saved_files.append((file_bytes, content_type, source, chat_id))

    monkeypatch.setattr(cheshire_cat, "save_file", fake_save_file)

    rabbit_hole = RabbitHole()
    rabbit_hole.cat = cheshire_cat
    rabbit_hole.stray = None

    docs = [Document(page_content="a text chunk", metadata={})]
    images = [_image_payload(b"\x89PNG\r\n\x1a\n")]

    points = await rabbit_hole.store_documents(
        docs=docs, source="test.txt", file_hash="hash", metadata={}, images=images,
    )

    assert stored["collection"] == str(VectorMemoryType.DECLARATIVE)
    # one text point + one image point
    assert len(points) == 2

    text_points = [p for p in points if not p.payload["metadata"].get("image")]
    image_points = [p for p in points if p.payload["metadata"].get("image")]

    assert len(text_points) == 1
    assert len(image_points) == 1

    image_metadata = image_points[0].payload["metadata"]
    assert image_metadata["image"] is True
    assert image_metadata["source"] == "test.txt"
    assert "image_base64" not in image_metadata
    assert "image_mime_type" not in image_metadata
    image_file = image_metadata["image_file"]
    assert image_file.startswith("test_img_0_")
    assert image_file.endswith(".jpg")

    # the image was saved as a file, not embedded in the point metadata
    assert saved_files == [(b"\x89PNG\r\n\x1a\n", "image/jpeg", image_file, None)]


async def test_store_documents_multimodal_uses_agent_embedder(cheshire_cat, monkeypatch):
    """The image points must come from the agent's own embedder (embed_images)."""
    stored: dict = {}

    async def fake_add_points(collection_name, points):
        stored["points"] = points

    calls: dict = {"images": []}

    class RecordingEmbedder(FakeMultimodalEmbedder):
        def embed_images(self, images):
            calls["images"] = list(images)
            return super().embed_images(images)

    async def fake_embedder():
        return RecordingEmbedder()

    monkeypatch.setattr(RabbitHole, "_is_multimodal_embedder", _multimodal)
    monkeypatch.setattr(cheshire_cat.lizard, "embedder", fake_embedder)
    monkeypatch.setattr(cheshire_cat.vector_memory_handler, "add_points_to_tenant", fake_add_points)

    # Record save_file calls instead of writing the image files to the real storage.
    saved_files: list = []

    async def fake_save_file(file_bytes, content_type, source, chat_id=None):
        saved_files.append((file_bytes, content_type, source, chat_id))

    monkeypatch.setattr(cheshire_cat, "save_file", fake_save_file)

    rabbit_hole = RabbitHole()
    rabbit_hole.cat = cheshire_cat
    rabbit_hole.stray = None

    docs = [Document(page_content="a text chunk")]

    await rabbit_hole.store_documents(docs=docs, source="test.txt", metadata={}, images=[_image_payload(b"IMG1")])

    # the raw image bytes are what gets embedded
    assert calls["images"] == [b"IMG1"]

    # the image was saved as a file in the agent storage
    assert len(saved_files) == 1
    assert saved_files[0][2].startswith("test_img_")


async def test_store_documents_multimodal_in_conversation_adds_chat_id(cheshire_cat, monkeypatch):
    """In a conversation (stray set), image points carry chat_id in their metadata
    and the image file is saved under the conversation id."""
    stored: dict = {}

    async def fake_add_points(collection_name, points):
        stored["collection"] = collection_name
        stored["points"] = points

    saved_files: list = []

    async def fake_save_file(file_bytes, content_type, source, chat_id=None):
        saved_files.append((file_bytes, content_type, source, chat_id))

    async def fake_embedder():
        return FakeMultimodalEmbedder()

    monkeypatch.setattr(RabbitHole, "_is_multimodal_embedder", _multimodal)
    monkeypatch.setattr(cheshire_cat.lizard, "embedder", fake_embedder)
    monkeypatch.setattr(cheshire_cat.vector_memory_handler, "add_points_to_tenant", fake_add_points)
    monkeypatch.setattr(cheshire_cat, "save_file", fake_save_file)

    rabbit_hole = RabbitHole()
    rabbit_hole.cat = cheshire_cat
    rabbit_hole.stray = types.SimpleNamespace(id="chat_abc")

    docs = [Document(page_content="a text chunk")]

    await rabbit_hole.store_documents(docs=docs, source="test.txt", metadata={}, images=[_image_payload(b"IMG1")])

    # image points land in the episodic collection and carry chat_id
    assert stored["collection"] == str(VectorMemoryType.EPISODIC)
    image_points = [p for p in stored["points"] if p.payload["metadata"].get("image")]
    assert len(image_points) == 1
    assert image_points[0].payload["metadata"]["chat_id"] == "chat_abc"
    # the saved image file is scoped to the conversation
    assert len(saved_files) == 1
    assert saved_files[0][3] == "chat_abc"


async def test_store_documents_multimodal_image_source_does_not_duplicate(cheshire_cat, monkeypatch):
    """Uploading an image file embeds the whole file (no derived files/points).

    The hi_res parser can split an image into sub-crops: those must be ignored and
    the source file itself embedded as a single image point (image_file = source).
    """
    stored: dict = {}
    saved_files: list = []

    async def fake_add_points(collection_name, points):
        stored["points"] = points

    async def fake_save_file(file_bytes, content_type, source, chat_id=None):
        saved_files.append((file_bytes, content_type, source, chat_id))

    async def fake_embedder():
        return FakeMultimodalEmbedder()

    monkeypatch.setattr(RabbitHole, "_is_multimodal_embedder", _multimodal)
    monkeypatch.setattr(cheshire_cat.lizard, "embedder", fake_embedder)
    monkeypatch.setattr(cheshire_cat.vector_memory_handler, "add_points_to_tenant", fake_add_points)
    monkeypatch.setattr(cheshire_cat, "save_file", fake_save_file)

    rabbit_hole = RabbitHole()
    rabbit_hole.cat = cheshire_cat
    rabbit_hole.stray = None

    docs = [Document(page_content="")]
    source_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 16  # fake jpeg payload
    # parser crops for the image source (must be ignored)
    crops = [_image_payload(b"crop1", mime="image/jpeg"), _image_payload(b"crop2", mime="image/jpeg")]

    points = await rabbit_hole.store_documents(
        docs=docs, source="photo.jpeg", metadata={}, images=crops, source_bytes=source_bytes,
    )

    image_points = [p for p in points if p.payload["metadata"].get("image")]
    # exactly ONE image point, no derived files
    assert len(image_points) == 1
    assert image_points[0].payload["metadata"]["image_file"] == "photo.jpeg"
    assert image_points[0].payload["metadata"]["source"] == "photo.jpeg"
    assert saved_files == []


async def test_store_documents_text_only_ignores_images(cheshire_cat, monkeypatch):
    """When the embedder is not multimodal, images are ignored and only text is stored."""
    stored: dict = {}

    async def fake_add_points(collection_name, points):
        stored["points"] = points

    # detection reports a text-only embedder
    monkeypatch.setattr(RabbitHole, "_is_multimodal_embedder", _text_only)
    monkeypatch.setattr(cheshire_cat.vector_memory_handler, "add_points_to_tenant", fake_add_points)

    rabbit_hole = RabbitHole()
    rabbit_hole.cat = cheshire_cat
    rabbit_hole.stray = None

    docs = [Document(page_content="a text chunk")]

    points = await rabbit_hole.store_documents(docs=docs, source="test.txt", metadata={}, images=[_image_payload(b"IMG1")])

    # only the text point is stored, the image is dropped
    assert len(points) == 1
    assert not points[0].payload["metadata"].get("image")


def test_agent_id_is_test_agent(cheshire_cat):
    # guard that tests run against the expected agent key
    assert cheshire_cat.agent_key == agent_id


async def test_is_multimodal_embedder_uses_lizard_context(cheshire_cat, monkeypatch):
    """The embedder factory must run in the lizard (system) plugin-manager context.

    The ``factory_allowed_embedders`` hooks are declared with a ``lizard`` parameter
    (core base_plugin and PLUS alike): ``MadHatter.context_execute_hook`` passes the
    caller under that keyword only when the executing plugin manager belongs to
    BillTheLizard. Using an agent plugin manager would pass ``cat`` and make the
    hooks raise ``TypeError: unexpected keyword argument 'cat'``.
    """
    captured = {}

    class FakeServiceFactory:
        def __init__(self, agent_key, hook_manager, **kwargs):
            captured["agent_key"] = agent_key
            captured["plugin_manager_agent_key"] = hook_manager.agent_key
            captured["kwargs"] = kwargs

        async def get_config_class_from_adapter(self, obj):
            return None

    monkeypatch.setattr("cat.rabbit_hole.ServiceFactory", FakeServiceFactory)

    rabbit_hole = RabbitHole()
    rabbit_hole.cat = cheshire_cat

    assert await rabbit_hole._is_multimodal_embedder() is False

    assert captured["agent_key"] == DEFAULT_SYSTEM_KEY
    assert captured["plugin_manager_agent_key"] == DEFAULT_SYSTEM_KEY
