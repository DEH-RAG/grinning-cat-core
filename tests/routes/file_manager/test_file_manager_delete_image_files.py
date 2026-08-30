import os
import types

from cat.routes.file_manager import delete_file


async def test_delete_file_removes_source_image_files(monkeypatch):
    """Deleting a source file must also cascade-delete its extracted image files.

    The image points carry ``metadata.image_file``; they must be queried and their
    files removed from the agent storage before the memory points are deleted.
    """
    removed_files = []
    deleted_points = []

    class FakeVectorMemoryHandler:
        def __init__(self):
            self.queries = []

        async def get_all_tenant_points(self, collection_name, limit, offset, metadata):
            self.queries.append((collection_name, limit, offset, metadata))
            points = [
                types.SimpleNamespace(payload={
                    "metadata": {"source": "test.txt", "image": True, "image_file": "test_img_0_abcd1234.jpg"},
                }),
                # a point without an image_file must not trigger any file removal
                types.SimpleNamespace(payload={"metadata": {"source": "test.txt"}}),
            ]
            return points, None

        async def delete_tenant_points(self, collection_name, metadata):
            deleted_points.append((collection_name, metadata))

    handler = FakeVectorMemoryHandler()

    class FakeFileManager:
        def remove_file(self, file_path):
            removed_files.append(file_path)
            return True

    class FakePluginManager:
        async def execute_hook(self, name, *args, caller=None):
            return None

    fake_cheshire_cat = types.SimpleNamespace(
        agent_key="agent",
        vector_memory_handler=handler,
        file_manager=FakeFileManager(),
        plugin_manager=FakePluginManager(),
    )
    fake_info = types.SimpleNamespace(cheshire_cat=fake_cheshire_cat, stray_cat=None)

    # fixed (path, collection_id, metadata) so no real storage/DB is touched
    monkeypatch.setattr(
        "cat.routes.file_manager.get_from_info",
        lambda info: ("path", "declarative", {}),
    )

    res = await delete_file("test.txt", info=fake_info)

    assert res.deleted is True
    # the source file and the extracted image file are both removed, in order
    assert removed_files == [
        os.path.join("path", "test.txt"),
        os.path.join("path", "test_img_0_abcd1234.jpg"),
    ]
    # image points are queried by source + image flag before the points are deleted
    assert handler.queries == [("declarative", 100, None, {"source": "test.txt", "image": True})]
    assert deleted_points == [("declarative", {"source": "test.txt"})]
