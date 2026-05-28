import os
import unittest
from unittest.mock import patch

from flask import Flask

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import api.images as images_api
import api.instances as instances_api
from core.state import instances, instances_lock


class ImageDeleteGuardTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.register_blueprint(images_api.bp)
        self.client = self.app.test_client()
        with instances_lock:
            self._saved = {k: dict(v) for k, v in instances.items()}
            instances.clear()

    def tearDown(self):
        with instances_lock:
            instances.clear()
            instances.update(self._saved)

    def test_delete_blocked_when_running_instance_uses_image(self):
        with instances_lock:
            instances["i1"] = {"id": "i1", "port": 8000, "status": "healthy",
                               "config": {"image": "ghcr.io/x:tag"}}
        resp = self.client.delete("/api/images", json={"image": "ghcr.io/x:tag"})
        self.assertEqual(resp.status_code, 409)
        self.assertIn("in use", resp.get_json()["error"])

    def test_delete_allowed_when_only_stopped_instance_uses_image(self):
        # A stopped instance must not block deletion; we only reach the docker
        # client (mocked) once the in-use guard passes.
        with instances_lock:
            instances["i1"] = {"id": "i1", "port": 8000, "status": "stopped",
                               "config": {"image": "ghcr.io/x:tag"}}
        fake_client = type("C", (), {"images": type("I", (), {
            "remove": staticmethod(lambda *a, **k: None)})()})()
        with patch("core.helpers.get_docker_client", return_value=fake_client), \
             patch("api.images._read_docker_images", return_value={"images": []}), \
             patch("api.images._write_docker_images"):
            resp = self.client.delete("/api/images", json={"image": "ghcr.io/x:tag"})
        self.assertEqual(resp.status_code, 200)


class CreatePassesImageTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.register_blueprint(instances_api.bp)
        self.client = self.app.test_client()
        with instances_lock:
            self._saved = {k: dict(v) for k, v in instances.items()}
            instances.clear()

    def tearDown(self):
        with instances_lock:
            instances.clear()
            instances.update(self._saved)

    @patch("api.instances.launch_instance")
    def test_create_forwards_image(self, launch_mock):
        launch_mock.return_value = ({"id": "x"}, None)
        resp = self.client.post("/api/instances", json={
            "model_path": "/models/m.gguf", "port": 8000, "ctx_size": 4096,
            "image": "ghcr.io/x:tag",
        })
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(launch_mock.call_args.kwargs.get("image"), "ghcr.io/x:tag")


if __name__ == "__main__":
    unittest.main()
