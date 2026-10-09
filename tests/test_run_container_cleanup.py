# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""A container whose start fails (mount error, full disk) is removed: docker's
containers.run creates it first, and a leftover holds its image so the image
can't be deleted."""

import os
import unittest
from unittest.mock import Mock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import docker

import api.instances as instances_api


class UnstartedContainerCleanupTests(unittest.TestCase):

    def _run(self, error):
        client = Mock()
        client.containers.run.side_effect = error
        leftover = Mock()
        client.containers.get.return_value = leftover
        with patch("api.instances.get_docker_client", return_value=client), \
             patch("api.instances.ensure_docker_network"), \
             patch("api.instances._start_log_relay"), \
             patch("api.instances.get_vendor", return_value="cpu"):
            container, err = instances_api._run_container(
                "inst1", "llamaman-test-8000", "/models/m.gguf", 8000,
                {"ctx_size": 4096}, os.path.join(REPO_ROOT, "test-logs", "x.log"))
        return container, err, client, leftover

    def test_failed_start_removes_the_created_container(self):
        _, err, client, leftover = self._run(
            docker.errors.APIError("500 Server Error: no space left on device"))
        self.assertIn("no space left on device", err)
        client.containers.get.assert_called_once_with("llamaman-test-8000")
        leftover.remove.assert_called_once_with(force=True)

    def test_missing_image_has_nothing_to_remove(self):
        _, err, client, _ = self._run(docker.errors.ImageNotFound("nope"))
        self.assertIsNotNone(err)
        client.containers.get.assert_not_called()

    def test_no_leftover_is_fine(self):
        client = Mock()
        client.containers.get.side_effect = docker.errors.NotFound("gone")
        with patch("api.instances.get_docker_client", return_value=client):
            instances_api._remove_unstarted_container("llamaman-test-8000")   # no raise


if __name__ == "__main__":
    unittest.main()
