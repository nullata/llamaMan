import os
import unittest
from unittest.mock import Mock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))

import api.llamaman as llamaman


class LlamamanEvictionTests(unittest.TestCase):
    @patch("api.llamaman.find_available_port", return_value=8001)
    @patch("api.llamaman._find_any_instance_for_model", return_value=None)
    @patch("api.llamaman._find_running_instance_for_model", return_value=None)
    @patch("api.llamaman._find_model_by_name", return_value={"path": "/models/embed.gguf"})
    def test_embedding_auto_launch_skips_evict_when_cap_reached(
        self,
        _find_model_mock,
        _find_running_mock,
        _find_any_mock,
        _find_port_mock,
    ):
        storage = Mock()
        storage.get_preset.return_value = {"embedding_model": True}

        with patch("api.llamaman.get_storage", return_value=storage), \
             patch("api.llamaman._evict_llamaman_instances_if_needed", return_value=True) as evict_mock, \
             patch("api.instances.launch_instance", return_value=({"id": "inst-embed"}, None)) as launch_mock, \
             patch.dict("api.llamaman.instances", {}, clear=True):
            inst, err = llamaman._ensure_model_running("embed")

        self.assertIsNone(err)
        self.assertEqual(inst["id"], "inst-embed")
        evict_mock.assert_called_once_with(incoming_embedding_model=True, can_evict_admin=None)
        self.assertTrue(launch_mock.called)
        self.assertTrue(launch_mock.call_args.kwargs["embedding_model"])

    @patch("api.llamaman.find_available_port", return_value=8002)
    @patch("api.llamaman._find_any_instance_for_model", return_value=None)
    @patch("api.llamaman._find_running_instance_for_model", return_value=None)
    @patch("api.llamaman._find_model_by_name", return_value={"path": "/models/chat.gguf"})
    def test_chat_auto_launch_still_uses_evict_path(
        self,
        _find_model_mock,
        _find_running_mock,
        _find_any_mock,
        _find_port_mock,
    ):
        storage = Mock()
        storage.get_preset.return_value = {"embedding_model": False}

        with patch("api.llamaman.get_storage", return_value=storage), \
             patch("api.llamaman._evict_llamaman_instances_if_needed", return_value=True) as evict_mock, \
             patch("api.instances.launch_instance", return_value=({"id": "inst-chat"}, None)) as launch_mock, \
             patch.dict("api.llamaman.instances", {}, clear=True):
            inst, err = llamaman._ensure_model_running("chat")

        self.assertIsNone(err)
        self.assertEqual(inst["id"], "inst-chat")
        evict_mock.assert_called_once_with(incoming_embedding_model=False, can_evict_admin=None)
        self.assertTrue(launch_mock.called)
        self.assertFalse(launch_mock.call_args.kwargs["embedding_model"])


if __name__ == "__main__":
    unittest.main()
