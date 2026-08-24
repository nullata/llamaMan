# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

# Regression: waking a sleeping instance for its own model must not be blocked
# by LLAMAMAN_MAX_MODELS - that slot was already claimed at launch and sleep
# only pauses the container. The original bug: with 2 sleeping models and
# MAX=2, an OpenAI-compat request for one of them returned 503 "model limit
# reached" instead of waking it, because the cap check ran before the code
# realized the request would just wake an existing instance rather than
# create a new one.
#
# Sleeping DOES still count against the cap - so an API request for a
# DIFFERENT model, when the cap is full of admin-launched sleepers, still
# gets 503 unless the "Allow API to evict admin-launched" toggle is on. This
# preserves the admin's slot claim across sleep/wake cycles.

import os
import unittest
from unittest.mock import Mock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))

import api.instances as instances_api
import api.llamaman as llamaman
from core.state import instances, instances_lock


def _mk_inst(inst_id, model_path, status, port, ts, managed=True, embedding=False):
    return {
        "id": inst_id,
        "model_name": os.path.basename(model_path),
        "model_path": model_path,
        "port": port,
        "status": status,
        "started_at": ts,
        "_last_request_at": ts,
        "_llamaman_managed": managed,
        "config": {"embedding_model": embedding},
    }


class SleepingStillCountsTests(unittest.TestCase):
    """Sleeping still holds a slot - the admin's launch claim persists across
    the container's idle-timeout pause."""

    def setUp(self):
        with instances_lock:
            self._saved = {k: dict(v) for k, v in instances.items()}
            instances.clear()

    def tearDown(self):
        with instances_lock:
            instances.clear()
            instances.update(self._saved)

    def test_llamaman_count_includes_sleeping(self):
        with instances_lock:
            instances["a"] = _mk_inst("a", "/models/a.gguf", "sleeping", 8000, 100)
            instances["b"] = _mk_inst("b", "/models/b.gguf", "sleeping", 8001, 101)
        self.assertEqual(llamaman._count_running_instances(), 2)

    def test_llamaman_count_excludes_only_stopped(self):
        with instances_lock:
            instances["s"] = _mk_inst("s", "/models/s.gguf", "sleeping", 8000, 100)
            instances["h"] = _mk_inst("h", "/models/h.gguf", "healthy", 8001, 101)
            instances["g"] = _mk_inst("g", "/models/g.gguf", "starting", 8002, 102)
            instances["x"] = _mk_inst("x", "/models/x.gguf", "stopped", 8003, 103)
        self.assertEqual(llamaman._count_running_instances(), 3)

    def test_instances_count_includes_sleeping(self):
        with instances_lock:
            instances["s"] = _mk_inst("s", "/models/s.gguf", "sleeping", 8000, 100)
            instances["h"] = _mk_inst("h", "/models/h.gguf", "healthy", 8001, 101)
        self.assertEqual(instances_api._count_running_chat_instances(), 2)


class OpenAIWakeExistingBypassesCapTests(unittest.TestCase):
    """The originally reported scenario: 2 sleeping models with MAX=2, an
    OpenAI-compat request for one of them must wake it - the cap check
    should not fire on wake-existing-instance."""

    def setUp(self):
        with instances_lock:
            self._saved = {k: dict(v) for k, v in instances.items()}
            instances.clear()

    def tearDown(self):
        with instances_lock:
            instances.clear()
            instances.update(self._saved)

    def test_openai_wakes_own_sleeping_instance_even_when_cap_full_of_sleepers(self):
        with instances_lock:
            instances["a"] = _mk_inst("a", "/models/a.gguf", "sleeping", 8000, 100)
            instances["b"] = _mk_inst("b", "/models/b.gguf", "sleeping", 8001, 101)

        storage = Mock()
        storage.get_preset.return_value = {"embedding_model": False}

        # allow_eviction=False is the OpenAI path with the override toggle off.
        with patch("api.llamaman.LLAMAMAN_MAX_MODELS", 2), \
             patch("api.llamaman.get_storage", return_value=storage), \
             patch("api.llamaman._find_model_by_name", return_value={"path": "/models/a.gguf"}), \
             patch("api.llamaman._find_running_instance_by_alias", return_value=None), \
             patch("core.model_alias.resolve_to_path", return_value="/models/a.gguf"), \
             patch("api.instances.relaunch_inactive_instance", return_value=True) as wake_mock:
            inst, err = llamaman._ensure_model_running("a", allow_eviction=False)

        self.assertIsNone(err, f"expected wake, got 503: {err}")
        self.assertEqual(inst["id"], "a")
        wake_mock.assert_called_once_with("a")

    def test_openai_wakes_sleeping_even_when_cap_full_of_healthy_running(self):
        """Sibling case: instance A is sleeping, B is healthy, cap=2. Request
        for A should still wake without needing to evict B - A already has a
        slot, waking uses that slot."""
        with instances_lock:
            instances["a"] = _mk_inst("a", "/models/a.gguf", "sleeping", 8000, 100)
            instances["b"] = _mk_inst("b", "/models/b.gguf", "healthy", 8001, 101)

        storage = Mock()
        storage.get_preset.return_value = {"embedding_model": False}

        with patch("api.llamaman.LLAMAMAN_MAX_MODELS", 2), \
             patch("api.llamaman.get_storage", return_value=storage), \
             patch("api.llamaman._find_model_by_name", return_value={"path": "/models/a.gguf"}), \
             patch("api.llamaman._find_running_instance_by_alias", return_value=None), \
             patch("core.model_alias.resolve_to_path", return_value="/models/a.gguf"), \
             patch("api.instances.relaunch_inactive_instance", return_value=True) as wake_mock:
            inst, err = llamaman._ensure_model_running("a", allow_eviction=False)

        self.assertIsNone(err)
        self.assertEqual(inst["id"], "a")
        wake_mock.assert_called_once_with("a")


class AdminSleeperProtectionTests(unittest.TestCase):
    """Sleeping still counts, so admin-launched sleepers are still protected
    from API-driven displacement without the override toggle."""

    def setUp(self):
        with instances_lock:
            self._saved = {k: dict(v) for k, v in instances.items()}
            instances.clear()

    def tearDown(self):
        with instances_lock:
            instances.clear()
            instances.update(self._saved)

    def test_openai_new_model_503s_when_cap_saturated_by_admin_sleepers(self):
        """The exact scenario the toggle is designed to protect: admin
        launched A and B via the UI, both went to sleep, and an API request
        arrives for a DIFFERENT model C. Without the override toggle, the
        API must not displace the admin's sleeping instances."""
        with instances_lock:
            instances["a"] = _mk_inst("a", "/models/a.gguf", "sleeping", 8000, 100, managed=False)
            instances["b"] = _mk_inst("b", "/models/b.gguf", "sleeping", 8001, 101, managed=False)

        storage = Mock()
        storage.get_preset.return_value = {"embedding_model": False}

        with patch("api.llamaman.LLAMAMAN_MAX_MODELS", 2), \
             patch("api.llamaman.get_storage", return_value=storage), \
             patch("api.llamaman._find_model_by_name", return_value={"path": "/models/c.gguf"}), \
             patch("api.llamaman._find_running_instance_by_alias", return_value=None), \
             patch("core.model_alias.resolve_to_path", return_value="/models/c.gguf"):
            inst, err = llamaman._ensure_model_running(
                "c", allow_eviction=False, can_evict_admin=False
            )

        self.assertIsNone(inst)
        self.assertIsNotNone(err)
        self.assertIn("model limit reached", err)


if __name__ == "__main__":
    unittest.main()
