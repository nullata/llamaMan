# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Auto-restart-on-crash loop guard. The relaunch itself needs Docker, but the
bounding logic (at most N relaunches per rolling window) is what keeps a model
that crashes on every launch from hot-looping Docker forever, so it's worth
pinning down."""

import os
import threading
import time
import unittest
from unittest.mock import patch

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import core.monitoring as mon


class AutoRestartGuardTests(unittest.TestCase):
    def setUp(self):
        with mon._auto_restart_lock:
            mon._auto_restart_log.clear()

    def tearDown(self):
        with mon._auto_restart_lock:
            mon._auto_restart_log.clear()

    def _patched_relaunch(self):
        calls = []
        fired = threading.Semaphore(0)

        def fake(inst_id):
            calls.append(inst_id)
            fired.release()
            return True

        return calls, fired, fake

    def test_crash_loop_is_capped(self):
        calls, fired, fake = self._patched_relaunch()
        with patch("api.instances.relaunch_inactive_instance", side_effect=fake):
            for _ in range(mon._AUTO_RESTART_MAX + 2):
                mon._maybe_auto_restart("inst-x")
            # exactly the cap many relaunch threads should fire
            for _ in range(mon._AUTO_RESTART_MAX):
                self.assertTrue(fired.acquire(timeout=2))
            time.sleep(0.1)  # any (wrongly) extra thread would land here
        self.assertEqual(len(calls), mon._AUTO_RESTART_MAX)

    def test_old_attempts_age_out_of_window(self):
        # MAX crashes long ago (outside the window) must not block a fresh restart.
        old = time.time() - mon._AUTO_RESTART_WINDOW_S - 1
        with mon._auto_restart_lock:
            mon._auto_restart_log["inst-y"] = [old] * mon._AUTO_RESTART_MAX

        calls, fired, fake = self._patched_relaunch()
        with patch("api.instances.relaunch_inactive_instance", side_effect=fake):
            mon._maybe_auto_restart("inst-y")
            self.assertTrue(fired.acquire(timeout=2))
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
