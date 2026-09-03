# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Guards on the async-stop path added in the peer-responsiveness change.

The DELETE /api/instances/<id> route used to be synchronous: it blocked the
request thread through docker's SIGTERM grace (up to ~10s per stop), which on
a cross-node click meant the initiating node's UI card sat unchanged for that
long plus another 5-10s until the peer's heartbeat republished. The route now
schedules a background worker that flips status "healthy|starting|sleeping"
-> "stopping" -> "stopped" and returns 202 immediately.

These tests pin down the invariants that keep that safe:
  - the sync stop_instance_by_id is still there for eviction (which MUST have
    the container gone before launching on the same GPU)
  - the async path publishes a heartbeat at BOTH transitions (so peers observe
    the transient state and the terminal state without waiting for the 5s tick)
  - a double-click / concurrent second call is a no-op that does not spawn a
    second worker or double-stop the container
  - a call for an unknown id returns False, matching the sync path's contract
"""

import os
import threading
import unittest
from unittest.mock import patch

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import api.instances as instances_api
from core.state import instances, instances_lock


def _fresh_inst(inst_id="i1", status="healthy", container_id="cid-abc"):
    return {
        "id": inst_id,
        "model_name": "chat.gguf",
        "model_path": "/models/chat.gguf",
        "port": 8000,
        "status": status,
        "container_id": container_id,
        "container_name": "llamaman-" + inst_id,
        "log_file": "",
        "config": {},
        "started_at": 0,
        "_last_request_at": 0,
        "stats": {},
    }


class _AsyncStopBase(unittest.TestCase):
    def setUp(self):
        with instances_lock:
            self._saved = {k: dict(v) for k, v in instances.items()}
            instances.clear()

    def tearDown(self):
        with instances_lock:
            instances.clear()
            instances.update(self._saved)


class StopInstanceAsyncTests(_AsyncStopBase):
    def test_unknown_id_returns_false(self):
        # The sync stop_instance_by_id returned False for an unknown id; the
        # DELETE route relied on that to distinguish 404 from success. The
        # async wrapper must preserve that signal.
        self.assertFalse(instances_api.stop_instance_async("does-not-exist"))

    def test_transitions_to_stopping_then_stopped(self):
        with instances_lock:
            instances["i1"] = _fresh_inst("i1", status="healthy")

        # The docker call runs on a background thread; block it via a gate so
        # we can observe the transient "stopping" state before it flips.
        release = threading.Event()
        seen_container_id = []

        def _fake_stop(container_id, timeout=10):
            seen_container_id.append(container_id)
            release.wait(timeout=2)

        with patch("api.instances.stop_container", side_effect=_fake_stop), \
             patch("api.instances._publish_cluster_heartbeat_safe"), \
             patch("api.instances.save_state"), \
             patch("api.instances.release_instance_reservations"):
            self.assertTrue(instances_api.stop_instance_async("i1"))

            # After the scheduling call returns, status must already be the
            # transient "stopping" - that is the whole point of the async path
            # (peers get to observe it via the piggyback heartbeat before the
            # docker grace elapses).
            with instances_lock:
                self.assertEqual(instances["i1"]["status"], "stopping")

            # Let the background worker finish and observe the terminal state.
            release.set()
            for _ in range(200):
                with instances_lock:
                    if instances["i1"]["status"] == "stopped":
                        break
                threading.Event().wait(0.01)
            with instances_lock:
                self.assertEqual(instances["i1"]["status"], "stopped")
                self.assertIsNone(instances["i1"]["container_id"])

        self.assertEqual(seen_container_id, ["cid-abc"])

    def test_publishes_heartbeat_at_both_transitions(self):
        # The whole responsiveness win depends on peers seeing "stopping" and
        # "stopped" without waiting for the 5s heartbeat tick. Both must fire.
        with instances_lock:
            instances["i1"] = _fresh_inst("i1", status="healthy")

        with patch("api.instances.stop_container"), \
             patch("api.instances.save_state"), \
             patch("api.instances.release_instance_reservations"), \
             patch("api.instances._publish_cluster_heartbeat_safe") as hb:
            self.assertTrue(instances_api.stop_instance_async("i1"))
            # Wait for the background worker to land.
            for _ in range(200):
                with instances_lock:
                    if instances["i1"]["status"] == "stopped":
                        break
                threading.Event().wait(0.01)

            # One for the "stopping" transition (before scheduling docker),
            # one for the "stopped" transition (after docker finishes).
            self.assertGreaterEqual(hb.call_count, 2)

    def test_double_call_while_stopping_is_a_noop(self):
        # A double-click Stop, or a stale request from a second UI, must not
        # spawn a second worker or re-run docker stop on the same container.
        with instances_lock:
            instances["i1"] = _fresh_inst("i1", status="stopping",
                                          container_id="cid-abc")

        with patch("api.instances.stop_container") as stop_mock, \
             patch("api.instances._publish_cluster_heartbeat_safe"), \
             patch("api.instances.save_state"), \
             patch("api.instances.release_instance_reservations"):
            self.assertTrue(instances_api.stop_instance_async("i1"))
            stop_mock.assert_not_called()

    def test_no_container_finishes_inline(self):
        # An instance record without a container_id (e.g. sleeping proxy with
        # container already gone, or a launch that failed before the container
        # attached) still needs to reach the terminal "stopped" state.
        with instances_lock:
            instances["i1"] = _fresh_inst("i1", status="sleeping",
                                          container_id=None)

        with patch("api.instances.stop_container") as stop_mock, \
             patch("api.instances._publish_cluster_heartbeat_safe"), \
             patch("api.instances.save_state"), \
             patch("api.instances.release_instance_reservations"):
            self.assertTrue(instances_api.stop_instance_async("i1"))
            # No container -> no docker call, transition happened inline.
            stop_mock.assert_not_called()
            with instances_lock:
                self.assertEqual(instances["i1"]["status"], "stopped")
                self.assertIsNone(instances["i1"]["container_id"])


class SyncStopStillWorksTests(_AsyncStopBase):
    """The sync stop_instance_by_id is still the eviction path - if it broke,
    a launch that had to evict a peer to free VRAM would race the freeing."""

    def test_sync_stops_container_before_returning(self):
        with instances_lock:
            instances["i1"] = _fresh_inst("i1", status="healthy")

        stop_order = []

        def _stop(cid, timeout=10):
            stop_order.append(("docker", cid))

        with patch("api.instances.stop_container", side_effect=_stop), \
             patch("api.instances.save_state",
                   side_effect=lambda: stop_order.append(("save",))), \
             patch("api.instances._publish_cluster_heartbeat_safe"), \
             patch("api.instances.release_instance_reservations",
                   side_effect=lambda _i: stop_order.append(("release",))):
            self.assertTrue(instances_api.stop_instance_by_id("i1"))

        # The eviction contract: by the time we return, the container is gone,
        # its reservations are released, and state is persisted.
        self.assertEqual([step[0] for step in stop_order],
                         ["docker", "release", "save"])
        with instances_lock:
            self.assertEqual(instances["i1"]["status"], "stopped")
            self.assertIsNone(instances["i1"]["container_id"])

    def test_sync_unknown_id_returns_false(self):
        self.assertFalse(instances_api.stop_instance_by_id("nope"))


if __name__ == "__main__":
    unittest.main()
