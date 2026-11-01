# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""#111 PR3: dispatch ranks sleepers last and awake gates queue rather than 429,
so a sleeper would otherwise only see traffic after a whole queue overflowed.
A pass beside the heartbeat wakes one when its group stays saturated."""

import os
import threading
import unittest
from unittest.mock import Mock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import api.cluster as ca
import core.monitoring as mon
import proxy
from core.state import instances, instances_lock

SELF = ca.cl.get_node_id()
# Sorts before and after this node's id in the stampede election.
LOWER, HIGHER = "0-" + SELF, SELF + "-z"


def _saturated(queued=3, sleeping=0, healthy=1):
    return {"active": 2, "queued": queued, "free": 0, "max_concurrent": 2, "fallback": False,
            "healthy": healthy, "starting": 0, "sleeping": sleeping}


def _node(node_id):
    return {"node_id": node_id, "node_name": node_id, "advertise_url": f"http://{node_id}:5000"}


class _Case(unittest.TestCase):

    def setUp(self):
        with instances_lock:
            saved = dict(instances)
            instances.clear()
        self.gates = []
        self.settings = {}
        self.nodes = {}   # node_id -> that peer's load map
        self.storage = Mock()
        self.storage.get_settings.side_effect = lambda: self.settings
        self.storage.list_nodes.side_effect = lambda: [_node(n) for n in self.nodes]
        self.wake = Mock(return_value=True)
        mon._eager_wake_streaks = {}
        mon._eager_wake_refused.clear()
        mon._eager_wakes_in_flight.clear()

        def restore():
            for inst_id in self.gates:
                proxy.remove_gate(inst_id)
            with instances_lock:
                instances.clear()
                instances.update(saved)
        self.addCleanup(restore)
        for target, value in (
            ("api.cluster.get_storage", lambda: self.storage),
            ("storage.get_storage", lambda: self.storage),
            ("core.node_settings.get_storage", lambda: self.storage),
            ("api.cluster._is_online", lambda _n: True),
            ("api.cluster._peer_load_map", lambda node: self.nodes[node["node_id"]]),
            ("api.cluster.cl.is_cluster_enabled", lambda: True),
            ("api.instances.relaunch_inactive_instance", self.wake),
            ("core.gpu.query_gpus", Mock(return_value=None)),
        ):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def add(self, inst_id, status, busy=0, queued=0, group="g", fallback=False, path=None, shared=False):
        path = path or f"/models/{inst_id}.gguf"
        proxy.create_gate(inst_id, 2, 10, model_path=path, share_queue=shared)
        self.gates.append(inst_id)
        gate = proxy.get_gate(inst_id)
        for _ in range(busy):
            self.assertTrue(gate.acquire(timeout=1))
        gate.queued += queued
        self.addCleanup(lambda: [gate.release() for _ in range(busy)])
        with instances_lock:
            instances[inst_id] = {
                "id": inst_id, "status": status, "model_path": path,
                "config": {"share_queue": True, "share_queue_group": group,
                           "share_queue_fallback": fallback, "max_concurrent": 2},
            }

    def tick(self, times=1):
        for _ in range(times):
            mon._eager_wake_tick()
        for t in threading.enumerate():
            if t.name.startswith("eager-wake-"):
                t.join(5)

    def woken(self):
        return [c.args[0] for c in self.wake.call_args_list]


class DecisionTests(_Case):

    def test_a_saturated_peer_wakes_the_sleeper_on_the_idle_node(self):
        # The main case: this node's gate is idle, the saturated node has nothing to wake.
        self.add("asleep", "sleeping")
        self.nodes = {HIGHER: {"g": _saturated()}}
        self.tick(3)
        self.assertEqual(self.woken(), ["asleep"])

    def test_this_nodes_own_saturation_counts(self):
        self.add("up", "healthy", busy=2, queued=1)
        self.add("asleep", "sleeping")
        self.tick(3)
        self.assertEqual(self.woken(), ["asleep"])

    def test_a_full_member_with_nothing_queued_is_not_pressure(self):
        self.add("asleep", "sleeping")
        self.nodes = {HIGHER: {"g": _saturated(queued=0)}}
        self.tick(5)
        self.assertEqual(self.woken(), [])

    def test_a_free_slot_anywhere_is_not_pressure(self):
        self.add("asleep", "sleeping")
        self.nodes = {HIGHER: {"g": _saturated()},
                      "other": {"g": {**_saturated(queued=0), "free": 1}}}
        self.tick(5)
        self.assertEqual(self.woken(), [])

    def test_a_member_already_loading_holds_off_another_wake(self):
        self.add("asleep", "sleeping")
        self.nodes = {HIGHER: {"g": {**_saturated(), "starting": 1}}}
        self.tick(5)
        self.assertEqual(self.woken(), [])

    def test_nothing_healthy_is_not_pressure(self):
        # A request wakes a sleeper itself then.
        self.add("asleep", "sleeping")
        self.nodes = {HIGHER: {"g": _saturated(healthy=0, sleeping=1)}}
        self.tick(5)
        self.assertEqual(self.woken(), [])

    def test_a_different_group_on_the_peer_is_not_pressure(self):
        self.add("asleep", "sleeping", group="chat")
        self.nodes = {HIGHER: {"chat-pool": _saturated()}}   # substring must not count
        self.tick(5)
        self.assertEqual(self.woken(), [])

    def test_an_older_peer_counts_as_one_healthy_member(self):
        self.add("asleep", "sleeping")
        old = {"active": 2, "queued": 3, "free": 0, "max_concurrent": 2, "fallback": False}
        self.nodes = {HIGHER: {"g": old}}
        self.tick(3)
        self.assertEqual(self.woken(), ["asleep"])

    def test_an_unreachable_peer_is_no_evidence(self):
        self.add("asleep", "sleeping")
        self.nodes = {HIGHER: {}}
        self.tick(5)
        self.assertEqual(self.woken(), [])

    def test_single_node_needs_no_cluster(self):
        self.add("up", "healthy", busy=2, queued=1)
        self.add("asleep", "sleeping")
        with patch("api.cluster.cl.is_cluster_enabled", return_value=False):
            self.tick(3)
        self.assertEqual(self.woken(), ["asleep"])
        self.storage.list_nodes.assert_not_called()


class StampedeTests(_Case):

    def test_only_the_lowest_node_with_a_sleeper_wakes(self):
        self.add("asleep", "sleeping")
        self.nodes = {LOWER: {"g": _saturated(healthy=0, sleeping=1)},
                      "busy": {"g": _saturated()}}
        self.tick(5)
        self.assertEqual(self.woken(), [])

    def test_this_node_wakes_when_it_is_the_lowest(self):
        self.add("asleep", "sleeping")
        self.nodes = {HIGHER: {"g": _saturated(sleeping=1)}}
        self.tick(3)
        self.assertEqual(self.woken(), ["asleep"])

    def test_one_wake_per_group_per_pass(self):
        self.add("a", "sleeping")
        self.add("b", "sleeping")
        self.nodes = {HIGHER: {"g": _saturated()}}
        self.tick(3)
        self.assertEqual(len(self.woken()), 1)

    def test_two_groups_wake_independently(self):
        self.add("a", "sleeping", group="g1")
        self.add("b", "sleeping", group="g2")
        self.nodes = {HIGHER: {"g1": _saturated(), "g2": _saturated()}}
        self.tick(3)
        self.assertCountEqual(self.woken(), ["a", "b"])

    def test_a_member_being_woken_is_not_woken_again(self):
        self.add("asleep", "sleeping")
        self.nodes = {HIGHER: {"g": _saturated()}}
        release = threading.Event()
        self.wake.side_effect = lambda _id: release.wait(5)
        self.addCleanup(release.set)
        for _ in range(6):
            mon._eager_wake_tick()
        self.assertEqual(self.wake.call_count, 1)
        release.set()


class EligibilityTests(_Case):

    def test_a_gateless_sleeper_is_skipped(self):
        with instances_lock:
            instances["free"] = {"id": "free", "status": "sleeping", "model_path": "/models/f.gguf",
                                 "config": {"share_queue": True, "share_queue_group": "g"}}
        self.nodes = {HIGHER: {"g": _saturated()}}
        self.tick(5)
        self.assertEqual(self.woken(), [])

    def test_a_sleeper_sharing_a_gate_with_an_awake_sibling_is_skipped(self):
        # It adds no gate capacity: the gate is what's full.
        self.add("up", "healthy", busy=2, queued=1, path="/models/same.gguf", shared=True)
        self.add("asleep", "sleeping", path="/models/same.gguf", shared=True)
        self.tick(5)
        self.assertEqual(self.woken(), [])

    def test_a_primary_wakes_before_a_fallback(self):
        self.add("spare", "sleeping", fallback=True)
        self.add("primary", "sleeping")
        self.nodes = {HIGHER: {"g": _saturated()}}
        self.tick(3)
        self.assertEqual(self.woken(), ["primary"])

    def test_a_member_without_share_queue_is_skipped(self):
        self.add("asleep", "sleeping")
        instances["asleep"]["config"]["share_queue"] = False
        self.nodes = {HIGHER: {"g": _saturated()}}
        self.tick(5)
        self.assertEqual(self.woken(), [])


class HysteresisTests(_Case):

    def setUp(self):
        super().setUp()
        self.add("asleep", "sleeping")
        self.nodes = {HIGHER: {"g": _saturated()}}

    def test_a_blip_does_not_wake(self):
        self.tick(2)
        self.assertEqual(self.woken(), [])

    def test_a_periodic_burst_never_wakes(self):
        # Waking on each would pay cold loads repeatedly, worse than not waking.
        for _ in range(4):
            self.nodes = {HIGHER: {"g": _saturated()}}
            self.tick(2)
            self.nodes = {HIGHER: {"g": _saturated(queued=0)}}
            self.tick(1)
        self.assertEqual(self.woken(), [])

    def test_the_poll_count_is_a_live_setting(self):
        self.settings = {"eager_wake_polls": 1}
        self.tick(1)
        self.assertEqual(self.woken(), ["asleep"])

    def test_it_can_be_turned_off(self):
        self.settings = {"eager_wake_enabled": False}
        self.tick(5)
        self.assertEqual(self.woken(), [])

    def test_a_bad_poll_count_falls_back(self):
        self.settings = {"eager_wake_polls": "soon"}
        self.tick(2)
        self.assertEqual(self.woken(), [])
        self.tick(1)
        self.assertEqual(self.woken(), ["asleep"])

    def test_settings_are_per_node(self):
        self.settings = {"nodes": {SELF: {"eager_wake_enabled": False}}, "eager_wake_enabled": True}
        self.tick(5)
        self.assertEqual(self.woken(), [])


class VramTests(_Case):

    def setUp(self):
        super().setUp()
        self.add("asleep", "sleeping")
        self.nodes = {HIGHER: {"g": _saturated()}}
        self.settings = {"eager_wake_polls": 1}
        self.size = patch("core.monitoring.os.path.getsize", return_value=8 * 1024 ** 3)
        self.size.start()
        self.addCleanup(self.size.stop)
        meta = patch("api.models.get_cached_gguf_metadata", return_value={})
        meta.start()
        self.addCleanup(meta.stop)

    def _gpus(self, *free_gb):
        return patch("core.gpu.query_gpus",
                     return_value=[{"memory_free_mb": gb * 1024} for gb in free_gb])

    def test_a_model_that_doesnt_fit_stays_asleep_and_is_logged_once(self):
        with self._gpus(4), self.assertLogs("llamaman", level="WARNING") as logs:
            self.tick(4)
        self.assertEqual(self.woken(), [])
        self.assertEqual(len([m for m in logs.output if "does not fit" in m]), 1)

    def test_a_model_that_fits_wakes(self):
        with self._gpus(16):
            self.tick()
        self.assertEqual(self.woken(), ["asleep"])

    def test_free_memory_across_gpus_counts(self):
        with self._gpus(4, 4):
            self.tick()
        self.assertEqual(self.woken(), ["asleep"])

    def test_it_wakes_once_memory_frees_up(self):
        with self._gpus(4):
            self.tick(2)
        with self._gpus(16):
            self.tick()
        self.assertEqual(self.woken(), ["asleep"])

    def test_a_cpu_only_member_needs_no_vram(self):
        instances["asleep"]["config"]["n_gpu_layers"] = 0
        with self._gpus(0):
            self.tick()
        self.assertEqual(self.woken(), ["asleep"])

    def test_partial_offload_counts_only_its_share(self):
        instances["asleep"]["config"]["n_gpu_layers"] = 10
        meta = {"general.architecture": "x", "x.block_count": 40}
        with self._gpus(4), patch("api.models.get_cached_gguf_metadata", return_value=meta):
            self.tick()
        self.assertEqual(self.woken(), ["asleep"])

    def test_no_gpu_stats_fails_open(self):
        with patch("core.gpu.query_gpus", return_value=None):
            self.tick()
        self.assertEqual(self.woken(), ["asleep"])

    def test_a_model_with_no_file_fails_open(self):
        with self._gpus(1), patch("core.monitoring.os.path.getsize", side_effect=OSError("virtual")):
            self.tick()
        self.assertEqual(self.woken(), ["asleep"])


class SettingsApiTests(unittest.TestCase):
    """Reachable through the settings API, per node, with bad values normalized."""

    def setUp(self):
        from flask import Flask

        import api.settings as settings_api
        self.saved = {}
        self.storage = Mock()
        self.storage.get_settings.side_effect = lambda: self.saved

        def merge(patch):
            nodes = patch.get("nodes") or {}
            for node_id, values in nodes.items():
                self.saved.setdefault("nodes", {}).setdefault(node_id, {}).update(values)
            self.saved.update({k: v for k, v in patch.items() if k != "nodes"})
            return self.saved

        self.storage.merge_settings.side_effect = merge
        for target in ("api.settings.get_storage", "core.node_settings.get_storage"):
            patcher = patch(target, lambda: self.storage)
            patcher.start()
            self.addCleanup(patcher.stop)
        app = Flask(__name__)
        app.register_blueprint(settings_api.bp)
        self.client = app.test_client()

    def _post(self, body):
        return self.client.post("/api/settings", json=body)

    def test_defaults_are_on_after_three_polls(self):
        body = self.client.get("/api/settings").get_json()
        self.assertIs(body["eager_wake_enabled"], True)
        self.assertEqual(body["eager_wake_polls"], 3)

    def test_values_are_stored_for_this_node_only(self):
        self.assertEqual(self._post({"eager_wake_enabled": False, "eager_wake_polls": 6}).status_code, 200)
        self.assertEqual(self.saved["nodes"][SELF], {"eager_wake_enabled": False, "eager_wake_polls": 6})
        self.assertNotIn("eager_wake_enabled", self.saved)
        body = self.client.get("/api/settings").get_json()
        self.assertIs(body["eager_wake_enabled"], False)
        self.assertEqual(body["eager_wake_polls"], 6)

    def test_bad_values_are_normalized(self):
        self._post({"eager_wake_enabled": "false", "eager_wake_polls": 0})
        self.assertIs(self.saved["nodes"][SELF]["eager_wake_enabled"], False)
        self.assertEqual(self.saved["nodes"][SELF]["eager_wake_polls"], 1)


class ResilienceTests(_Case):

    def test_nothing_asleep_reads_neither_storage_nor_peers(self):
        self.add("up", "healthy")
        self.tick()
        self.storage.get_settings.assert_not_called()
        self.storage.list_nodes.assert_not_called()

    def test_a_group_with_a_free_local_slot_asks_no_peer(self):
        self.add("up", "healthy")
        self.add("asleep", "sleeping")
        self.tick()
        self.storage.list_nodes.assert_not_called()

    def test_a_failed_wake_does_not_block_the_next(self):
        self.add("asleep", "sleeping")
        self.nodes = {HIGHER: {"g": _saturated()}}
        self.settings = {"eager_wake_polls": 1}
        self.wake.side_effect = [False, True]
        self.tick()
        self.tick()
        self.assertEqual(self.woken(), ["asleep", "asleep"])

    def test_a_raising_wake_clears_the_in_flight_mark(self):
        self.add("asleep", "sleeping")
        self.nodes = {HIGHER: {"g": _saturated()}}
        self.settings = {"eager_wake_polls": 1}
        self.wake.side_effect = RuntimeError("boom")
        self.tick()
        self.assertEqual(mon._eager_wakes_in_flight, set())

    def test_a_storage_failure_only_skips_the_pass(self):
        self.add("asleep", "sleeping")
        self.storage.get_settings.side_effect = RuntimeError("database unreachable")
        calls = []

        def stop_after_two(_seconds):
            calls.append(1)
            if len(calls) > 2:
                raise SystemExit
        with patch("core.monitoring.time.sleep", stop_after_two), \
                self.assertRaises(SystemExit), self.assertLogs("llamaman", level="WARNING"):
            mon._eager_wake_loop()
        self.assertEqual(len(calls), 3)   # went round again after the failures


if __name__ == "__main__":
    unittest.main()
