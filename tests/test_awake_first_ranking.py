# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""#111 gap 1: a sleeping member's gate reads active=0, so on load alone it won
every tie and any backlog, and queued work migrated onto it as "free". Load
reporting now tells members apart by how awake they are, and everything that
picks a node ranks awake capacity ahead of a cold load."""

import os
import unittest
from unittest.mock import Mock, patch

from flask import Flask

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import api.cluster as ca
import proxy
from core.state import instances, instances_lock

SELF = ca.cl.get_node_id()


def _candidate(name, load=0, fallback=False, tier=0, is_self=False):
    return {"node": {"node_id": name}, "load": load, "fallback": fallback,
            "tier": tier, "is_self": is_self}


def _live(healthy=0, starting=0, sleeping=0, active=0, queued=0, free=0, fallback=False):
    return {"active": active, "queued": queued, "free": free, "max_concurrent": 2,
            "fallback": fallback, "healthy": healthy, "starting": starting, "sleeping": sleeping}


def _peer(name, features=("alias_wake",), instances_=()):
    return {"node_id": name, "node_name": name, "advertise_url": f"http://{name}:5000",
            "snapshot": {"system": {"features": list(features)}, "instances": list(instances_)}}


class OrderTests(unittest.TestCase):

    def _names(self, candidates):
        return [c["node"]["node_id"] for c in ca._order_candidates(candidates)]

    def test_the_full_tier_order(self):
        by_name = {
            "primary-healthy": _candidate("primary-healthy", load=9),
            "primary-starting": _candidate("primary-starting", load=8, tier=1),
            "fallback-healthy": _candidate("fallback-healthy", load=7, fallback=True),
            "fallback-starting": _candidate("fallback-starting", load=6, fallback=True, tier=1),
            "primary-sleeping": _candidate("primary-sleeping", tier=2),
            "fallback-sleeping": _candidate("fallback-sleeping", fallback=True, tier=2),
        }
        self.assertEqual(self._names(list(by_name.values())[::-1]), list(by_name))

    def test_a_tie_with_a_sleeper_never_goes_to_the_sleeper(self):
        # "Roughly every other request fails": both read load 0 and used to rotate.
        for _ in range(6):
            self.assertEqual(self._names([_candidate("asleep", tier=2), _candidate("up")])[0], "up")

    def test_a_backlog_on_the_awake_member_does_not_lose_to_a_sleeper(self):
        names = self._names([_candidate("asleep", tier=2), _candidate("up", load=50)])
        self.assertEqual(names, ["up", "asleep"])

    def test_ties_within_a_tier_still_rotate(self):
        firsts = {self._names([_candidate("a"), _candidate("b")])[0] for _ in range(4)}
        self.assertEqual(firsts, {"a", "b"})

    def test_a_bare_candidate_counts_as_awake(self):
        bare = {"node": {"node_id": "old"}, "load": 5, "is_self": False}
        self.assertEqual(self._names([_candidate("asleep", tier=2), bare]), ["old", "asleep"])


class _LocalCase(unittest.TestCase):

    def setUp(self):
        with instances_lock:
            saved = dict(instances)
            instances.clear()
        self.gates = []

        def restore():
            for inst_id in self.gates:
                proxy.remove_gate(inst_id)
            with instances_lock:
                instances.clear()
                instances.update(saved)
        self.addCleanup(restore)

    def add(self, inst_id, status, path=None, busy=0, fallback=False, group="g", mc=2, shared=False):
        path = path or f"/models/{inst_id}.gguf"
        proxy.create_gate(inst_id, mc, 10, model_path=path, share_queue=shared)
        self.gates.append(inst_id)
        gate = proxy.get_gate(inst_id)
        for _ in range(busy):
            self.assertTrue(gate.acquire(timeout=1))
        self.addCleanup(lambda: [gate.release() for _ in range(busy)])
        with instances_lock:
            instances[inst_id] = {
                "id": inst_id, "status": status, "model_path": path,
                "config": {"share_queue": True, "share_queue_group": group,
                           "share_queue_fallback": fallback, "max_concurrent": mc},
            }


class LocalLoadTests(_LocalCase):

    def test_a_sleeper_adds_no_free_slots_beside_a_saturated_member(self):
        # The issue's repro: this used to report free=2.
        self.add("up", "healthy", busy=2)
        self.add("asleep", "sleeping")
        entry = ca.local_load_by_model()["g"]
        self.assertEqual(entry["free"], 0)
        self.assertEqual((entry["healthy"], entry["starting"], entry["sleeping"]), (1, 0, 1))

    def test_a_starting_member_adds_no_free_slots(self):
        self.add("loading", "starting")
        entry = ca.local_load_by_model()["g"]
        self.assertEqual(entry["free"], 0)
        self.assertEqual(entry["starting"], 1)

    def test_only_sleepers_report_as_sleeping_with_nothing_free(self):
        self.add("asleep", "sleeping")
        entry = ca.local_load_by_model()["g"]
        self.assertEqual((entry["free"], entry["healthy"], entry["sleeping"]), (0, 0, 1))

    def test_stopped_and_stopping_members_are_not_counted(self):
        self.add("gone", "stopped")
        self.add("going", "stopping")
        self.assertEqual(ca.local_load_by_model(), {})

    def test_members_sharing_a_gate_count_it_once(self):
        self.add("one", "healthy", path="/models/same.gguf", busy=1, shared=True)
        self.add("two", "healthy", path="/models/same.gguf", shared=True)
        entry = ca.local_load_by_model()["g"]
        self.assertEqual((entry["active"], entry["free"], entry["max_concurrent"]), (1, 1, 2))
        self.assertEqual(entry["healthy"], 2)

    def test_a_gate_shared_with_a_sleeper_is_free_whichever_is_listed_first(self):
        self.add("asleep", "sleeping", path="/models/same.gguf", shared=True)
        self.add("up", "healthy", path="/models/same.gguf", shared=True)
        self.assertEqual(ca.local_load_by_model()["g"]["free"], 2)

    def test_a_gateless_member_is_unlimited(self):
        with instances_lock:
            instances["free"] = {"id": "free", "status": "healthy", "model_path": "/models/f.gguf",
                                 "config": {"share_queue": True, "share_queue_group": "g"}}
        self.assertEqual(ca.local_load_by_model()["g"]["free"], 1)

    def test_a_healthy_fallback_outranks_the_sleeping_primary_beside_it(self):
        self.add("asleep", "sleeping")
        self.add("spare", "healthy", fallback=True)
        self.assertTrue(ca.local_load_by_model()["g"]["fallback"])
        self.assertTrue(ca._local_group_load("g")[1])

    def test_one_primary_in_the_best_tier_still_flips_the_flag(self):
        self.add("primary", "healthy")
        self.add("spare", "healthy", fallback=True)
        self.assertFalse(ca.local_load_by_model()["g"]["fallback"])
        self.assertFalse(ca._local_group_load("g")[1])


class LocalGroupLoadTests(_LocalCase):

    def test_a_sleeper_does_not_make_a_busy_node_look_idle(self):
        # The issue's repro: min() across members used to report 0.
        self.add("up", "healthy", busy=2)
        self.add("asleep", "sleeping")
        self.assertEqual(ca._local_group_load("g"), (2, False, ca._TIER_HEALTHY))

    def test_starting_is_used_when_nothing_is_healthy(self):
        self.add("loading", "starting", busy=1)
        self.add("asleep", "sleeping")
        self.assertEqual(ca._local_group_load("g"), (1, False, ca._TIER_STARTING))

    def test_a_node_of_only_sleepers_is_the_sleeping_tier(self):
        self.add("asleep", "sleeping")
        self.assertEqual(ca._local_group_load("g"), (0, False, ca._TIER_SLEEPING))

    def test_no_members_is_none(self):
        self.add("gone", "stopped")
        self.assertIsNone(ca._local_group_load("g"))


class _PeersCase(_LocalCase):

    def peers(self, *entries):
        """entries: (node, live load)."""
        loads = {node["node_id"]: live for node, live in entries}
        storage = Mock()
        storage.list_nodes.return_value = [node for node, _ in entries]
        for target, value in (
            ("api.cluster.get_storage", lambda: storage),
            ("api.cluster._is_online", lambda _n: True),
            ("api.cluster._inflight_get", lambda _n: 0),
            ("api.cluster._peer_live_load", lambda node, _m: loads[node["node_id"]]),
            ("api.cluster.cl.is_cluster_enabled", lambda: True),
        ):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)


class FreePeerTests(_PeersCase):

    def test_a_queued_request_does_not_migrate_to_a_sleeper(self):
        # Its `free` is computed from healthy members, so a new peer reports 0.
        self.peers((_peer("asleep"), _live(sleeping=1, free=0)))
        self.assertIsNone(ca._find_free_peer("g"))

    def test_it_migrates_to_a_real_free_slot(self):
        up = _peer("up")
        self.peers((_peer("asleep"), _live(sleeping=1, free=0)), (up, _live(healthy=1, free=1)))
        self.assertIs(ca._find_free_peer("g"), up)


class EvacuationTests(_PeersCase):

    def test_an_awake_peer_beats_an_idle_sleeper(self):
        busy = _peer("busy")
        self.peers((_peer("asleep"), _live(sleeping=1)), (busy, _live(healthy=1, active=9)))
        self.assertIs(ca._find_evacuation_peer("g"), busy)

    def test_a_sleeper_is_still_a_target_when_nothing_else_is_left(self):
        asleep = _peer("asleep")
        self.peers((asleep, _live(sleeping=1)))
        self.assertIs(ca._find_evacuation_peer("g"), asleep)

    def test_an_awake_fallback_beats_a_sleeping_primary(self):
        spare = _peer("spare")
        self.peers((_peer("asleep"), _live(sleeping=1)), (spare, _live(healthy=1, fallback=True)))
        self.assertIs(ca._find_evacuation_peer("g"), spare)

    def test_least_backlog_wins_within_a_tier(self):
        quiet = _peer("quiet")
        self.peers((_peer("busy"), _live(healthy=1, active=5)), (quiet, _live(healthy=1, active=1)))
        self.assertIs(ca._find_evacuation_peer("g"), quiet)


class OlderPeerTests(_PeersCase):
    """A peer without alias_wake 500s an alias unless its member is healthy."""

    # An older peer's live load: no counts, and a sleeper reads as free.
    OLD_LIVE = {"active": 0, "queued": 0, "free": 2, "max_concurrent": 2, "fallback": False}

    @staticmethod
    def _inst(status):
        return {"status": status, "model_path": "/models/m.gguf",
                "config": {"share_queue": True, "share_queue_group": "g"}}

    def _candidate_nodes(self):
        return [c["node"]["node_id"] for c in ca._group_candidates("g")]

    def test_a_sleeping_member_is_skipped_everywhere(self):
        self.peers((_peer("old", features=(), instances_=[self._inst("sleeping")]), self.OLD_LIVE))
        self.assertEqual(self._candidate_nodes(), [])
        self.assertIsNone(ca._find_free_peer("g"))
        self.assertIsNone(ca._find_evacuation_peer("g"))

    def test_a_healthy_member_still_gets_traffic(self):
        # Old peers keep working as they do today.
        self.peers((_peer("old", features=(), instances_=[self._inst("healthy")]), self.OLD_LIVE))
        self.assertEqual(self._candidate_nodes(), ["old"])

    def test_its_candidate_counts_as_awake(self):
        self.peers((_peer("old", features=(), instances_=[self._inst("healthy")]), self.OLD_LIVE))
        self.assertEqual(ca._group_candidates("g")[0]["tier"], ca._TIER_HEALTHY)

    def test_a_peer_advertising_the_feature_is_taken_at_its_word(self):
        self.peers((_peer("new", instances_=[self._inst("sleeping")]), _live(sleeping=1)))
        candidates = ca._group_candidates("g")
        self.assertEqual(candidates[0]["tier"], ca._TIER_SLEEPING)

    def test_no_snapshot_leaves_the_live_load_to_decide(self):
        node = {"node_id": "bare", "advertise_url": "http://bare:5000"}
        self.peers((node, self.OLD_LIVE))
        self.assertEqual(self._candidate_nodes(), ["bare"])


class DispatchTests(_PeersCase):
    """The real dispatch path, entry node included."""

    def _dispatch(self):
        forward = Mock(return_value="forwarded")
        app = Flask(__name__)
        with app.test_request_context("/v1/chat/completions", method="POST", json={}), \
                patch("api.cluster._forward_inference", forward):
            return ca.dispatch_inference("g"), forward

    def test_every_request_goes_to_the_healthy_peer_not_the_sleeper(self):
        up = _peer("up")
        self.peers((_peer("asleep"), _live(sleeping=1)), (up, _live(healthy=1)))
        for _ in range(6):
            _, forward = self._dispatch()
            self.assertIs(forward.call_args.args[0], up)

    def test_a_healthy_peer_is_preferred_over_this_nodes_own_sleeper(self):
        self.add("mine", "sleeping")
        up = _peer("up")
        self.peers((up, _live(healthy=1, active=3)))
        out, forward = self._dispatch()
        self.assertEqual(out, "forwarded")
        self.assertIs(forward.call_args.args[0], up)

    def test_this_nodes_healthy_member_is_served_here_over_a_peers_sleeper(self):
        self.add("mine", "healthy", busy=2)
        self.peers((_peer("asleep"), _live(sleeping=1)))
        out, forward = self._dispatch()
        self.assertIsNone(out)
        forward.assert_not_called()

    def test_a_sleeper_is_used_when_nothing_is_awake(self):
        asleep = _peer("asleep")
        self.add("mine", "stopped")
        self.peers((asleep, _live(sleeping=1)), (_peer("spare"), _live(sleeping=1, fallback=True)))
        _, forward = self._dispatch()
        self.assertIs(forward.call_args.args[0], asleep)

    def test_a_peer_that_cannot_serve_falls_through_to_the_next(self):
        asleep = _peer("asleep")
        self.peers((_peer("up"), _live(healthy=1)), (asleep, _live(sleeping=1)))
        forward = Mock(side_effect=[None, "forwarded"])
        app = Flask(__name__)
        with app.test_request_context("/v1/chat/completions", method="POST", json={}), \
                patch("api.cluster._forward_inference", forward):
            self.assertEqual(ca.dispatch_inference("g"), "forwarded")
        self.assertIs(forward.call_args.args[0], asleep)


if __name__ == "__main__":
    unittest.main()
