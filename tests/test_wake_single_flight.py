# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""relaunch_inactive_instance is single-flight per instance. Status stays
"sleeping" until the container is up, so without it concurrent wakes (proxy
wake-on-request, kb, auto-restart, model API) all pass the status check and
start the same container: on Docker the loser's 409 name conflict used to
force-remove the winner's container; bare-metal the loser saw the port taken
and marked the instance stopped."""

import os
import threading
import unittest
from unittest.mock import Mock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import docker

import api.instances as instances_api
from core.state import instances, instances_lock

CALLERS = 8
TIMEOUT = 10


def _sleeper(inst_id):
    return {
        "id": inst_id, "status": "sleeping", "model_path": f"/models/{inst_id}.gguf",
        "port": 8000, "_internal_port": 9000, "container_name": f"llamaman-{inst_id}",
        "config": {"ctx_size": 4096},
    }


def _conflict():
    return docker.errors.APIError("Conflict: name already in use",
                                  response=Mock(status_code=409))


class _FakeDocker:
    """Enforces Docker's name rule: creating a live name is a 409. run() holds
    until released, so a test controls how long a wake stays in flight."""

    def __init__(self):
        self.containers = self
        self.release = threading.Event()
        self.lock = threading.Lock()
        self.live = {}
        self.run_calls = 0
        self.removed = []

    def run(self, **kwargs):
        name = kwargs["name"]
        with self.lock:
            self.run_calls += 1
            if name in self.live:
                raise _conflict()
            container = Mock(id=f"id-{name}", status="running")
            self.live[name] = container
        self.release.wait(TIMEOUT)
        return container

    def get(self, name):
        with self.lock:
            if name not in self.live:
                raise docker.errors.NotFound("gone")
            container = self.live[name]

        def remove(force=False):
            with self.lock:
                self.removed.append(name)
                self.live.pop(name, None)
        container.remove = remove
        return container


class _WakeCase(unittest.TestCase):

    def setUp(self):
        with instances_lock:
            self._saved = dict(instances)
            instances.clear()
        self.addCleanup(self._restore)
        for target, value in (
            ("api.instances.save_state", Mock()),
            ("api.instances.is_port_available", Mock(return_value=True)),
            ("api.instances._merge_preset_into_config", lambda _path, config: config),
            ("api.instances.resolve_llama_endpoint", Mock(return_value=("host", 8080))),
            ("api.instances.wait_for_healthy", Mock(return_value=True)),
            ("api.instances.load_timeout_for", Mock(return_value=1)),
        ):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _restore(self):
        with instances_lock:
            instances.clear()
            instances.update(self._saved)

    def _wake_all(self, inst_id, count=CALLERS, before_join=None):
        """`count` concurrent wakes; returns their results."""
        results = []
        lock = threading.Lock()

        def wake():
            result = instances_api.relaunch_inactive_instance(inst_id)
            with lock:
                results.append(result)

        threads = [threading.Thread(target=wake, daemon=True) for _ in range(count)]
        for t in threads:
            t.start()
        if before_join:
            before_join()
        for t in threads:
            t.join(TIMEOUT)
        self.assertFalse(any(t.is_alive() for t in threads), "a wake never returned")
        return results


class SingleFlightTests(_WakeCase):

    def test_concurrent_wakes_start_one_container_and_all_succeed(self):
        # The real _run_container against a fake daemon: this is the Docker
        # deployment's failure, where a loser's 409 removed the live container.
        instances["a"] = _sleeper("a")
        fake = _FakeDocker()
        with patch("api.instances.get_docker_client", return_value=fake), \
             patch("api.instances.ensure_docker_network"), \
             patch("api.instances._start_log_relay"), \
             patch("api.instances.get_vendor", return_value="cpu"):
            # Let the others reach the wake while the first is mid-start.
            results = self._wake_all(
                "a", before_join=lambda: (threading.Event().wait(0.2), fake.release.set()))

        self.assertEqual(results, [True] * CALLERS)
        self.assertEqual(fake.run_calls, 1)
        self.assertEqual(fake.removed, [])
        self.assertIn("llamaman-a", fake.live)
        self.assertEqual(instances["a"]["status"], "healthy")

    def test_a_port_taken_by_our_own_wake_does_not_mark_it_stopped(self):
        # Bare-metal: once the first wake's container holds the port, a second
        # caller's port check fails. It must never get that far.
        instances["a"] = _sleeper("a")
        release = threading.Event()
        run = Mock(side_effect=lambda *a, **k: (release.wait(TIMEOUT), (Mock(id="c"), None))[1])
        seen = []

        def port_free(_port):
            seen.append(_port)
            return len(seen) == 1

        with patch("api.instances._run_container", run), \
             patch("api.instances.is_port_available", port_free), \
             patch("api.instances.get_engine"):
            results = self._wake_all(
                "a", before_join=lambda: (threading.Event().wait(0.2), release.set()))

        self.assertEqual(results, [True] * CALLERS)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(instances["a"]["status"], "healthy")

    def test_joined_callers_share_a_failed_wake_and_a_later_one_retries(self):
        instances["a"] = _sleeper("a")
        release = threading.Event()
        run = Mock(side_effect=lambda *a, **k: (release.wait(TIMEOUT), (None, "boom"))[1])
        with patch("api.instances._run_container", run):
            results = self._wake_all(
                "a", before_join=lambda: (threading.Event().wait(0.2), release.set()))
        self.assertFalse(any(results))
        self.assertEqual(instances["a"]["status"], "sleeping")

        # A failed wake must not block the next attempt.
        calls_before = run.call_count
        run.side_effect = None
        run.return_value = (Mock(id="c"), None)
        with patch("api.instances._run_container", run), patch("api.instances.get_engine"):
            self.assertTrue(instances_api.relaunch_inactive_instance("a"))
        self.assertEqual(run.call_count, calls_before + 1)
        self.assertEqual(instances["a"]["status"], "healthy")

    def test_a_raising_wake_releases_joined_callers_and_the_next_attempt(self):
        instances["a"] = _sleeper("a")
        started = threading.Event()
        release = threading.Event()

        def explode(*_a, **_k):
            started.set()
            release.wait(TIMEOUT)
            raise RuntimeError("boom")

        leader_error = []

        def leader():
            try:
                instances_api.relaunch_inactive_instance("a")
            except RuntimeError as e:
                leader_error.append(e)

        with patch("api.instances._run_container", side_effect=explode):
            thread = threading.Thread(target=leader, daemon=True)
            thread.start()
            self.assertTrue(started.wait(TIMEOUT))
            results = self._wake_all(
                "a", count=3, before_join=lambda: (threading.Event().wait(0.2), release.set()))
            thread.join(TIMEOUT)

        self.assertEqual(len(leader_error), 1)   # the caller that ran it sees the error
        self.assertEqual(results, [False] * 3)
        with patch("api.instances._run_container", return_value=(Mock(id="c"), None)) as run, \
             patch("api.instances.get_engine"):
            self.assertTrue(instances_api.relaunch_inactive_instance("a"))
        run.assert_called_once()

    def test_different_instances_wake_in_parallel(self):
        # Single-flight is per instance: one cold load must not hold up another.
        instances["a"] = _sleeper("a")
        instances["b"] = _sleeper("b")
        both_in_flight = threading.Barrier(2, timeout=TIMEOUT)

        def run(*_a, **_k):
            both_in_flight.wait()   # BrokenBarrierError if they were serialized
            return Mock(id="c"), None

        results = []
        with patch("api.instances._run_container", side_effect=run), \
             patch("api.instances.get_engine"):
            threads = [threading.Thread(
                target=lambda i=i: results.append(instances_api.relaunch_inactive_instance(i)),
                daemon=True) for i in ("a", "b")]
            for t in threads:
                t.start()
            for t in threads:
                t.join(TIMEOUT * 2)
        self.assertEqual(results, [True, True])

    def test_awake_instances_still_short_circuit(self):
        instances["a"] = {**_sleeper("a"), "status": "healthy"}
        instances["b"] = {**_sleeper("b"), "status": "starting"}
        instances["c"] = {**_sleeper("c"), "status": "stopping"}
        with patch("api.instances._run_container") as run:
            self.assertTrue(instances_api.relaunch_inactive_instance("a"))
            self.assertTrue(instances_api.relaunch_inactive_instance("b"))
            self.assertFalse(instances_api.relaunch_inactive_instance("c"))
            self.assertFalse(instances_api.relaunch_inactive_instance("missing"))
        run.assert_not_called()


class NameConflictCleanupTests(unittest.TestCase):
    """A 409 says the name belongs to something that isn't ours to assume dead."""

    def _run(self, error, existing_status):
        client = Mock()
        client.containers.run.side_effect = error
        existing = Mock(status=existing_status)
        client.containers.get.return_value = existing
        with patch("api.instances.get_docker_client", return_value=client), \
             patch("api.instances.ensure_docker_network"), \
             patch("api.instances._start_log_relay"), \
             patch("api.instances.get_vendor", return_value="cpu"):
            container, err = instances_api._run_container(
                "inst1", "llamaman-test-8000", "/models/m.gguf", 8000,
                {"ctx_size": 4096}, os.path.join(REPO_ROOT, "test-logs", "x.log"))
        self.assertIsNone(container)
        self.assertIn("Docker API error", err)
        return existing

    def test_a_running_container_is_left_alone(self):
        self._run(_conflict(), "running").remove.assert_not_called()

    def test_a_dead_leftover_still_gets_cleared(self):
        # Otherwise a stale container blocks that name for good.
        for status in ("exited", "created", "dead"):
            self._run(_conflict(), status).remove.assert_called_once_with(force=True)

    def test_other_api_errors_still_remove_whatever_was_created(self):
        error = docker.errors.APIError("no space left on device", response=Mock(status_code=500))
        self._run(error, "running").remove.assert_called_once_with(force=True)


if __name__ == "__main__":
    unittest.main()
