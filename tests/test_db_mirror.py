# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Local DB mirror + degraded-mode operation (storage/resilient.py).

The guarantees under test:
  - with mirroring off the wrapper is indistinguishable from the bare backend;
  - the breaker opens only for connection failures, never for bugs;
  - once open, reads come from the mirror and writes are journalled or refused
    according to whether they can be replayed safely against a shared database;
  - replay is order-preserving, idempotent, and does not clobber concurrent
    edits a peer made to other rows/fields/keys during the outage.
"""

import os
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

from storage.json_backend import JsonBackend
from storage.resilient import (
    ResilientBackend, StorageDegradedError, _apply_preset_delta, _preset_delta,
    build_mirror,
)

NODE = "test-node"


class _Boom(Exception):
    """Stands in for a connection-class error; _is_connection_error is patched."""


class FakePrimary(JsonBackend):
    """A real JsonBackend that can be told to fail like an unreachable database.

    Subclassing the real backend (rather than a MagicMock) keeps the semantics
    honest: replay and reconcile run against something that actually stores rows.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.down = False
        self.calls = []

    def _guard(self, name):
        self.calls.append(name)
        if self.down:
            raise _Boom(f"{name}: database unreachable")

    # Only the methods the tests exercise need the guard.
    for _name in (
        "get_settings", "merge_settings", "save_settings",
        "edit_settings_list", "replace_settings_key",
        "get_preset", "get_all_presets", "save_preset", "delete_preset",
        "get_api_keys", "save_api_key", "delete_api_key", "verify_api_key",
        "user_count", "get_user", "save_user",
        "load_instances", "load_downloads", "save_state",
        "get_model_files", "upsert_model_file", "delete_model_files",
        "append_request_log", "request_log_stats", "list_conversations",
    ):
        def _make(name):
            def _wrapped(self, *a, **kw):
                self._guard(name)
                return getattr(JsonBackend, name)(self, *a, **kw)
            return _wrapped
        locals()[_name] = _make(_name)
    del _name, _make


def _always_connection_error(exc):
    return isinstance(exc, _Boom)


class ResilientTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        d = self._tmp.name
        self.primary_dir = os.path.join(d, "primary")
        self.mirror_dir = os.path.join(d, "mirror")
        os.makedirs(self.primary_dir, exist_ok=True)
        os.makedirs(self.mirror_dir, exist_ok=True)

        self.primary = FakePrimary(
            os.path.join(self.primary_dir, "state.json"),
            os.path.join(self.primary_dir, "presets.json"),
            os.path.join(self.primary_dir, "users.json"),
            os.path.join(self.primary_dir, "settings.json"),
            api_keys_file=os.path.join(self.primary_dir, "api_keys.json"),
            recordings_dir=os.path.join(self.primary_dir, "request_log"),
        )
        self.mirror = build_mirror(self.mirror_dir)

        # Treat _Boom as the connection-class error for the whole module.
        import storage.resilient as R
        self._real_pred = R._is_connection_error
        R._is_connection_error = _always_connection_error

    def tearDown(self):
        import storage.resilient as R
        R._is_connection_error = self._real_pred
        self._tmp.cleanup()

    def make(self, enabled=True, fail_threshold=3):
        return ResilientBackend(
            self.primary, self.mirror_dir, node_id=NODE,
            builder=lambda: self.primary, enabled=enabled,
            probe_interval=3600,  # never fires on its own; tests drive recovery
            fail_threshold=fail_threshold, mirror=self.mirror,
        )

    def trip(self, backend):
        """Drive the breaker open through the configured threshold."""
        self.primary.down = True
        for _ in range(3):
            try:
                backend.get_settings()
            except _Boom:
                pass
        self.assertTrue(backend.is_degraded(), "breaker should be open")


class PassThroughTests(ResilientTestCase):
    """Toggle off must be indistinguishable from using the backend directly."""

    def test_disabled_does_no_mirror_io(self):
        backend = self.make(enabled=False)
        backend.save_preset("/models/a.gguf", {"ctx_size": 2048})
        backend.merge_settings({"require_auth": False})
        self.assertEqual(self.primary.get_preset("/models/a.gguf"), {"ctx_size": 2048})
        # Nothing written to the mirror directory beyond the owner marker.
        self.assertIsNone(self.mirror.get_preset("/models/a.gguf"))
        self.assertEqual(self.mirror.get_settings(), {})

    def test_disabled_propagates_connection_errors(self):
        backend = self.make(enabled=False)
        self.primary.down = True
        with self.assertRaises(_Boom):
            backend.get_settings()
        self.assertFalse(backend.is_degraded())


class BreakerTests(ResilientTestCase):
    def test_opens_only_after_threshold(self):
        backend = self.make()
        self.primary.down = True
        for i in range(2):
            with self.assertRaises(_Boom, msg=f"failure {i + 1} should still raise"):
                backend.get_settings()
            self.assertFalse(backend.is_degraded())
        # Third failure trips it, and that call falls back instead of raising.
        self.assertEqual(backend.get_settings(), {})
        self.assertTrue(backend.is_degraded())

    def test_success_resets_the_failure_count(self):
        backend = self.make()
        self.primary.down = True
        for _ in range(2):
            with self.assertRaises(_Boom):
                backend.get_settings()
        self.primary.down = False
        backend.get_settings()
        self.primary.down = True
        with self.assertRaises(_Boom):
            backend.get_settings()
        self.assertFalse(backend.is_degraded(), "counter should have reset on success")

    def test_non_connection_errors_never_trip_it(self):
        """IntegrityError-class failures are bugs, not outages."""
        backend = self.make()

        def _explode(*a, **kw):
            raise ValueError("constraint violated")

        self.primary.save_preset = _explode
        for _ in range(5):
            with self.assertRaises(ValueError):
                backend.save_preset("/models/a.gguf", {})
        self.assertFalse(backend.is_degraded())


class DegradedReadTests(ResilientTestCase):
    def test_reads_fall_back_to_mirror(self):
        backend = self.make()
        backend.merge_settings({"require_auth": False})
        backend.save_api_key({"id": "k1", "name": "n", "key_hash": "h", "prefix": "p"})
        self.trip(backend)

        self.assertIs(backend.get_settings().get("require_auth"), False)
        self.assertEqual([k["id"] for k in backend.get_api_keys()], ["k1"])

    def test_request_log_is_dropped_not_buffered(self):
        backend = self.make()
        self.trip(backend)
        backend.append_request_log({"conversation_id": "c" * 32, "created_at": 1}, "per_request")
        self.assertEqual(backend.request_log_stats()["turn_count"], 0)
        self.assertEqual(backend.list_conversations(), [])


class BlockedWriteTests(ResilientTestCase):
    def test_save_settings_and_save_user_are_refused(self):
        backend = self.make()
        self.trip(backend)
        with self.assertRaises(StorageDegradedError):
            backend.save_settings({"anything": 1})
        with self.assertRaises(StorageDegradedError):
            backend.save_user("bob", "hash")


class JournalReplayTests(ResilientTestCase):
    def recover(self, backend):
        self.primary.down = False
        self.assertTrue(backend._try_recover())
        self.assertFalse(backend.is_degraded())

    def test_new_preset_created_offline_reaches_the_database(self):
        backend = self.make()
        self.trip(backend)
        backend.save_preset("/models/new.gguf", {"ctx_size": 4096, "n_gpu_layers": -1})
        self.assertEqual(backend.get_preset("/models/new.gguf")["ctx_size"], 4096)
        self.recover(backend)
        self.assertEqual(self.primary.get_preset("/models/new.gguf")["ctx_size"], 4096)

    def test_preset_delta_preserves_a_peer_edit_to_another_field(self):
        """The assertion that justifies journalling a delta instead of the blob."""
        backend = self.make()
        backend.save_preset("/models/m.gguf", {"ctx_size": 2048, "note": "original"})
        self.trip(backend)

        # This node edits ctx_size while offline.
        backend.save_preset("/models/m.gguf", {"ctx_size": 8192, "note": "original"})

        # Meanwhile a peer edits `note` on the same model, straight into the DB.
        self.primary.down = False
        self.primary.save_preset(
            "/models/m.gguf", {"ctx_size": 2048, "note": "peer changed this"})

        self.recover(backend)
        merged = self.primary.get_preset("/models/m.gguf")
        self.assertEqual(merged["ctx_size"], 8192, "our offline edit must apply")
        self.assertEqual(merged["note"], "peer changed this", "peer's field must survive")

    def test_preset_delete_replays(self):
        backend = self.make()
        backend.save_preset("/models/gone.gguf", {"ctx_size": 1})
        self.trip(backend)
        backend.delete_preset("/models/gone.gguf")
        self.recover(backend)
        self.assertIsNone(self.primary.get_preset("/models/gone.gguf"))

    def test_api_key_created_offline_works_locally_then_syncs(self):
        backend = self.make()
        self.trip(backend)
        entry = {"id": "abc123", "name": "offline", "key_hash": _sha("secret"),
                 "prefix": "llm-abcd"}
        backend.save_api_key(entry)
        # Usable immediately on this node, before the database is back.
        self.assertTrue(backend.verify_api_key("secret"))
        self.recover(backend)
        self.assertIn("abc123", [k["id"] for k in self.primary.get_api_keys()])

    def test_api_key_deleted_offline_replays(self):
        backend = self.make()
        backend.save_api_key({"id": "old", "name": "x", "key_hash": _sha("s"), "prefix": "p"})
        self.trip(backend)
        backend.delete_api_key("old")
        self.recover(backend)
        self.assertEqual(self.primary.get_api_keys(), [])

    def test_hf_token_added_offline_merges_with_a_peer_addition(self):
        """Journalling the EDIT, not the list: the peer's token must survive.

        This is the whole reason edit_settings_list exists instead of
        merge_settings({"huggingface_tokens": [...]}) - offline, the window
        between reading the list and replaying it is the length of the outage.
        """
        backend = self.make()
        self.trip(backend)
        backend.edit_settings_list("huggingface_tokens",
                                   add=[{"id": "mine", "name": "added offline"}])

        # A peer adds its own token straight into the database meanwhile.
        self.primary.down = False
        self.primary.edit_settings_list("huggingface_tokens",
                                        add=[{"id": "theirs", "name": "peer"}])

        self.recover(backend)
        ids = {t["id"] for t in self.primary.get_settings()["huggingface_tokens"]}
        self.assertEqual(ids, {"mine", "theirs"})

    def test_hf_token_removed_offline_replays_by_id(self):
        backend = self.make()
        backend.edit_settings_list("huggingface_tokens", add=[{"id": "old", "name": "x"}])
        self.trip(backend)
        backend.edit_settings_list("huggingface_tokens", remove_ids=["old"])
        self.recover(backend)
        self.assertEqual(self.primary.get_settings()["huggingface_tokens"], [])

    def test_replace_key_replays(self):
        backend = self.make()
        self.trip(backend)
        backend.replace_settings_key("model_sources", {"/models/a": {"repo_id": "org/a"}})
        self.recover(backend)
        self.assertEqual(self.primary.get_settings()["model_sources"],
                         {"/models/a": {"repo_id": "org/a"}})

    def test_settings_patch_preserves_a_peer_edit_to_another_key(self):
        backend = self.make()
        self.trip(backend)
        backend.merge_settings({"global_speed_limit_mbps": 50})

        self.primary.down = False
        self.primary.merge_settings({"recording_mode": "per_request"})

        self.recover(backend)
        final = self.primary.get_settings()
        self.assertEqual(final["global_speed_limit_mbps"], 50)
        self.assertEqual(final["recording_mode"], "per_request")

    def test_replay_preserves_order(self):
        backend = self.make()
        self.trip(backend)
        for ctx in (1024, 2048, 4096):
            backend.save_preset("/models/m.gguf", {"ctx_size": ctx})
        self.recover(backend)
        self.assertEqual(self.primary.get_preset("/models/m.gguf")["ctx_size"], 4096)

    def test_replay_is_idempotent(self):
        """A crash mid-replay re-runs the whole journal on the next attempt."""
        backend = self.make()
        self.trip(backend)
        backend.save_preset("/models/m.gguf", {"ctx_size": 4096})
        backend.save_api_key({"id": "k", "name": "n", "key_hash": _sha("s"), "prefix": "p"})

        records = backend._journal.load()
        self.primary.down = False
        for _ in range(3):  # replay the same records repeatedly
            for rec in records:
                backend._replay_one(self.primary, rec)

        self.assertEqual(self.primary.get_preset("/models/m.gguf")["ctx_size"], 4096)
        self.assertEqual(len(self.primary.get_api_keys()), 1)

    def test_journal_survives_a_failed_replay(self):
        backend = self.make()
        self.trip(backend)
        backend.save_preset("/models/m.gguf", {"ctx_size": 4096})
        # Database still down: recovery must not clear the journal.
        with self.assertRaises(_Boom):
            backend._replay_journal(self.primary)
        self.assertEqual(len(backend._journal.load()), 1)

    def test_unknown_op_is_skipped_not_fatal(self):
        backend = self.make()
        backend._replay_one(self.primary, {"op": "nonsense.op"})  # must not raise


class ModelFileReconcileTests(ResilientTestCase):
    def test_deleted_while_degraded_does_not_resurrect(self):
        backend = self.make()
        backend.upsert_model_file(NODE, "/models/a.gguf", repo_id="org/a")
        self.trip(backend)
        backend.delete_model_files(NODE, "/models/a.gguf")

        self.primary.down = False
        backend._reconcile_node_rows(self.primary)
        self.assertNotIn("/models/a.gguf", self.primary.get_model_files(NODE))

    def test_added_while_degraded_is_pushed_back(self):
        backend = self.make()
        self.trip(backend)
        backend.upsert_model_file(NODE, "/models/b.gguf", sha256="b" * 64)

        self.primary.down = False
        backend._reconcile_node_rows(self.primary)
        self.assertEqual(
            self.primary.get_model_files(NODE)["/models/b.gguf"]["sha256"], "b" * 64)


class RefreshTests(ResilientTestCase):
    def test_refresh_pulls_peer_written_rows(self):
        backend = self.make()
        self.primary.save_preset("/models/peer.gguf", {"ctx_size": 512})
        self.assertTrue(backend.refresh_mirror())
        self.assertEqual(self.mirror.get_preset("/models/peer.gguf")["ctx_size"], 512)

    def test_refresh_is_skipped_while_the_journal_is_pending(self):
        """Pulling the database over the mirror would discard unreplayed edits."""
        backend = self.make()
        self.trip(backend)
        backend.save_preset("/models/m.gguf", {"ctx_size": 4096})
        self.primary.down = False
        backend._degraded = False  # database back, journal not yet replayed
        self.assertFalse(backend.refresh_mirror())

    def test_refresh_drops_keys_removed_on_another_node(self):
        backend = self.make()
        self.mirror.save_api_key({"id": "stale", "name": "x", "key_hash": "h", "prefix": "p"})
        self.assertTrue(backend.refresh_mirror())
        self.assertEqual(self.mirror.get_api_keys(), [])


class RoundTripTests(ResilientTestCase):
    """Healthy -> degraded -> healthy, driven through the real call paths.

    These install the wrapper as the storage singleton so core.state.save_state()
    re-enters it, which is the difference that matters: the node-row reconcile
    runs through save_state(), and if it fired while the breaker was still open
    it would land in the mirror and never reach the database.
    """

    def setUp(self):
        super().setUp()
        import core.state as state
        import storage as storage_mod
        self.state = state
        self.storage_mod = storage_mod
        self._saved_backend = storage_mod._backend
        self.backend = self.make()
        storage_mod._backend = self.backend
        state.instances.clear()
        state.downloads.clear()

    def tearDown(self):
        self.storage_mod._backend = self._saved_backend
        self.state.instances.clear()
        self.state.downloads.clear()
        super().tearDown()

    @staticmethod
    def _instance(inst_id, port):
        return {
            "id": inst_id, "model_name": "m.gguf", "model_path": "/models/m.gguf",
            "port": port, "status": "healthy", "container_id": "c" + inst_id,
            "container_name": "llama-" + inst_id, "log_file": "", "config": {},
            "started_at": 1.0, "stats": {},
        }

    def recover(self):
        self.primary.down = False
        self.assertTrue(self.backend._try_recover())

    def test_instance_launched_while_degraded_reaches_the_database(self):
        self.trip(self.backend)
        self.state.instances["i1"] = self._instance("i1", 8000)
        self.state.save_state()

        self.primary.down = False
        self.assertEqual(self.primary.load_instances(NODE), [],
                         "must not have reached the database while degraded")

        self.assertTrue(self.backend._try_recover())
        self.assertEqual([r["id"] for r in self.primary.load_instances(NODE)], ["i1"])

    def test_instance_stopped_while_degraded_is_removed_on_resume(self):
        self.state.instances["i1"] = self._instance("i1", 8000)
        self.state.save_state()  # healthy: reaches the database
        self.assertEqual(len(self.primary.load_instances(NODE)), 1)

        self.trip(self.backend)
        del self.state.instances["i1"]
        self.state.save_state()

        self.recover()
        self.assertEqual(self.primary.load_instances(NODE), [],
                         "reconcile replaces this node's rows wholesale")

    def test_a_peers_rows_are_never_touched(self):
        self.primary.save_state([{"id": "peer-inst"}], [], node_id="other-node")
        self.state.instances["i1"] = self._instance("i1", 8000)
        self.trip(self.backend)
        self.state.save_state()
        self.recover()

        peer = self.primary.load_instances("other-node")
        self.assertEqual([r["id"] for r in peer], ["peer-inst"])

    def test_repeated_outages_each_sync_correctly(self):
        """Flapping must not strand a journal or double-apply anything."""
        for round_no, ctx in enumerate((1024, 2048, 4096), start=1):
            self.trip(self.backend)
            self.backend.save_preset("/models/m.gguf", {"ctx_size": ctx})
            self.state.instances[f"i{round_no}"] = self._instance(f"i{round_no}", 8000 + round_no)
            self.state.save_state()
            self.recover()

            self.assertEqual(self.primary.get_preset("/models/m.gguf")["ctx_size"], ctx)
            self.assertEqual(self.backend._journal.pending_count(), 0)
            self.assertEqual(len(self.primary.load_instances(NODE)), round_no)
            self.assertEqual(self.backend._consecutive_failures, 0,
                             "recovery must reset the counter, or the next "
                             "outage would trip early")

    def test_migrations_run_against_the_primary_not_the_wrapper(self):
        """Through the wrapper, get_schema_version() would read the MIRROR's
        recorded version, which can be ahead of what this node applied."""
        import core.migrations as migrations
        seen = []
        real = migrations.run_pending_migrations
        migrations.run_pending_migrations = lambda s: seen.append(s)
        try:
            self.trip(self.backend)
            self.recover()
        finally:
            migrations.run_pending_migrations = real
        self.assertEqual(seen, [self.primary])

    def test_journal_survives_a_process_restart(self):
        """A restart mid-outage must not lose queued changes."""
        self.trip(self.backend)
        self.backend.save_preset("/models/m.gguf", {"ctx_size": 4096})
        self.backend.edit_settings_list("huggingface_tokens", add=[{"id": "t", "token": "x"}])

        # A brand-new wrapper over the same mirror directory == a restart.
        restarted = self.make()
        self.assertEqual(restarted._journal.pending_count(), 2)

        self.primary.down = False
        self.assertTrue(restarted._try_recover())
        self.assertEqual(self.primary.get_preset("/models/m.gguf")["ctx_size"], 4096)
        self.assertEqual(
            [t["id"] for t in self.primary.get_settings()["huggingface_tokens"]], ["t"])


class ToggleTests(ResilientTestCase):
    def _patch(self, value):
        return {"nodes": {NODE: {"db_mirror_enabled": value}}}

    def test_disabling_while_degraded_is_deferred_not_applied(self):
        """Switching off mid-outage would drop the node into pass-through
        against a database it already knows is unreachable - every read would
        raise instead of falling back, and a node that BOOTED degraded has no
        primary object to call at all."""
        backend = self.make()
        self.trip(backend)
        backend.resolve_enabled(self._patch(False))

        self.assertTrue(backend.mirror_enabled(), "disable must be deferred")
        self.assertEqual(backend.get_settings(), {}, "reads must still fall back")

    def test_deferred_disable_applies_after_recovery(self):
        backend = self.make()
        self.trip(backend)
        backend.resolve_enabled(self._patch(False))
        self.assertTrue(backend.mirror_enabled())

        self.primary.down = False
        self.primary.merge_settings(self._patch(False))
        self.assertTrue(backend._try_recover())
        self.assertFalse(backend.mirror_enabled(),
                         "recovery must re-read the toggle it deferred")

    def test_absent_key_never_flips_the_toggle(self):
        """A settings write carrying a partial blob must not switch mirroring
        off underneath a node that has it running."""
        backend = self.make()
        backend.resolve_enabled({"require_auth": False})
        self.assertTrue(backend.mirror_enabled())

    def test_enabling_while_healthy_kicks_an_immediate_full_sync(self):
        """Otherwise the mirror holds only what this node happened to write
        until the next daily sync - up to 24h of near-empty fallback."""
        import threading as _t
        backend = self.make(enabled=False)
        self.primary.save_preset("/models/peer.gguf", {"ctx_size": 512})

        done = _t.Event()
        real = backend.refresh_mirror
        backend.refresh_mirror = lambda: (real(), done.set())[0]

        backend.resolve_enabled(self._patch(True))
        self.assertTrue(done.wait(5), "an initial sync should have been kicked off")
        self.assertEqual(self.mirror.get_preset("/models/peer.gguf")["ctx_size"], 512)

    def test_enabling_while_degraded_starts_the_probe(self):
        backend = ResilientBackend(
            None, self.mirror_dir, node_id=NODE, builder=lambda: self.primary,
            enabled=False, probe_interval=3600, mirror=self.mirror)
        self.assertTrue(backend.is_degraded())
        backend.resolve_enabled(self._patch(True))
        self.assertTrue(backend.mirror_enabled())
        self.assertIsNotNone(backend._probe_thread,
                             "nothing else would ever start recovery")

    def test_merge_that_enables_the_mirror_takes_effect_immediately(self):
        """The write that switches mirroring ON arrives while the wrapper is
        still in pass-through, so pass-through has to re-resolve the toggle. It
        used to return early instead: the setting reached the database but the
        wrapper stayed off until a restart, and the UI - which drives its
        checkbox from the effective state - just unchecked the box again."""
        backend = self.make(enabled=False)

        merged = backend.merge_settings(self._patch(True))

        self.assertTrue(merged["nodes"][NODE]["db_mirror_enabled"],
                        "the setting must still reach the database")
        self.assertTrue(backend.mirror_enabled(),
                        "and must take effect without a restart")
        self.assertTrue(backend.status()["mirror_enabled"],
                        "status() is what the UI reads back")

    def test_enabling_via_merge_seeds_the_mirror(self):
        """Enabling through a merge must seed like any other enable - otherwise
        the directory holds nothing but the owner marker until the daily sync."""
        import threading as _t
        backend = self.make(enabled=False)
        self.primary.save_preset("/models/peer.gguf", {"ctx_size": 512})

        done = _t.Event()
        real = backend.refresh_mirror
        backend.refresh_mirror = lambda: (real(), done.set())[0]

        backend.merge_settings(self._patch(True))

        self.assertTrue(done.wait(5), "an initial sync should have been kicked off")
        self.assertEqual(self.mirror.get_preset("/models/peer.gguf")["ctx_size"], 512)

    def test_unrelated_merge_in_pass_through_does_not_enable_anything(self):
        """The re-resolve must not make pass-through mirror-curious: a settings
        write that says nothing about the toggle leaves it exactly as it was."""
        backend = self.make(enabled=False)
        before = list(self.primary.calls)

        backend.merge_settings({"require_auth": False})

        self.assertFalse(backend.mirror_enabled())
        self.assertEqual(self.mirror.get_settings(), {},
                         "pass-through must still do zero mirror I/O")
        self.assertEqual(self.primary.calls[len(before):], ["merge_settings"],
                         "and exactly one primary call, as before")


class JournalDurabilityTests(ResilientTestCase):
    def test_a_write_is_refused_when_it_cannot_be_journalled(self):
        """Better a visible 503 than silently accepting a change that can never
        reach the database."""
        import storage.resilient as R
        backend = self.make()
        self.trip(backend)
        # Real journal, unwritable path - so the actual OSError -> degraded
        # conversion runs rather than being stubbed out.
        backend._journal = R._Journal("/proc/cannot/exist/journal.jsonl")
        with self.assertRaises(StorageDegradedError):
            backend.save_preset("/models/m.gguf", {"ctx_size": 1})

    def test_journal_oserror_surfaces_as_the_degraded_error(self):
        import storage.resilient as R
        journal = R._Journal("/proc/cannot/exist/journal.jsonl")
        with self.assertRaises(StorageDegradedError):
            journal.append({"op": "preset.delete", "path": "/models/x"})


class ClusterResumeTests(ResilientTestCase):
    """What peers see, and what this node sees of them, across an outage."""

    def test_degraded_node_sees_only_itself(self):
        backend = self.make()
        self.primary.register_node({"node_id": "peer", "node_name": "peer",
                                    "advertise_url": "http://peer:5000"})
        backend.register_node({"node_id": NODE, "node_name": NODE,
                               "advertise_url": "http://self:5000"})
        self.assertEqual(
            {n["node_id"] for n in backend.list_nodes()}, {"peer", NODE})

        self.trip(backend)
        # Peers live only in the shared registry, so a degraded node correctly
        # loses sight of them and serves locally instead of dispatching into
        # the dark.
        self.assertEqual([n["node_id"] for n in backend.list_nodes()], [NODE])

    def test_degraded_node_stops_refreshing_its_own_db_heartbeat(self):
        """This is what makes peers mark it offline and stop routing to it."""
        backend = self.make()
        backend.register_node({"node_id": NODE, "node_name": NODE})
        before = self.primary.get_node(NODE)["last_heartbeat_at"]

        self.trip(backend)
        for _ in range(3):  # heartbeat loop keeps ticking while degraded
            backend.register_node({"node_id": NODE, "node_name": NODE})

        self.primary.down = False
        self.assertEqual(self.primary.get_node(NODE)["last_heartbeat_at"], before,
                         "heartbeat must not advance in the database while degraded")

    def test_peer_registry_survives_the_outage_untouched(self):
        backend = self.make()
        self.primary.register_node({"node_id": "peer", "node_name": "peer",
                                    "advertise_url": "http://peer:5000"})
        self.trip(backend)
        for _ in range(3):
            backend.register_node({"node_id": NODE, "node_name": NODE})

        self.primary.down = False
        self.assertTrue(backend._try_recover())
        peer = self.primary.get_node("peer")
        self.assertIsNotNone(peer, "recovery must not drop peers from the registry")
        self.assertEqual(peer["advertise_url"], "http://peer:5000")

    def test_heartbeat_reaches_the_database_again_after_resume(self):
        backend = self.make()
        self.trip(backend)
        self.primary.down = False
        self.assertTrue(backend._try_recover())

        # The 5s heartbeat loop's next tick now lands in the database, which is
        # what makes peers see this node online again.
        backend.register_node({"node_id": NODE, "node_name": NODE,
                               "advertise_url": "http://self:5000"})
        row = self.primary.get_node(NODE)
        self.assertIsNotNone(row)
        self.assertEqual(row["advertise_url"], "http://self:5000")

    def test_refresh_never_republishes_peer_heartbeats_into_the_mirror(self):
        """JsonBackend.register_node stamps the LOCAL clock, so mirroring peer
        rows would make unreachable peers look freshly alive to _is_online and
        let the dispatcher forward inference to them."""
        backend = self.make()
        self.primary.register_node({"node_id": "peer", "node_name": "peer"})
        self.assertTrue(backend.refresh_mirror())
        self.assertNotIn("peer", {n["node_id"] for n in self.mirror.list_nodes()})


class SettingsKeyIsolationTests(unittest.TestCase):
    """The bug these primitives replace: read whole blob, mutate one key, write
    whole blob back - which drops whatever another writer changed in between."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        d = self._tmp.name
        self.storage = JsonBackend(
            os.path.join(d, "state.json"), os.path.join(d, "presets.json"),
            os.path.join(d, "users.json"), os.path.join(d, "settings.json"),
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_edit_settings_list_does_not_disturb_other_keys(self):
        self.storage.merge_settings({"require_auth": False, "recording_mode": "off"})
        self.storage.edit_settings_list("huggingface_tokens", add=[{"id": "a", "token": "t"}])
        s = self.storage.get_settings()
        self.assertIs(s["require_auth"], False)
        self.assertEqual(s["recording_mode"], "off")
        self.assertEqual([t["id"] for t in s["huggingface_tokens"]], ["a"])

    def test_stale_reader_cannot_clobber_a_concurrent_key(self):
        """A caller holding a pre-edit snapshot must not be able to undo a key
        it never touched - the failure mode of the old save_settings path."""
        self.storage.edit_settings_list("huggingface_tokens", add=[{"id": "a", "token": "t"}])
        stale = self.storage.get_settings()  # snapshot taken before the next write

        # Another writer (another node) changes an unrelated key.
        self.storage.merge_settings({"recording_mode": "per_request"})

        # Our caller now adds a token, holding that stale snapshot.
        self.assertIn("huggingface_tokens", stale)
        self.storage.edit_settings_list("huggingface_tokens", add=[{"id": "b", "token": "u"}])

        s = self.storage.get_settings()
        self.assertEqual(s["recording_mode"], "per_request", "peer's key must survive")
        self.assertEqual(sorted(t["id"] for t in s["huggingface_tokens"]), ["a", "b"])

    def test_add_replaces_an_entry_with_the_same_id(self):
        self.storage.edit_settings_list("huggingface_tokens", add=[{"id": "a", "name": "old"}])
        self.storage.edit_settings_list("huggingface_tokens", add=[{"id": "a", "name": "new"}])
        tokens = self.storage.get_settings()["huggingface_tokens"]
        self.assertEqual(tokens, [{"id": "a", "name": "new"}])

    def test_replace_key_can_remove_entries_a_merge_would_keep(self):
        self.storage.merge_settings({"model_sources": {"/a": {"repo_id": "x"},
                                                       "/b": {"repo_id": "y"}}})
        self.storage.merge_settings({"require_auth": False})
        self.storage.replace_settings_key("model_sources", {"/a": {"repo_id": "x"}})
        s = self.storage.get_settings()
        self.assertEqual(s["model_sources"], {"/a": {"repo_id": "x"}})
        self.assertIs(s["require_auth"], False, "unrelated key must survive")


class PresetDeltaTests(unittest.TestCase):
    def test_delta_round_trips(self):
        base = {"a": 1, "b": 2, "c": 3}
        new = {"a": 1, "b": 99, "d": 4}
        changed, removed = _preset_delta(base, new)
        self.assertEqual(changed, {"b": 99, "d": 4})
        self.assertEqual(sorted(removed), ["c"])
        self.assertEqual(_apply_preset_delta(base, changed, removed), new)

    def test_unset_removes_the_key_on_replay(self):
        self.assertEqual(
            _apply_preset_delta({"a": 1, "b": 2}, {}, ["b"]), {"a": 1})


def _sha(raw):
    import hashlib
    return hashlib.sha256(raw.encode()).hexdigest()


if __name__ == "__main__":
    unittest.main()
