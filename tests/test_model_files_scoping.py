# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Per-node model file metadata: node scoping, and the upgrade/rollback path.

The guarantee under test is that moving repo_id/sha256 to a node-scoped store
did not break a cluster running mixed versions, in either direction.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import core.model_sources as ms
from storage.json_backend import JsonBackend

MODEL = "/models/shared/m.gguf"
SHA_A = "a" * 64
SHA_B = "b" * 64


class BackendModelFileTests(unittest.TestCase):
    """The storage primitive itself."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        d = self._tmp.name
        self.storage = JsonBackend(
            os.path.join(d, "state.json"), os.path.join(d, "presets.json"),
            os.path.join(d, "users.json"), os.path.join(d, "settings.json"),
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_same_path_on_two_nodes_does_not_collide(self):
        """The whole point: one path, two nodes, two different files."""
        self.storage.upsert_model_file("srv1", MODEL, sha256=SHA_A)
        self.storage.upsert_model_file("srv2", MODEL, sha256=SHA_B)
        self.assertEqual(self.storage.get_model_files("srv1")[MODEL]["sha256"], SHA_A)
        self.assertEqual(self.storage.get_model_files("srv2")[MODEL]["sha256"], SHA_B)

    def test_unknown_node_reads_empty(self):
        self.assertEqual(self.storage.get_model_files("nobody"), {})

    def test_hash_stamp_does_not_clear_repo_id(self):
        self.storage.upsert_model_file("srv1", MODEL, repo_id="org/repo")
        self.storage.upsert_model_file("srv1", MODEL, sha256=SHA_A)
        row = self.storage.get_model_files("srv1")[MODEL]
        self.assertEqual(row["repo_id"], "org/repo")
        self.assertEqual(row["sha256"], SHA_A)

    def test_repo_id_write_does_not_clear_hash(self):
        self.storage.upsert_model_file("srv1", MODEL, sha256=SHA_A)
        self.storage.upsert_model_file("srv1", MODEL, repo_id="org/repo")
        row = self.storage.get_model_files("srv1")[MODEL]
        self.assertEqual(row["sha256"], SHA_A)

    def test_delete_removes_directory_contents_for_that_node_only(self):
        self.storage.upsert_model_file("srv1", "/models/x/a.gguf", sha256=SHA_A)
        self.storage.upsert_model_file("srv1", "/models/x/sub/b.gguf", sha256=SHA_A)
        self.storage.upsert_model_file("srv1", "/models/keep/c.gguf", sha256=SHA_A)
        self.storage.upsert_model_file("srv2", "/models/x/a.gguf", sha256=SHA_B)

        self.storage.delete_model_files("srv1", "/models/x")

        self.assertEqual(sorted(self.storage.get_model_files("srv1")), ["/models/keep/c.gguf"])
        # srv2 still owns its file at the same path.
        self.assertIn("/models/x/a.gguf", self.storage.get_model_files("srv2"))

    def test_delete_of_unknown_path_is_a_noop(self):
        self.storage.upsert_model_file("srv1", MODEL, sha256=SHA_A)
        self.storage.delete_model_files("srv1", "/models/other")
        self.assertIn(MODEL, self.storage.get_model_files("srv1"))


class UpgradeAndRollbackTests(unittest.TestCase):
    """core.model_sources: node rows authoritative, legacy blob as fallback."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        d = self._tmp.name
        self.storage = JsonBackend(
            os.path.join(d, "state.json"), os.path.join(d, "presets.json"),
            os.path.join(d, "users.json"), os.path.join(d, "settings.json"),
        )
        self._ctx = [
            patch.object(ms, "get_storage", return_value=self.storage),
            patch.object(ms, "_local_node_id", return_value="srv1"),
            # Paths in these tests are notional; keep realpath from rewriting them.
            patch.object(ms, "normalize_model_source_path", side_effect=lambda p: p or ""),
        ]
        for c in self._ctx:
            c.start()

    def tearDown(self):
        for c in self._ctx:
            c.stop()
        self._tmp.cleanup()

    def test_legacy_blob_is_still_readable_after_upgrade(self):
        """A model recorded by the old code must not lose its repo_id."""
        self.storage.merge_settings({"model_sources": {MODEL: {"repo_id": "org/legacy"}}})
        self.assertEqual(ms.get_model_sources().get(MODEL), "org/legacy")

    def test_legacy_hash_is_still_readable_after_upgrade(self):
        """Hashes stamped by this feature's first iteration lived in settings;
        reading them back avoids forcing a re-hash of every model."""
        self.storage.merge_settings({"model_sources": {MODEL: {"sha256": SHA_A}}})
        self.assertEqual(ms.get_model_sha(MODEL), SHA_A)

    def test_node_row_wins_over_legacy_blob(self):
        self.storage.merge_settings({"model_sources": {MODEL: {"repo_id": "org/legacy"}}})
        self.storage.upsert_model_file("srv1", MODEL, repo_id="org/mine")
        self.assertEqual(ms.get_model_sources().get(MODEL), "org/mine")

    def test_download_writes_only_the_node_scoped_store(self):
        """Post-004 the shared blob is no longer written - that shared row is
        what let two nodes disagree about the same path."""
        ms.record_model_source("/models/m", "org/repo", model_path=MODEL)
        self.assertEqual(self.storage.get_model_files("srv1")[MODEL]["repo_id"], "org/repo")
        self.assertEqual(self.storage.get_model_files("srv1")["/models/m"]["repo_id"], "org/repo")
        self.assertNotIn("model_sources", self.storage.get_settings())

    def test_hash_stamp_never_touches_the_shared_settings_blob(self):
        """Hashes are per-node and written often; keeping them out of the
        single read-modify-write settings row is what avoids lost updates."""
        before = self.storage.get_settings()
        ms.record_model_sha(MODEL, SHA_A)
        self.assertEqual(self.storage.get_settings(), before)
        self.assertEqual(self.storage.get_model_files("srv1")[MODEL]["sha256"], SHA_A)

    def test_two_nodes_disagree_without_clobbering_each_other(self):
        """The bug this whole change exists to fix."""
        ms.record_model_sha(MODEL, SHA_A)                      # as srv1
        self.storage.upsert_model_file("srv2", MODEL, sha256=SHA_B)
        self.assertEqual(ms.get_model_sha(MODEL), SHA_A)
        with patch.object(ms, "_local_node_id", return_value="srv2"):
            self.assertEqual(ms.get_model_sha(MODEL), SHA_B)

    def test_delete_clears_this_nodes_rows_and_legacy_entries(self):
        ms.record_model_source("/models/x", "org/repo", model_path="/models/x/a.gguf")
        # A leftover legacy entry from before the upgrade must be cleaned up too.
        self.storage.merge_settings({"model_sources": {"/models/x/a.gguf": {"repo_id": "org/old"}}})
        self.storage.upsert_model_file("srv2", "/models/x/a.gguf", sha256=SHA_B)

        ms.remove_model_sources_for_path("/models/x")

        self.assertEqual(self.storage.get_model_files("srv1"), {})
        self.assertEqual(self.storage.get_settings().get("model_sources"), {})
        # The peer's row for its own copy survives.
        self.assertIn("/models/x/a.gguf", self.storage.get_model_files("srv2"))

    def test_storage_failure_falls_back_instead_of_raising(self):
        """A node-row read that blows up must degrade to the legacy blob, not
        fail the request that triggered it."""
        self.storage.merge_settings({"model_sources": {MODEL: {"repo_id": "org/legacy"}}})
        with patch.object(self.storage, "get_model_files", side_effect=RuntimeError("db down")):
            self.assertEqual(ms.get_model_sources().get(MODEL), "org/legacy")


class PerNodeSchemaVersionTests(unittest.TestCase):
    """A single cluster-wide version meant nodes 2..N skipped every migration."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        d = self._tmp.name
        self.storage = JsonBackend(
            os.path.join(d, "state.json"), os.path.join(d, "presets.json"),
            os.path.join(d, "users.json"), os.path.join(d, "settings.json"),
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _as(self, node_id):
        return patch.object(type(self.storage), "_node_key", return_value=node_id)

    def test_each_node_tracks_its_own_version(self):
        with self._as("srv1"):
            self.storage.set_schema_version(4)
        with self._as("srv2"):
            self.assertLess(self.storage.get_schema_version(), 4,
                            "srv2 inherited srv1's version and would skip its migrations")
        with self._as("srv1"):
            self.assertEqual(self.storage.get_schema_version(), 4)

    def test_existing_install_does_not_replay_old_global_migrations(self):
        """A pre-per-node install recorded version 3 globally; that node must
        not run migrations 1-3 again just because the key moved."""
        self.storage.merge_settings({"_schema_version": 3})
        with self._as("srv1"):
            self.assertEqual(self.storage.get_schema_version(), 3)

    def test_global_version_is_capped_so_per_node_migrations_still_run(self):
        """The old global key only ever covered global work. A node inheriting
        it must still run everything from 4 on for itself."""
        self.storage.merge_settings({"_schema_version": 9})
        with self._as("fresh-node"):
            self.assertEqual(self.storage.get_schema_version(),
                             self.storage.LEGACY_GLOBAL_MAX_VERSION)

    def test_unmigrated_node_starts_at_zero(self):
        with self._as("brand-new"):
            self.assertEqual(self.storage.get_schema_version(), 0)


class AdoptLegacyModelSourcesTests(unittest.TestCase):
    """Migration 004: each node claims only what is on its own disk."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        d = self._tmp.name
        self.models = os.path.join(d, "models")
        os.makedirs(self.models)
        self.storage = JsonBackend(
            os.path.join(d, "state.json"), os.path.join(d, "presets.json"),
            os.path.join(d, "users.json"), os.path.join(d, "settings.json"),
        )
        self._ctx = [
            patch.object(ms, "get_storage", return_value=self.storage),
            patch.object(ms, "_local_node_id", return_value="srv1"),
        ]
        for c in self._ctx:
            c.start()

    def tearDown(self):
        for c in self._ctx:
            c.stop()
        self._tmp.cleanup()

    def _model(self, name):
        p = os.path.join(self.models, name)
        with open(p, "wb") as f:
            f.write(b"gguf")
        return os.path.realpath(p)

    def test_adopts_only_paths_present_on_this_nodes_disk(self):
        mine = self._model("mine.gguf")
        self.storage.merge_settings({"model_sources": {
            mine: {"repo_id": "org/mine", "sha256": SHA_A},
            "/models/on-another-node/theirs.gguf": {"repo_id": "org/theirs"},
        }})

        self.assertEqual(ms.adopt_legacy_model_sources(), 1)

        rows = self.storage.get_model_files("srv1")
        self.assertEqual(list(rows), [mine])
        self.assertEqual(rows[mine], {"repo_id": "org/mine", "sha256": SHA_A})

    def test_leaves_the_legacy_blob_intact_for_rollback(self):
        mine = self._model("mine.gguf")
        self.storage.merge_settings({"model_sources": {mine: {"repo_id": "org/mine"}}})
        ms.adopt_legacy_model_sources()
        self.assertEqual(
            self.storage.get_settings()["model_sources"][mine]["repo_id"], "org/mine")

    def test_is_idempotent_and_never_overwrites_a_live_row(self):
        mine = self._model("mine.gguf")
        self.storage.merge_settings({"model_sources": {mine: {"sha256": SHA_A}}})
        # A hash recorded since the upgrade must win over the stale blob value.
        self.storage.upsert_model_file("srv1", mine, sha256=SHA_B)

        self.assertEqual(ms.adopt_legacy_model_sources(), 0)
        self.assertEqual(self.storage.get_model_files("srv1")[mine]["sha256"], SHA_B)

    def test_no_legacy_data_is_a_noop(self):
        self.assertEqual(ms.adopt_legacy_model_sources(), 0)


if __name__ == "__main__":
    unittest.main()
