# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import hashlib
import os
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import core.model_updates as updates

REMOTE_SHA = "6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83"
OTHER_SHA = "1111111111111111111111111111111111111111111111111111111111111111"


def _head_response(headers, status=200):
    resp = Mock()
    resp.status_code = status
    resp.headers = headers
    resp.raise_for_status = Mock()
    return resp


class RemoteFileInfoTests(unittest.TestCase):
    def test_reads_sha_from_x_linked_etag_not_plain_etag(self):
        """HF returns two different 64-hex values; only x-linked-etag is the
        blob's sha256. Trusting `etag` would flag every model as changed."""
        resp = _head_response({
            "x-linked-etag": f'"{REMOTE_SHA}"',
            "etag": '"7314cd624de8068beee86215e529a23665ff09e458977e32f30b8149764e7be1"',
            "x-linked-size": "807694464",
            "x-repo-commit": "067b946c",
        })
        with patch.object(updates.requests, "head", return_value=resp):
            info = updates.remote_file_info("org/repo", "model.gguf")
        self.assertEqual(info["sha256"], REMOTE_SHA)
        self.assertEqual(info["size"], 807694464)
        self.assertEqual(info["commit"], "067b946c")

    def test_does_not_follow_redirects(self):
        """x-linked-etag/-size/x-repo-commit live on HF's 302, not on the CDN
        response it points to. Following the redirect silently loses all three
        and leaves the hash empty."""
        resp = _head_response({"x-linked-etag": f'"{REMOTE_SHA}"'}, status=302)
        head = Mock(return_value=resp)
        with patch.object(updates.requests, "head", head):
            info = updates.remote_file_info("org/repo", "model.gguf")
        self.assertEqual(info["sha256"], REMOTE_SHA)
        self.assertIs(head.call_args.kwargs["allow_redirects"], False)

    def test_no_lfs_hash_yields_empty_sha(self):
        resp = _head_response({"content-length": "1234"})
        with patch.object(updates.requests, "head", return_value=resp):
            info = updates.remote_file_info("org/repo", "config.json")
        self.assertEqual(info["sha256"], "")
        self.assertEqual(info["size"], 1234)


class CheckModelUpdateTests(unittest.TestCase):
    def _check(self, local_bytes, local_sha, remote):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.gguf")
            with open(path, "wb") as f:
                f.write(local_bytes)
            with patch.object(updates, "remote_file_info", return_value=remote), \
                 patch("core.model_sources.get_model_sha", return_value=local_sha):
                return updates.check_model_update(path, "org/repo")

    def test_matching_hash_is_up_to_date(self):
        r = self._check(b"x" * 10, REMOTE_SHA, {"sha256": REMOTE_SHA, "size": 10, "commit": ""})
        self.assertEqual(r["status"], updates.STATUS_UP_TO_DATE)

    def test_differing_hash_is_an_update(self):
        r = self._check(b"x" * 10, OTHER_SHA, {"sha256": REMOTE_SHA, "size": 10, "commit": ""})
        self.assertEqual(r["status"], updates.STATUS_UPDATE_AVAILABLE)

    def test_matching_hash_but_wrong_size_is_not_up_to_date(self):
        """The hash is stamped when a download starts, so a pull that died
        partway must not be reported as current."""
        r = self._check(b"x" * 5, REMOTE_SHA, {"sha256": REMOTE_SHA, "size": 10, "commit": ""})
        self.assertEqual(r["status"], updates.STATUS_UPDATE_AVAILABLE)
        self.assertIn("incomplete", r["detail"])

    def test_no_stored_hash_and_differing_size_is_an_update(self):
        r = self._check(b"x" * 5, "", {"sha256": REMOTE_SHA, "size": 10, "commit": ""})
        self.assertEqual(r["status"], updates.STATUS_UPDATE_AVAILABLE)

    def test_no_stored_hash_and_same_size_is_unverified(self):
        r = self._check(b"x" * 10, "", {"sha256": REMOTE_SHA, "size": 10, "commit": ""})
        self.assertEqual(r["status"], updates.STATUS_UNVERIFIED)

    def test_model_without_repo_reports_no_repo(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.gguf")
            with open(path, "wb") as f:
                f.write(b"x")
            r = updates.check_model_update(path, "")
        self.assertEqual(r["status"], updates.STATUS_NO_REPO)

    def test_network_failure_is_unknown_not_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.gguf")
            with open(path, "wb") as f:
                f.write(b"x")
            with patch.object(updates, "remote_file_info", side_effect=RuntimeError("boom")):
                r = updates.check_model_update(path, "org/repo")
        self.assertEqual(r["status"], updates.STATUS_UNKNOWN)
        self.assertIn("boom", r["detail"])


class SwapInUpdateTests(unittest.TestCase):
    def test_replaces_existing_file_and_removes_temp(self):
        with tempfile.TemporaryDirectory() as d:
            live = os.path.join(d, "model.gguf")
            with open(live, "wb") as f:
                f.write(b"old")
            temp = updates.update_temp_dir(live)
            os.makedirs(temp)
            with open(os.path.join(temp, "model.gguf"), "wb") as f:
                f.write(b"new")

            moved, err = updates.swap_in_update(temp, d)

        self.assertIsNone(err)
        self.assertEqual(moved, ["model.gguf"])

    def test_swaps_every_shard_of_a_multipart_model(self):
        with tempfile.TemporaryDirectory() as d:
            for i in (1, 2):
                with open(os.path.join(d, f"m-0000{i}-of-00002.gguf"), "wb") as f:
                    f.write(b"old")
            temp = updates.update_temp_dir(os.path.join(d, "m-00001-of-00002.gguf"))
            os.makedirs(temp)
            for i in (1, 2):
                with open(os.path.join(temp, f"m-0000{i}-of-00002.gguf"), "wb") as f:
                    f.write(b"new")

            moved, err = updates.swap_in_update(temp, d)

            self.assertIsNone(err)
            self.assertEqual(sorted(moved), ["m-00001-of-00002.gguf", "m-00002-of-00002.gguf"])
            for i in (1, 2):
                with open(os.path.join(d, f"m-0000{i}-of-00002.gguf"), "rb") as f:
                    self.assertEqual(f.read(), b"new")
            self.assertFalse(os.path.exists(temp))

    def test_empty_staging_dir_is_an_error_and_leaves_the_model_alone(self):
        with tempfile.TemporaryDirectory() as d:
            live = os.path.join(d, "model.gguf")
            with open(live, "wb") as f:
                f.write(b"old")
            temp = updates.update_temp_dir(live)
            os.makedirs(temp)

            moved, err = updates.swap_in_update(temp, d)

            self.assertEqual(moved, [])
            self.assertIsNotNone(err)
            with open(live, "rb") as f:
                self.assertEqual(f.read(), b"old")

    def test_temp_dir_is_nested_in_the_model_dir(self):
        """Same filesystem is what makes the swap an atomic replace."""
        temp = updates.update_temp_dir("/models/foo/model.gguf")
        self.assertEqual(os.path.dirname(temp), "/models/foo")


class LocalHashTests(unittest.TestCase):
    """Hashing the file on disk - the cheap way to resolve `unverified`."""

    def _hash_and_wait(self, path, stamp):
        with patch("core.model_sources.record_model_sha", stamp):
            updates.start_local_hash(path)
            for _ in range(200):
                state = updates.local_hash_state(path)
                if state and state["status"] != "hashing":
                    return state
                time.sleep(0.02)
        self.fail("hashing did not finish")

    def test_computes_the_real_sha256_and_stamps_it(self):
        payload = b"llamaman" * 1000
        expected = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.gguf")
            with open(path, "wb") as f:
                f.write(payload)
            stamp = Mock()
            state = self._hash_and_wait(path, stamp)

        self.assertEqual(state["status"], "done")
        self.assertEqual(state["sha256"], expected)
        self.assertEqual(state["hashed_bytes"], len(payload))
        # Stamped, so every later check is exact and instant.
        stamp.assert_called_once_with(path, expected)

    def test_reports_total_size_for_progress(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.gguf")
            with open(path, "wb") as f:
                f.write(b"x" * 4096)
            state = self._hash_and_wait(path, Mock())
        self.assertEqual(state["total_bytes"], 4096)

    def test_second_request_rejoins_instead_of_rereading(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.gguf")
            with open(path, "wb") as f:
                f.write(b"x" * 2048)
            with patch("core.model_sources.record_model_sha", Mock()):
                first = updates.start_local_hash(path)
                second = updates.start_local_hash(path)
                self.assertEqual(first["total_bytes"], second["total_bytes"])
                for _ in range(200):
                    if updates.local_hash_state(path)["status"] != "hashing":
                        break
                    time.sleep(0.02)

    def test_missing_file_reports_an_error(self):
        state = updates.start_local_hash("/definitely/not/here.gguf")
        self.assertEqual(state["status"], "error")

    def test_a_hashed_model_then_compares_exactly(self):
        """End state of the verify flow: no longer `unverified`."""
        payload = b"z" * 512
        sha = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.gguf")
            with open(path, "wb") as f:
                f.write(payload)
            remote = {"sha256": sha, "size": len(payload), "commit": ""}
            with patch.object(updates, "remote_file_info", return_value=remote), \
                 patch("core.model_sources.get_model_sha", return_value=sha):
                result = updates.check_model_update(path, "org/repo")
        self.assertEqual(result["status"], updates.STATUS_UP_TO_DATE)


class CompletionHookTests(unittest.TestCase):
    """core.monitoring._finish_model_update - what runs when the pull exits 0."""

    def test_ordinary_download_is_untouched(self):
        import core.monitoring as monitoring
        self.assertIsNone(monitoring._finish_model_update(
            {"id": "d1", "repo_id": "org/repo", "status": "completed"}))

    def test_swaps_staged_file_in_and_stamps_the_new_hash(self):
        import core.monitoring as monitoring
        with tempfile.TemporaryDirectory() as d:
            live = os.path.join(d, "m.gguf")
            with open(live, "wb") as f:
                f.write(b"old")
            temp = updates.update_temp_dir(live)
            os.makedirs(temp)
            with open(os.path.join(temp, "m.gguf"), "wb") as f:
                f.write(b"new")

            with patch("core.model_sources.record_model_sha") as stamp:
                err = monitoring._finish_model_update({
                    "update_model_path": live,
                    "update_temp_dir": temp,
                    "update_sha256": REMOTE_SHA,
                })

            self.assertIsNone(err)
            with open(live, "rb") as f:
                self.assertEqual(f.read(), b"new")
            self.assertFalse(os.path.exists(temp))
            stamp.assert_called_once_with(live, REMOTE_SHA)

    def test_failed_swap_reports_an_error_and_keeps_the_staged_files(self):
        """The download is marked failed rather than completed, so the staged
        bytes stay on disk for a retry instead of being silently discarded."""
        import core.monitoring as monitoring
        with tempfile.TemporaryDirectory() as d:
            live = os.path.join(d, "m.gguf")
            with open(live, "wb") as f:
                f.write(b"old")
            temp = updates.update_temp_dir(live)
            os.makedirs(temp)  # empty: nothing was actually downloaded

            err = monitoring._finish_model_update({
                "update_model_path": live,
                "update_temp_dir": temp,
                "update_sha256": REMOTE_SHA,
            })

            self.assertIsNotNone(err)
            with open(live, "rb") as f:
                self.assertEqual(f.read(), b"old")


class AutoScanTests(unittest.TestCase):
    """Opt-in background scan (Settings -> Downloads)."""

    def setUp(self):
        updates._last_scan_status.clear()

    def _scan(self, models_dir, sources, shas, *, downloading=False):
        import api.models as models_api
        with patch.object(models_api, "MODELS_DIR", models_dir), \
             patch("config.MODELS_DIR", models_dir), \
             patch.object(updates, "_downloads_active", return_value=downloading), \
             patch("core.model_sources.get_model_sources", return_value=sources), \
             patch("core.model_sources.get_model_sha", side_effect=lambda p: shas.get(p, "")), \
             patch.object(updates, "start_local_hash") as start_hash, \
             patch.object(updates, "remote_file_info",
                          return_value={"sha256": REMOTE_SHA, "size": 4, "commit": ""}):
            return updates.run_auto_update_scan(), start_hash

    def _models(self, d, *names):
        for n in names:
            with open(os.path.join(d, n), "wb") as f:
                f.write(b"gguf")
        return [os.path.join(d, n) for n in names]

    def test_skips_entirely_while_a_download_is_running(self):
        """Hashing competes with a download for the same disk."""
        with tempfile.TemporaryDirectory() as d:
            self._models(d, "a.gguf")
            summary, start_hash = self._scan(d, {d: "org/repo"}, {}, downloading=True)
        self.assertEqual(summary, {"skipped": "download in progress"})
        start_hash.assert_not_called()

    def test_ignores_models_with_no_known_repo(self):
        with tempfile.TemporaryDirectory() as d:
            self._models(d, "a.gguf")
            summary, start_hash = self._scan(d, {}, {})
        self.assertEqual(summary["checked"], 0)
        start_hash.assert_not_called()

    def test_hashes_at_most_one_model_per_pass(self):
        """Three models missing hashes must not kick off three full disk reads."""
        with tempfile.TemporaryDirectory() as d:
            self._models(d, "a.gguf", "b.gguf", "c.gguf")
            summary, start_hash = self._scan(d, {d: "org/repo"}, {})
        self.assertEqual(summary["checked"], 3)
        self.assertEqual(summary["awaiting_hash"], 3)
        self.assertEqual(start_hash.call_count, 1)

    def test_does_not_rehash_models_that_already_have_one(self):
        with tempfile.TemporaryDirectory() as d:
            (a,) = self._models(d, "a.gguf")
            summary, start_hash = self._scan(d, {d: "org/repo"}, {a: REMOTE_SHA})
        self.assertEqual(summary["awaiting_hash"], 0)
        start_hash.assert_not_called()

    def test_records_a_verdict_per_model_for_the_ui(self):
        with tempfile.TemporaryDirectory() as d:
            (a,) = self._models(d, "a.gguf")
            self._scan(d, {d: "org/repo"}, {a: OTHER_SHA})
            status = updates.last_scan_status()
        self.assertEqual(status[a]["status"], updates.STATUS_UPDATE_AVAILABLE)
        self.assertEqual(status[a]["repo_id"], "org/repo")

    def test_disabled_by_default(self):
        storage = Mock()
        storage.get_settings.return_value = {}
        with patch("storage.get_storage", return_value=storage):
            enabled, interval = updates.auto_scan_settings()
        self.assertFalse(enabled)
        self.assertEqual(interval, updates.AUTO_SCAN_DEFAULT_INTERVAL_HOURS * 3600)

    def test_interval_comes_from_settings(self):
        storage = Mock()
        storage.get_settings.return_value = {
            "auto_update_scan_enabled": True, "auto_update_scan_interval_hours": 6}
        with patch("storage.get_storage", return_value=storage):
            enabled, interval = updates.auto_scan_settings()
        self.assertTrue(enabled)
        self.assertEqual(interval, 6 * 3600)

    def test_unreadable_settings_leave_it_off(self):
        storage = Mock()
        storage.get_settings.side_effect = RuntimeError("db down")
        with patch("storage.get_storage", return_value=storage):
            enabled, _ = updates.auto_scan_settings()
        self.assertFalse(enabled)


class DeleteDuringHashTests(unittest.TestCase):
    """C5: deleting a model out from under a running hash."""

    def test_delete_is_refused_while_the_model_is_being_hashed(self):
        from flask import Flask
        import api.models as models_api
        app = Flask(__name__)
        app.register_blueprint(models_api.bp)
        client = app.test_client()

        with tempfile.TemporaryDirectory() as models_dir:
            live = os.path.join(models_dir, "m.gguf")
            with open(live, "wb") as f:
                f.write(b"gguf")
            with patch.object(models_api, "MODELS_DIR", models_dir), \
                 patch.object(models_api, "instances", {}), \
                 patch.object(updates, "local_hash_state",
                              return_value={"status": "hashing"}):
                resp = client.post("/api/models/delete", json={"path": live})

            self.assertEqual(resp.status_code, 409)
            self.assertIn("hashed", resp.get_json()["error"])
            self.assertTrue(os.path.exists(live), "model was deleted despite the guard")

    def test_delete_proceeds_when_no_hash_is_running(self):
        from flask import Flask
        import api.models as models_api
        app = Flask(__name__)
        app.register_blueprint(models_api.bp)
        client = app.test_client()

        with tempfile.TemporaryDirectory() as models_dir:
            live = os.path.join(models_dir, "m.gguf")
            with open(live, "wb") as f:
                f.write(b"gguf")
            with patch.object(models_api, "MODELS_DIR", models_dir), \
                 patch.object(models_api, "instances", {}), \
                 patch.object(updates, "local_hash_state", return_value=None), \
                 patch.object(models_api, "remove_model_sources_for_path"):
                resp = client.post("/api/models/delete", json={"path": live})

            self.assertEqual(resp.status_code, 200)
            self.assertFalse(os.path.exists(live))


class UpdateEndpointTests(unittest.TestCase):
    """The /api/models/update route: staging setup and the in-use guard."""

    def _app(self):
        from flask import Flask
        import api.models as models_api
        app = Flask(__name__)
        app.register_blueprint(models_api.bp)
        return app, models_api

    def _settings_storage(self, model_dir, repo_id="org/repo"):
        storage = Mock()
        storage.get_settings.return_value = {
            "model_sources": {model_dir: {"repo_id": repo_id}},
        }
        return storage

    def test_starts_a_staging_download_without_touching_the_live_file(self):
        import api.downloads as downloads_api
        app, models_api = self._app()
        client = app.test_client()

        with tempfile.TemporaryDirectory() as models_dir:
            model_dir = os.path.join(models_dir, "m")
            os.makedirs(model_dir)
            live = os.path.join(model_dir, "m.gguf")
            with open(live, "wb") as f:
                f.write(b"old")

            storage = self._settings_storage(model_dir)
            proc, log_fh = Mock(pid=4321), Mock()
            with patch.object(models_api, "MODELS_DIR", models_dir), \
                 patch.object(models_api, "get_storage", return_value=storage), \
                 patch.object(models_api, "instances", {}), \
                 patch("core.downloader.list_repo_files",
                       return_value=[{"name": "m.gguf", "size": 99, "sha256": REMOTE_SHA}]), \
                 patch.object(downloads_api, "_spawn_download_process",
                              return_value=(proc, log_fh, "/tmp/dl.log")), \
                 patch.object(downloads_api, "save_state"):
                resp = client.post("/api/models/update", json={"path": live})

            self.assertEqual(resp.status_code, 201)
            body = resp.get_json()
            # Staged inside the model's own dir, and the live file is untouched.
            self.assertEqual(body["update_temp_dir"], updates.update_temp_dir(live))
            self.assertEqual(body["dest_path"], updates.update_temp_dir(live))
            self.assertEqual(body["update_sha256"], REMOTE_SHA)
            with open(live, "rb") as f:
                self.assertEqual(f.read(), b"old")

    def test_refuses_while_an_instance_has_the_model_loaded(self):
        app, models_api = self._app()
        client = app.test_client()

        with tempfile.TemporaryDirectory() as models_dir:
            model_dir = os.path.join(models_dir, "m")
            os.makedirs(model_dir)
            live = os.path.join(model_dir, "m.gguf")
            with open(live, "wb") as f:
                f.write(b"old")

            running = {"i1": {"status": "healthy", "model_path": live, "port": 8000}}
            with patch.object(models_api, "MODELS_DIR", models_dir), \
                 patch.object(models_api, "get_storage",
                              return_value=self._settings_storage(model_dir)), \
                 patch.object(models_api, "instances", running):
                resp = client.post("/api/models/update", json={"path": live})

        self.assertEqual(resp.status_code, 409)
        self.assertIn("in use", resp.get_json()["error"])

    def test_refuses_a_model_with_no_known_repo(self):
        app, models_api = self._app()
        client = app.test_client()

        with tempfile.TemporaryDirectory() as models_dir:
            live = os.path.join(models_dir, "m.gguf")
            with open(live, "wb") as f:
                f.write(b"old")
            storage = Mock()
            storage.get_settings.return_value = {}
            with patch.object(models_api, "MODELS_DIR", models_dir), \
                 patch.object(models_api, "get_storage", return_value=storage), \
                 patch.object(models_api, "instances", {}):
                resp = client.post("/api/models/update", json={"path": live})

        self.assertEqual(resp.status_code, 400)

    def test_get_verify_hash_reports_without_starting_a_job(self):
        """Polling and model-selection use GET. If GET started a hash, merely
        selecting a model - or a peer being asked about one - would kick off a
        multi-minute read of it."""
        app, models_api = self._app()
        client = app.test_client()
        with tempfile.TemporaryDirectory() as models_dir:
            live = os.path.join(models_dir, "m.gguf")
            with open(live, "wb") as f:
                f.write(b"x" * 32)
            with patch.object(models_api, "MODELS_DIR", models_dir), \
                 patch.object(updates, "start_local_hash") as start:
                resp = client.get("/api/models/verify-hash", query_string={"path": live})

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["status"], "none")
        start.assert_not_called()

    def test_post_verify_hash_starts_the_job(self):
        app, models_api = self._app()
        client = app.test_client()
        with tempfile.TemporaryDirectory() as models_dir:
            live = os.path.join(models_dir, "m.gguf")
            with open(live, "wb") as f:
                f.write(b"x" * 32)
            with patch.object(models_api, "MODELS_DIR", models_dir), \
                 patch.object(updates, "start_local_hash",
                              return_value={"status": "hashing"}) as start:
                resp = client.post("/api/models/verify-hash", json={"path": live})

        self.assertEqual(resp.status_code, 200)
        start.assert_called_once()

    def test_rejects_a_path_outside_the_models_dir(self):
        app, models_api = self._app()
        client = app.test_client()
        with tempfile.TemporaryDirectory() as models_dir:
            with patch.object(models_api, "MODELS_DIR", models_dir):
                resp = client.post("/api/models/update", json={"path": "/etc/passwd"})
        self.assertEqual(resp.status_code, 403)


if __name__ == "__main__":
    unittest.main()
