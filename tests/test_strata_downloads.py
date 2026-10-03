# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Pre-downloading Strata's GGUF shards with llamaman's downloader.

The shards come from the Hugging Face revision Strata pins (so the bytes
match what its setup would fetch), land in MODELS_DIR/strata/<tag>/, and are
bind-mounted into the container once complete. Downloads without a revision
must behave exactly as before (resolve/main, same listing URL).
"""

import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from flask import Flask

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import api.downloads as downloads_api
import api.engines as engines_api
import api.instances as instances_api
import core.downloader as downloader
from core.engines import ENGINES
from core.engines import strata as S
from core.state import downloads, downloads_lock, instances, instances_lock

STRATA = ENGINES["strata"]
QWEN = "/strata/qwen-IQ2_XS"
PIN = S.HF_REVISIONS["ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"]
SHARDS = S.repo_files("qwen", "IQ2_XS")


class _Isolated(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.models_dir = self._tmp.name
        with downloads_lock:
            self._saved_dl = dict(downloads)
            downloads.clear()
        with instances_lock:
            self._saved_inst = dict(instances)
            instances.clear()
        self._p = [patch("config.MODELS_DIR", self.models_dir),
                   patch("config.STRATA_ENABLED", True),
                   patch("api.engines.get_vendor", return_value="cuda")]
        for p in self._p:
            p.start()

    def tearDown(self):
        for p in reversed(self._p):
            p.stop()
        with downloads_lock:
            downloads.clear()
            downloads.update(self._saved_dl)
        with instances_lock:
            instances.clear()
            instances.update(self._saved_inst)
        self._tmp.cleanup()

    def _write_shards(self, subdir="IQ2_XS"):
        d = os.path.join(self.models_dir, "strata", "iq2_xs", subdir)
        os.makedirs(d, exist_ok=True)
        for name in S.shard_files("qwen", "IQ2_XS"):
            open(os.path.join(d, name), "w").close()
        return d

    def _add_download(self, status):
        with downloads_lock:
            downloads["dl-1"] = {"id": "dl-1", "repo_id": "x", "status": status,
                                 "engine_model": "strata/qwen-IQ2_XS", "started_at": 1,
                                 "dest_path": os.path.join(self.models_dir, "strata", "iq2_xs")}


class DownloaderRevisionTests(unittest.TestCase):

    def test_default_revision_is_main(self):
        self.assertEqual(downloader.revision, "main")

    def _listing_url(self, **kw):
        resp = Mock(status_code=200)
        resp.json.return_value = {"siblings": []}
        with patch("core.downloader.requests.get", return_value=resp) as get:
            downloader.list_repo_files("org/repo", "", **kw)
        return get.call_args.args[0]

    def test_listing_url_unchanged_without_revision(self):
        self.assertEqual(self._listing_url(), "https://huggingface.co/api/models/org/repo?blobs=true")

    def test_listing_url_with_revision(self):
        self.assertEqual(self._listing_url(rev="abc123"),
                         "https://huggingface.co/api/models/org/repo/revision/abc123?blobs=true")

    def _download_url(self, rev):
        resp = Mock(status_code=416)
        with tempfile.TemporaryDirectory() as d, \
             patch.object(downloader, "local_dir", d), \
             patch.object(downloader, "repo_id", "org/repo"), \
             patch.object(downloader, "revision", rev), \
             patch("core.downloader.requests.get", return_value=resp) as get:
            downloader.download_file("sub/f.gguf")
        return get.call_args.args[0]

    def test_download_url_uses_revision(self):
        self.assertEqual(self._download_url("main"), "https://huggingface.co/org/repo/resolve/main/sub/f.gguf")
        self.assertEqual(self._download_url("abc"), "https://huggingface.co/org/repo/resolve/abc/sub/f.gguf")

    def test_env_carries_revision(self):
        with patch("api.downloads.get_storage") as st:
            st.return_value.get_settings.return_value = {}
            env = downloads_api._build_download_env("r", "/d", "f", "", 0, "", "abc")
            self.assertEqual(env["HF_REVISION"], "abc")
            env = downloads_api._build_download_env("r", "/d", "f", "", 0)
            self.assertEqual(env["HF_REVISION"], "")  # empty = main


class StrataDownloadEndpointTests(_Isolated):

    def setUp(self):
        super().setUp()
        app = Flask(__name__)
        app.register_blueprint(engines_api.bp)
        self.client = app.test_client()

    def _post(self, body):
        listing = [{"name": n, "size": 1, "sha256": ""} for n in SHARDS]
        with patch("api.downloads.list_repo_files", return_value=listing) as list_mock, \
             patch("api.downloads._spawn_download_process",
                   return_value=(Mock(pid=42), Mock(), "/tmp/dl.log")) as spawn, \
             patch("api.downloads.record_model_source") as rec_src, \
             patch("api.downloads.record_model_sha") as rec_sha, \
             patch("api.downloads.save_state"):
            resp = self.client.post("/api/engines/strata/download", json=body)
        return resp, list_mock, spawn, rec_src, rec_sha

    def test_starts_pinned_download_into_strata_dir(self):
        resp, list_mock, spawn, rec_src, rec_sha = self._post({"model": "strata/qwen-IQ2_XS"})
        self.assertEqual(resp.status_code, 201, resp.get_json())
        self.assertEqual(list_mock.call_args.args, ("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", None, PIN))
        args, kw = spawn.call_args
        dl_id, repo, dest, filename, token, mbps = args
        self.assertEqual(repo, "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF")
        self.assertEqual(dest, os.path.join(self.models_dir, "strata", "iq2_xs"))
        self.assertEqual(filename, SHARDS[0])
        self.assertEqual(kw["revision"], PIN)
        # Pinned files must not feed update checks.
        rec_src.assert_not_called()
        rec_sha.assert_not_called()
        body = resp.get_json()
        self.assertEqual(body["engine_model"], "strata/qwen-IQ2_XS")
        self.assertEqual(body["revision"], PIN)

    def test_accepts_path_form(self):
        self.assertEqual(self._post({"model": QWEN})[0].status_code, 201)

    def test_errors(self):
        self.assertEqual(self.client.post("/api/engines/nope/download", json={}).status_code, 404)
        self.assertEqual(self._post({"model": "strata/qwen-IQ9"})[0].status_code, 400)
        self.assertEqual(self.client.post("/api/engines/llamacpp/download",
                                          json={"model": "/models/a.gguf"}).status_code, 400)
        with patch("config.STRATA_ENABLED", False):
            self.assertEqual(self._post({"model": QWEN})[0].status_code, 400)

    def test_refuses_when_already_downloaded(self):
        self._write_shards()
        self.assertEqual(self._post({"model": QWEN})[0].status_code, 409)

    def test_refuses_second_download(self):
        for status in ("downloading", "paused", "failed"):
            self._add_download(status)
            resp = self._post({"model": QWEN})[0]
            self.assertEqual(resp.status_code, 409, status)
            self.assertEqual(resp.get_json()["download_id"], "dl-1")

    def test_ordinary_download_route_unchanged(self):
        # No revision: spawn is called exactly as before (no revision kwarg).
        app = Flask(__name__)
        app.register_blueprint(downloads_api.bp)
        listing = [{"name": "m.gguf", "size": 1, "sha256": "abc"}]
        with patch("api.downloads.list_repo_files", return_value=listing) as list_mock, \
             patch("api.downloads._spawn_download_process",
                   return_value=(Mock(pid=1), Mock(), "/tmp/x.log")) as spawn, \
             patch("api.downloads.record_model_source") as rec_src, \
             patch("api.downloads.record_model_sha"), \
             patch("api.downloads.save_state"), \
             patch.object(downloads_api, "MODELS_DIR", self.models_dir):
            resp = app.test_client().post("/api/downloads", json={"repo_id": "o/r", "filename": "m.gguf"})
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(list_mock.call_args.args, ("o/r", None))
        self.assertEqual(spawn.call_args.kwargs, {"log_mode": "w"})
        rec_src.assert_called_once()
        self.assertNotIn("revision", resp.get_json())
        self.assertNotIn("engine_model", resp.get_json())


class ResumeKeepsRevisionTests(unittest.TestCase):

    def test_restart_passes_saved_revision(self):
        dl = {"id": "d", "repo_id": "r", "dest_path": "/x", "filename": "f", "revision": "abc"}
        with patch("api.downloads._spawn_download_process",
                   return_value=(Mock(), Mock(), "/l")) as spawn:
            downloads_api._restart_existing_download(dl)
        self.assertEqual(spawn.call_args.kwargs["revision"], "abc")

    def test_save_state_writes_revision_only_when_set(self):
        from core import state
        storage = Mock()
        with downloads_lock:
            saved = dict(downloads)
            downloads.clear()
            downloads["a"] = {"id": "a", "repo_id": "r", "status": "completed"}
            downloads["b"] = {"id": "b", "repo_id": "r", "status": "paused",
                              "revision": "abc", "engine_model": "strata/qwen-IQ2_XS"}
        try:
            with patch("storage.get_storage", return_value=storage):
                state.save_state()
        finally:
            with downloads_lock:
                downloads.clear()
                downloads.update(saved)
        rows = {r["id"]: r for r in storage.save_state.call_args.args[1]}
        self.assertNotIn("revision", rows["a"])
        self.assertNotIn("engine_model", rows["a"])
        self.assertEqual(rows["b"]["revision"], "abc")
        self.assertEqual(rows["b"]["engine_model"], "strata/qwen-IQ2_XS")


class ShardReadinessTests(_Isolated):

    def test_complete_files_without_record_are_used(self):
        d = self._write_shards()
        self.assertEqual(S.local_shard_dir("qwen", "IQ2_XS"), d)

    def test_in_flight_download_hides_files(self):
        self._write_shards()
        for status in ("downloading", "paused", "failed", "cancelled"):
            self._add_download(status)
            self.assertIsNone(S.local_shard_dir("qwen", "IQ2_XS"), status)
        self._add_download("completed")
        self.assertIsNotNone(S.local_shard_dir("qwen", "IQ2_XS"))

    def test_catalogue_reports_download(self):
        self._add_download("downloading")
        entry = next(m for m in STRATA.virtual_models() if m["path"] == QWEN)
        self.assertEqual(entry["download"], {"id": "dl-1", "status": "downloading"})
        self.assertFalse(entry["local_shards"])

    def test_launch_blocked_while_downloading(self):
        self._add_download("downloading")
        self.assertIn("still being downloaded", STRATA.launch_blocker(QWEN))
        with patch("api.instances._run_container") as run_mock, \
             patch("api.instances.is_port_available", return_value=True):
            inst, err = instances_api.launch_instance(model_path=QWEN, port=8001, ctx_size=32768)
        self.assertIsNone(inst)
        self.assertIn("still being downloaded", err)
        run_mock.assert_not_called()

    def test_launch_allowed_after_completion_or_without_download(self):
        self.assertIsNone(STRATA.launch_blocker(QWEN))
        self._add_download("completed")
        self.assertIsNone(STRATA.launch_blocker(QWEN))
        self._add_download("failed")  # failed: Strata may fetch the files itself
        self.assertIsNone(STRATA.launch_blocker(QWEN))


if __name__ == "__main__":
    unittest.main()
