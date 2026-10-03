# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Model archive (core/archive.py, api/archive.py): moving models between
MODELS_DIR and ARCHIVE_DIR without ever losing data.

Pinned here: what a model "unit" is, the rename and copy paths (verify before
deleting the source), cancel and failure leave the source untouched, stale
partial copies are cleaned up, and the API refuses moves of models an
instance or download depends on.
"""

import os
import tempfile
import time
import unittest
from unittest.mock import patch

from flask import Flask

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import api.archive as archive_api
import api.instances as instances_api
import api.models as models_api
from core import archive
from core.archive import ArchiveError
from core.state import downloads, downloads_lock, instances, instances_lock


def _write(path, size=1000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(os.urandom(size))


def _job(src, dst, unit, job_id="j1"):
    import threading
    return {"id": job_id, "src_base": src, "dst_base": dst, "unit": unit,
            "bytes_total": archive.unit_size(src, unit), "bytes_done": 0, "speed": 0.0,
            "method": None, "warning": None, "_cancel": threading.Event()}


class _Dirs(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.models = os.path.join(self._tmp.name, "models")
        self.arch = os.path.join(self._tmp.name, "archive")
        os.makedirs(self.models)
        os.makedirs(self.arch)
        self._p = [patch("config.MODELS_DIR", self.models),
                   patch("config.ARCHIVE_DIR", self.arch),
                   patch("api.archive.MODELS_DIR", self.models)]
        for p in self._p:
            p.start()
        with archive.jobs_lock:
            self._saved_jobs = dict(archive.jobs)
            archive.jobs.clear()

    def tearDown(self):
        # Let a running move finish before the directories vanish.
        deadline = time.time() + 10
        while time.time() < deadline and any(j["status"] in archive.ACTIVE_STATUSES
                                             for j in archive.jobs.values()):
            time.sleep(0.02)
        with archive.jobs_lock:
            archive.jobs.clear()
            archive.jobs.update(self._saved_jobs)
        for p in reversed(self._p):
            p.stop()
        self._tmp.cleanup()

    def wait(self, job_id, timeout=10):
        deadline = time.time() + timeout
        while time.time() < deadline:
            with archive.jobs_lock:
                job = dict(archive.jobs[job_id])
            if job["status"] not in archive.ACTIVE_STATUSES:
                return job
            time.sleep(0.02)
        self.fail(f"job {job_id} did not finish")


class StatusTests(_Dirs):

    def test_disabled_without_archive_dir(self):
        with patch("config.ARCHIVE_DIR", ""):
            st = archive.status()
        self.assertEqual((st["enabled"], st["available"]), (False, False))

    def test_missing_dir_is_enabled_but_unavailable(self):
        with patch("config.ARCHIVE_DIR", os.path.join(self.arch, "nope")):
            st = archive.status()
        self.assertTrue(st["enabled"])
        self.assertFalse(st["available"])
        self.assertIn("does not exist", st["reason"])

    def test_available(self):
        st = archive.status()
        self.assertTrue(st["available"])
        self.assertIsInstance(st["free_bytes"], int)


class UnitTests(_Dirs):

    def test_folder_is_the_unit(self):
        p = os.path.join(self.models, "Qwen-UD-Q4", "Qwen-UD-Q4-00001-of-00002.gguf")
        _write(p)
        _write(os.path.join(self.models, "Qwen-UD-Q4", "Qwen-UD-Q4-00002-of-00002.gguf"))
        self.assertEqual(archive.model_unit(p, self.models), ["Qwen-UD-Q4"])
        deep = os.path.join(self.models, "repo", "sub", "m.gguf")
        _write(deep)
        self.assertEqual(archive.model_unit(deep, self.models), ["repo"])

    def test_strata_folder_per_model(self):
        p = os.path.join(self.models, "strata", "iq2_xs", "IQ2_XS", "x-00001-of-00002.gguf")
        _write(p)
        self.assertEqual(archive.model_unit(p, self.models), [os.path.join("strata", "iq2_xs")])

    def test_loose_file_and_multipart_siblings(self):
        single = os.path.join(self.models, "solo.gguf")
        _write(single)
        self.assertEqual(archive.model_unit(single, self.models), ["solo.gguf"])
        for i in (1, 2, 3):
            _write(os.path.join(self.models, f"big-0000{i}-of-00003.gguf"))
        _write(os.path.join(self.models, "big-00001-of-00002.gguf"))  # different set
        self.assertEqual(archive.model_unit(os.path.join(self.models, "big-00002-of-00003.gguf"), self.models),
                         ["big-00001-of-00003.gguf", "big-00002-of-00003.gguf", "big-00003-of-00003.gguf"])

    def test_rejects_paths_outside_or_partial(self):
        with self.assertRaises(ArchiveError):
            archive.model_unit("/etc/passwd", self.models)
        with self.assertRaises(ArchiveError):
            archive.model_unit(self.models, self.models)
        with self.assertRaises(ArchiveError):
            archive.model_unit(os.path.join(self.models, "missing.gguf"), self.models)
        p = os.path.join(self.models, archive.PARTIAL_PREFIX + "x", "m.gguf")
        _write(p)
        with self.assertRaises(ArchiveError):
            archive.model_unit(p, self.models)


class MoveTests(_Dirs):

    def _model(self, name="M", files=("a-00001-of-00002.gguf", "a-00002-of-00002.gguf"), size=50_000):
        data = {}
        for f in files:
            p = os.path.join(self.models, name, f)
            _write(p, size)
            with open(p, "rb") as fh:
                data[f] = fh.read()
        return data

    def _assert_moved(self, data, name="M"):
        self.assertFalse(os.path.exists(os.path.join(self.models, name)))
        for f, content in data.items():
            with open(os.path.join(self.arch, name, f), "rb") as fh:
                self.assertEqual(fh.read(), content)
        self.assertEqual([n for n in os.listdir(self.arch) if n.startswith(archive.PARTIAL_PREFIX)], [])

    def test_copy_path_moves_verifies_and_deletes_source(self):
        data = self._model()
        job = _job(self.models, self.arch, ["M"])
        with patch("core.archive._same_device", return_value=False):
            archive.run_move(job)
        self.assertEqual(job["method"], "copy")
        self.assertEqual(job["bytes_done"], job["bytes_total"])
        self._assert_moved(data)

    def test_rename_path(self):
        data = self._model()
        job = _job(self.models, self.arch, ["M"])
        archive.run_move(job)  # same tmpfs -> rename
        self.assertEqual(job["method"], "rename")
        self._assert_moved(data)

    def test_rename_exdev_falls_back_to_copy(self):
        data = self._model()
        job = _job(self.models, self.arch, ["M"])
        real_rename = os.rename
        calls = {"n": 0}

        def flaky(a, b):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError(18, "Invalid cross-device link")
            return real_rename(a, b)
        with patch("core.archive.os.rename", side_effect=flaky):
            archive.run_move(job)
        self.assertEqual(job["method"], "copy")
        self._assert_moved(data)

    def test_cancel_leaves_source_and_no_partial(self):
        data = self._model(size=200_000)
        job = _job(self.models, self.arch, ["M"])
        job["_cancel"].set()
        with patch("core.archive._same_device", return_value=False), \
             patch("core.archive._CHUNK", 1024):
            with self.assertRaises(archive._Cancelled):
                archive.run_move(job)
        for f in data:
            self.assertTrue(os.path.isfile(os.path.join(self.models, "M", f)))
        self.assertEqual(os.listdir(self.arch), [])

    def test_verification_failure_keeps_source(self):
        data = self._model()
        job = _job(self.models, self.arch, ["M"])
        real_copy = archive._copy_unit

        def truncating(job, src_base, tmp, unit):
            real_copy(job, src_base, tmp, unit)
            with open(os.path.join(tmp, "M", "a-00002-of-00002.gguf"), "r+b") as fh:
                fh.truncate(10)
        with patch("core.archive._same_device", return_value=False), \
             patch("core.archive._copy_unit", side_effect=truncating):
            with self.assertRaises(ArchiveError):
                archive.run_move(job)
        for f in data:
            self.assertTrue(os.path.isfile(os.path.join(self.models, "M", f)))
        self.assertEqual(os.listdir(self.arch), [])

    def test_not_enough_space(self):
        self._model()
        job = _job(self.models, self.arch, ["M"])
        fake = type("U", (), {"free": 10})()
        with patch("core.archive._same_device", return_value=False), \
             patch("core.archive.shutil.disk_usage", return_value=fake):
            with self.assertRaises(ArchiveError) as cm:
                archive.run_move(job)
        self.assertIn("free space", str(cm.exception))
        self.assertTrue(os.path.isdir(os.path.join(self.models, "M")))

    def test_destination_exists(self):
        self._model()
        os.makedirs(os.path.join(self.arch, "M"))
        with self.assertRaises(ArchiveError):
            archive.start_job("archive", os.path.join(self.models, "M"), self.models, self.arch, ["M"])

    def test_strata_unit_prunes_empty_parent(self):
        p = os.path.join(self.models, "strata", "iq2_xs", "f.gguf")
        _write(p)
        job = _job(self.models, self.arch, [os.path.join("strata", "iq2_xs")])
        with patch("core.archive._same_device", return_value=False):
            archive.run_move(job)
        self.assertTrue(os.path.isfile(os.path.join(self.arch, "strata", "iq2_xs", "f.gguf")))
        self.assertFalse(os.path.exists(os.path.join(self.models, "strata")))

    def test_worker_round_trip(self):
        data = self._model()
        job = archive.start_job("archive", os.path.join(self.models, "M", "a-00001-of-00002.gguf"),
                                self.models, self.arch, ["M"])
        done = self.wait(job["id"])
        self.assertEqual(done["status"], "completed", done.get("error"))
        self._assert_moved(data)
        back = archive.start_job("restore", os.path.join(self.arch, "M"), self.arch, self.models, ["M"])
        self.assertEqual(self.wait(back["id"])["status"], "completed")
        self.assertTrue(os.path.isdir(os.path.join(self.models, "M")))

    def test_cleanup_partials(self):
        os.makedirs(os.path.join(self.arch, archive.PARTIAL_PREFIX + "dead", "M"))
        os.makedirs(os.path.join(self.models, archive.PARTIAL_PREFIX + "dead2"))
        os.makedirs(os.path.join(self.models, "keep"))
        self.assertEqual(archive.cleanup_partials(), 2)
        self.assertEqual(os.listdir(self.arch), [])
        self.assertEqual(os.listdir(self.models), ["keep"])

    def test_discover_models_skips_partial_copies(self):
        _write(os.path.join(self.models, archive.PARTIAL_PREFIX + "x", "M", "m.gguf"))
        _write(os.path.join(self.models, "real", "r.gguf"))
        self.assertEqual([m["name"] for m in models_api.discover_models(self.models)], ["r"])


class _Api(_Dirs):
    def setUp(self):
        super().setUp()
        app = Flask(__name__)
        app.register_blueprint(archive_api.bp)
        app.register_blueprint(models_api.bp)
        self.client = app.test_client()
        with instances_lock:
            self._saved_inst = dict(instances)
            instances.clear()
        with downloads_lock:
            self._saved_dl = dict(downloads)
            downloads.clear()

    def tearDown(self):
        with instances_lock:
            instances.clear()
            instances.update(self._saved_inst)
        with downloads_lock:
            downloads.clear()
            downloads.update(self._saved_dl)
        super().tearDown()


class ArchiveApiTests(_Api):

    def setUp(self):
        super().setUp()
        self.gguf = os.path.join(self.models, "Qwen", "q-00001-of-00002.gguf")
        _write(self.gguf)
        _write(os.path.join(self.models, "Qwen", "q-00002-of-00002.gguf"))

    def test_status(self):
        data = self.client.get("/api/archive").get_json()
        self.assertTrue(data["available"])
        self.assertEqual(data["jobs"], [])

    def test_disabled(self):
        with patch("config.ARCHIVE_DIR", ""):
            resp = self.client.post("/api/archive", json={"path": self.gguf})
        self.assertEqual(resp.status_code, 400)

    def test_archive_then_restore(self):
        resp = self.client.post("/api/archive", json={"path": self.gguf})
        self.assertEqual(resp.status_code, 202, resp.get_json())
        self.assertEqual(self.wait(resp.get_json()["id"])["status"], "completed")
        archived = os.path.join(self.arch, "Qwen", "q-00001-of-00002.gguf")
        self.assertTrue(os.path.isfile(archived))

        resp = self.client.post("/api/archive/restore", json={"path": archived})
        self.assertEqual(resp.status_code, 202, resp.get_json())
        self.assertEqual(self.wait(resp.get_json()["id"])["status"], "completed")
        self.assertTrue(os.path.isfile(self.gguf))

    def test_refuses_model_used_by_instance_even_sleeping(self):
        for status in ("healthy", "starting", "sleeping"):
            with instances_lock:
                instances["i"] = {"id": "i", "status": status, "port": 8001,
                                  "model_path": self.gguf, "config": {}}
            resp = self.client.post("/api/archive", json={"path": self.gguf})
            self.assertEqual(resp.status_code, 409, status)
            self.assertIn("port 8001", resp.get_json()["error"])
        with instances_lock:
            instances["i"]["status"] = "stopped"
        self.assertEqual(self.client.post("/api/archive", json={"path": self.gguf}).status_code, 202)

    def test_refuses_draft_and_mmproj_dependencies(self):
        with instances_lock:
            instances["i"] = {"id": "i", "status": "healthy", "port": 8002,
                              "model_path": os.path.join(self.models, "other.gguf"),
                              "config": {"spec_draft_model": self.gguf}}
        self.assertEqual(self.client.post("/api/archive", json={"path": self.gguf}).status_code, 409)

    def test_refuses_while_downloading_into_it(self):
        with downloads_lock:
            downloads["d"] = {"id": "d", "status": "downloading",
                              "dest_path": os.path.join(self.models, "Qwen")}
        resp = self.client.post("/api/archive", json={"path": self.gguf})
        self.assertEqual(resp.status_code, 409)
        self.assertIn("download", resp.get_json()["error"])

    def test_bad_paths(self):
        self.assertEqual(self.client.post("/api/archive", json={"path": "/etc/passwd"}).status_code, 400)
        self.assertEqual(self.client.post("/api/archive/restore", json={"path": self.gguf}).status_code, 400)

    def test_restore_refuses_existing_destination(self):
        _write(os.path.join(self.arch, "Qwen", "q-00001-of-00002.gguf"))
        resp = self.client.post("/api/archive/restore",
                                json={"path": os.path.join(self.arch, "Qwen", "q-00001-of-00002.gguf")})
        self.assertEqual(resp.status_code, 409)

    def test_cancel_queued_and_forget_finished(self):
        with patch("core.archive._next_job", return_value=None):  # worker never picks it up
            job = self.client.post("/api/archive", json={"path": self.gguf}).get_json()
            resp = self.client.delete(f"/api/archive/jobs/{job['id']}")
        self.assertEqual(resp.get_json()["status"], "cancelled")
        self.assertTrue(os.path.isfile(self.gguf))
        resp = self.client.delete(f"/api/archive/jobs/{job['id']}")  # now finished -> forget
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(self.client.delete(f"/api/archive/jobs/{job['id']}").status_code, 404)

    def test_models_list_flags_archived(self):
        _write(os.path.join(self.arch, "Old", "old.gguf"))
        with patch("api.models.get_storage") as st, patch.object(models_api, "MODELS_DIR", self.models):
            st.return_value.get_settings.return_value = {}
            data = self.client.get("/api/models").get_json()
        old = next(m for m in data if m["name"] == "old")
        self.assertTrue(old["archived"])
        self.assertEqual(old["restore_path"], os.path.join(self.models, "Old", "old.gguf"))
        self.assertNotIn("archived", next(m for m in data if m["name"].startswith("q-")))

    def test_archived_models_not_in_compat_library(self):
        import api.llamaman as llamaman
        _write(os.path.join(self.arch, "Old", "old.gguf"))
        with patch.object(llamaman, "MODELS_DIR", self.models):
            names = [m["name"] for m in llamaman._library()]
        self.assertNotIn("old", names)


class BusyGuardTests(_Api):

    def test_launch_and_delete_blocked_while_moving(self):
        gguf = os.path.join(self.models, "M", "m.gguf")
        _write(gguf)
        with patch("core.archive._next_job", return_value=None):  # stays queued = active
            job = archive.start_job("archive", gguf, self.models, self.arch, ["M"])
            try:
                self.assertIn("being archived", archive.busy_reason(gguf))
                with patch("api.instances.is_port_available", return_value=True), \
                     patch("api.instances._run_container") as run_mock:
                    inst, err = instances_api.launch_instance(model_path=gguf, port=8001, ctx_size=4096)
                self.assertIsNone(inst)
                self.assertIn("being archived", err)
                run_mock.assert_not_called()
                with patch.object(models_api, "MODELS_DIR", self.models):
                    resp = self.client.post("/api/models/delete", json={"path": gguf})
                self.assertEqual(resp.status_code, 409)
                self.assertIsNone(archive.busy_reason(os.path.join(self.models, "other.gguf")))
            finally:
                archive.cancel_job(job["id"])


if __name__ == "__main__":
    unittest.main()
