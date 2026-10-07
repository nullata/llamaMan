# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Settings -> Docker Images: the llama.cpp auto-update pulls every tracked
image, and the Strata image is built from its downloaded repository
(core/strata_build.py) - by the button, or by the auto-update only when the
repository is already there."""

import base64
import io
import json
import os
import shutil
import tarfile
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from flask import Flask

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import api.images as images_api
from core import strata_build

SHA1 = "a" * 40
SHA2 = "b" * 40


class _Store:
    """In-memory docker_images settings."""

    def __init__(self, initial=None):
        self.data = dict(initial or {})

    def read(self):
        return json.loads(json.dumps(self.data))

    def write(self, d):
        self.data = json.loads(json.dumps(d))


def _store_patches(store):
    return (patch("api.images._read_docker_images", side_effect=store.read),
            patch("api.images._write_docker_images", side_effect=store.write))


class LlamaCppAutoUpdateAllTests(unittest.TestCase):

    def _run(self, initial):
        store = _Store(initial)
        r, w = _store_patches(store)
        with r, w, patch("api.images.LLAMA_IMAGE", "ghcr.io/ggml-org/llama.cpp:server-cuda"), \
             patch("api.images._trigger_pulls", return_value=True) as trig:
            return images_api.check_and_pull_all_if_needed(), trig

    def test_off_pulls_nothing(self):
        started, trig = self._run({"auto_update_enabled": False})
        self.assertEqual(started, [])
        trig.assert_not_called()

    def test_pulls_every_due_image(self):
        now = time.time()
        started, trig = self._run({
            "auto_update_enabled": True, "auto_update_interval_hours": 24,
            "images": [
                {"name": "ghcr.io/ggml-org/llama.cpp:server-cuda", "last_pulled_at": now - 90000},
                {"name": "ghcr.io/ggml-org/llama.cpp:server-vulkan", "last_pulled_at": now - 90000},
                {"name": "local/fresh:1", "last_pulled_at": now - 60},
            ]})
        self.assertEqual(started, ["ghcr.io/ggml-org/llama.cpp:server-cuda",
                                   "ghcr.io/ggml-org/llama.cpp:server-vulkan"])
        trig.assert_called_once_with(started)

    def test_default_image_pulled_even_when_untracked(self):
        started, _ = self._run({"auto_update_enabled": True})
        self.assertEqual(started, ["ghcr.io/ggml-org/llama.cpp:server-cuda"])


class ImageSettingsRouteTests(unittest.TestCase):

    def _post(self, initial, body):
        store = _Store(initial)
        r, w = _store_patches(store)
        app = Flask(__name__)
        app.register_blueprint(images_api.bp)
        with r, w:
            self.assertEqual(app.test_client().post("/api/images/settings", json=body).status_code, 200)
        return store.data

    def test_strata_save_keeps_llamacpp_settings(self):
        data = self._post({"auto_update_enabled": True, "auto_update_interval_hours": 6},
                          {"strata_auto_update_enabled": True, "strata_auto_update_interval_hours": 48})
        self.assertTrue(data["auto_update_enabled"])
        self.assertEqual(data["auto_update_interval_hours"], 6)
        self.assertEqual(data["strata"], {"auto_update_enabled": True, "auto_update_interval_hours": 48})

    def test_llamacpp_save_keeps_strata_settings(self):
        data = self._post({"strata": {"auto_update_enabled": True, "built_sha": SHA1}},
                          {"auto_update_enabled": False, "auto_update_interval_hours": 0})
        self.assertFalse(data["auto_update_enabled"])
        self.assertEqual(data["auto_update_interval_hours"], 1)
        self.assertEqual(data["strata"], {"auto_update_enabled": True, "built_sha": SHA1})


class _SrcDirCase(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="strata-build-", dir=os.environ.get("TMPDIR"))
        self.src = os.path.join(self.tmp, "engines", "Strata")
        self._p = [patch("config.STRATA_SRC_DIR", self.src),
                   patch("config.STRATA_REPO", "Niko1221/Strata"),
                   patch("config.STRATA_ENABLED", True),
                   patch("config.STRATA_IMAGE", "strata:latest")]
        for p in self._p:
            p.start()
        strata_build._state.update(status="idle", message="")

    def tearDown(self):
        for p in self._p:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_source(self, sha=SHA1):
        os.makedirs(self.src, exist_ok=True)
        with open(os.path.join(self.src, "Dockerfile"), "w") as f:
            f.write("FROM scratch\n")
        if sha:
            with open(self.src + ".source.json", "w") as f:
                json.dump({"sha": sha, "fetched_at": 1.0}, f)


class AutoUpdateGateTests(_SrcDirCase):

    def _check(self, rec):
        store = _Store({"strata": rec})
        r, w = _store_patches(store)
        with r, w, patch("core.strata_build.start", return_value=True) as start:
            return strata_build.check_and_update_if_needed(), start, store.data

    def test_never_downloads_the_repository(self):
        started, start, _ = self._check({"auto_update_enabled": True})
        self.assertFalse(started)
        start.assert_not_called()

    def test_off(self):
        self._make_source()
        started, start, _ = self._check({"auto_update_enabled": False})
        self.assertFalse(started)
        start.assert_not_called()

    def test_due_starts_and_records_check(self):
        self._make_source()
        started, start, data = self._check({"auto_update_enabled": True, "auto_update_interval_hours": 24,
                                            "last_auto_check_at": time.time() - 90000})
        self.assertTrue(started)
        start.assert_called_once_with("auto")
        self.assertGreater(data["strata"]["last_auto_check_at"], time.time() - 60)

    def test_not_due(self):
        self._make_source()
        started, start, _ = self._check({"auto_update_enabled": True, "auto_update_interval_hours": 24,
                                         "last_auto_check_at": time.time() - 60})
        self.assertFalse(started)
        start.assert_not_called()


class RunTests(_SrcDirCase):

    def _run(self, trigger, remote, rec=None, present=True):
        store = _Store({"strata": rec or {}})
        r, w = _store_patches(store)
        with r, w, patch("core.strata_build.remote_sha", return_value=remote), \
             patch("core.strata_build.download_source") as dl, \
             patch("core.strata_build.build_image", return_value="sha256:abc") as build, \
             patch("core.strata_build._image_present", return_value=present):
            strata_build._run(trigger)
        return dl, build, store.data

    def test_auto_up_to_date_does_not_build(self):
        self._make_source(SHA1)
        dl, build, _ = self._run("auto", SHA1, {"built_sha": SHA1})
        dl.assert_not_called()
        build.assert_not_called()
        self.assertEqual(strata_build.get_state()["message"], "Up to date")

    def test_auto_new_commit_downloads_and_builds(self):
        self._make_source(SHA1)
        dl, build, data = self._run("auto", SHA2, {"built_sha": SHA1})
        dl.assert_called_once_with(SHA2)
        build.assert_called_once()
        self.assertEqual(data["strata"]["built_sha"], SHA2)
        self.assertEqual(strata_build.get_state()["status"], "done")

    def test_auto_rebuilds_missing_image(self):
        self._make_source(SHA1)
        dl, build, _ = self._run("auto", SHA1, {"built_sha": SHA1}, present=False)
        dl.assert_not_called()
        build.assert_called_once()

    def test_auto_without_source_does_nothing(self):
        dl, build, _ = self._run("auto", SHA2)
        dl.assert_not_called()
        build.assert_not_called()

    def test_manual_downloads_first_time(self):
        dl, build, _ = self._run("manual", SHA1)
        dl.assert_called_once_with(SHA1)
        build.assert_called_once()

    def test_manual_same_commit_still_builds(self):
        self._make_source(SHA1)
        dl, build, _ = self._run("manual", SHA1, {"built_sha": SHA1})
        dl.assert_not_called()
        build.assert_called_once()

    def test_build_error_is_reported(self):
        self._make_source(SHA1)
        store = _Store({})
        r, w = _store_patches(store)
        with r, w, patch("core.strata_build.remote_sha", return_value=SHA1), \
             patch("core.strata_build.build_image", side_effect=RuntimeError("nvcc failed")):
            strata_build._run("manual")
        st = strata_build.get_state()
        self.assertEqual((st["status"], st["message"]), ("error", "nvcc failed"))


class DownloadSourceTests(_SrcDirCase):

    def _tarball(self, files):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for name, body in files.items():
                info = tarfile.TarInfo(f"Strata-{SHA2}/{name}")
                info.size = len(body)
                tf.addfile(info, io.BytesIO(body))
        buf.seek(0)
        resp = MagicMock()
        resp.raw = buf
        resp.__enter__.return_value = resp
        return resp

    def test_replaces_old_copy(self):
        self._make_source(SHA1)
        with open(os.path.join(self.src, "stale.txt"), "w") as f:
            f.write("old")
        resp = self._tarball({"Dockerfile": b"FROM x\n", "src/main.cu": b"//"})
        with patch("requests.get", return_value=resp) as get:
            strata_build.download_source(SHA2)
        self.assertIn(SHA2, get.call_args[0][0])
        self.assertTrue(os.path.isfile(os.path.join(self.src, "src", "main.cu")))
        self.assertFalse(os.path.exists(os.path.join(self.src, "stale.txt")))
        self.assertEqual(strata_build.source_info()["sha"], SHA2)
        self.assertEqual(sorted(os.listdir(os.path.dirname(self.src))), ["Strata", "Strata.source.json"])

    def test_bad_download_keeps_old_copy(self):
        self._make_source(SHA1)
        resp = self._tarball({"README.md": b"no dockerfile"})
        with patch("requests.get", return_value=resp):
            with self.assertRaises(RuntimeError):
                strata_build.download_source(SHA2)
        self.assertEqual(strata_build.source_info()["sha"], SHA1)
        self.assertTrue(os.path.isfile(os.path.join(self.src, "Dockerfile")))
        self.assertEqual(sorted(os.listdir(os.path.dirname(self.src))), ["Strata", "Strata.source.json"])


class BuildImageTests(_SrcDirCase):

    def _build(self, chunks, cuda=""):
        self._make_source()
        api = MagicMock()
        api._url.side_effect = lambda p: "http+docker://localhost" + p
        api._stream_helper.return_value = iter(chunks)
        api.pull.return_value = iter([{"status": "Pull complete"}])
        client = MagicMock(api=api)
        with patch("core.helpers.get_docker_client", return_value=client), \
             patch("config.STRATA_CUDA_ARCHITECTURES", cuda):
            return strata_build.build_image(), api

    def test_buildkit_build_with_progress(self):
        trace = base64.b64encode(b"\x0a\x10junk[2/5] RUN pip install -r requirements.txt\x12\x00").decode()
        msgs = []
        with patch("core.strata_build._set", side_effect=lambda **kw: msgs.append(kw.get("message"))):
            image_id, api = self._build([
                {"id": "moby.buildkit.trace", "aux": trace},
                {"id": "moby.image.id", "aux": {"ID": "sha256:feed"}},
            ], cuda="89")
        self.assertEqual(image_id, "sha256:feed")
        params = api._post.call_args.kwargs["params"]
        self.assertEqual((params["version"], params["t"], params["dockerfile"]),
                         ("2", "strata:latest", strata_build.CONTEXT_DOCKERFILE))
        self.assertEqual(json.loads(params["buildargs"]), {"CUDA_ARCHITECTURES": "89"})
        self.assertIn("[2/5] RUN pip install -r requirements.txt", msgs)

    def test_build_error_raises(self):
        with self.assertRaisesRegex(RuntimeError, "exit code 2"):
            self._build([{"errorDetail": {"message": "process exited with exit code 2"}}])

    def test_context_is_gzipped(self):
        # BuildKit only recognises the upload as an archive from its first 1 KB.
        self._make_source()
        with open(os.path.join(self.src, ".dockerignore"), "w") as f:
            f.write(".git\n/engine/\n")
        os.makedirs(os.path.join(self.src, "engine"))
        open(os.path.join(self.src, "engine", "big.bin"), "w").close()
        ctx = strata_build._build_context(self.src)
        try:
            self.assertEqual(ctx.read(3), b"\x1f\x8b\x08")
            ctx.seek(0)
            with tarfile.open(fileobj=ctx, mode="r:gz") as tf:
                names = tf.getnames()
        finally:
            ctx.close()
        self.assertIn(strata_build.CONTEXT_DOCKERFILE, names)
        self.assertNotIn("engine/big.bin", names)

    def test_syntax_directive_dropped(self):
        # Resolving docker/dockerfile:1 needs a client session the API build lacks.
        self._make_source()
        with open(os.path.join(self.src, "Dockerfile"), "w") as f:
            f.write("# syntax=docker/dockerfile:1\n#\n# notes\nFROM scratch\n"
                    "RUN python3 - <<'EOF'\n# syntax=keep-in-heredoc\nEOF\n")
        ctx = strata_build._build_context(self.src)
        try:
            with tarfile.open(fileobj=ctx, mode="r:gz") as tf:
                text = tf.extractfile(strata_build.CONTEXT_DOCKERFILE).read().decode()
        finally:
            ctx.close()
        self.assertEqual(text, "#\n# notes\nFROM scratch\nRUN python3 - <<'EOF'\n# syntax=keep-in-heredoc\nEOF\n")

    def test_base_images_pulled_before_build(self):
        _, api = self._build([{"id": "moby.image.id", "aux": {"ID": "sha256:feed"}}])
        api.pull.assert_not_called()    # FROM scratch: nothing to pull
        with open(os.path.join(self.src, "Dockerfile"), "w") as f:
            f.write("FROM nvidia/cuda:13.0.0-devel-ubuntu24.04\n")
        api.pull.reset_mock()
        api.pull.return_value = iter([])
        api._stream_helper.return_value = iter([{"id": "moby.image.id", "aux": {"ID": "sha256:feed"}}])
        with patch("core.helpers.get_docker_client", return_value=MagicMock(api=api)):
            strata_build.build_image()
        api.pull.assert_called_once_with("nvidia/cuda", tag="13.0.0-devel-ubuntu24.04",
                                         stream=True, decode=True)

    def test_base_images_parsing(self):
        self.assertEqual(strata_build.base_images(
            "FROM --platform=linux/amd64 nvidia/cuda:13.0.0-devel AS build\n"
            "FROM build AS app\nfrom scratch\nFROM ${BASE}\n"
            "FROM localhost:5000/x/y\nFROM ubuntu\n"),
            ["nvidia/cuda:13.0.0-devel", "localhost:5000/x/y", "ubuntu"])

    def test_pull_error_raises(self):
        self._make_source()
        with open(os.path.join(self.src, "Dockerfile"), "w") as f:
            f.write("FROM ubuntu:24.04\n")
        api = MagicMock()
        api.pull.return_value = iter([{"error": "toomanyrequests"}])
        with patch("core.helpers.get_docker_client", return_value=MagicMock(api=api)):
            with self.assertRaisesRegex(RuntimeError, "toomanyrequests"):
                strata_build.pull_base_images(self.src)

    def test_no_build_args_by_default(self):
        _, api = self._build([{"id": "moby.image.id", "aux": {"ID": "sha256:feed"}}])
        self.assertNotIn("buildargs", api._post.call_args.kwargs["params"])


if __name__ == "__main__":
    unittest.main()
