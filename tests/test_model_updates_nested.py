# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Regression tests for update-checking models that are NESTED in their repo
(unsloth-style quant subfolders) and/or MULTIPART, e.g.

    repo:  unsloth/Qwen3.8-Flash-Next-GGUF
    file:  UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
    local: /models/.../Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf

The local path only ever carries the basename, so probing
`resolve/main/<basename>` 404s even though the file plainly exists under the
subfolder - the original bug: the button died with "File not found in repo".

The fix retries a 404 through the repo file listing (the downloader's
resolve_filename, shared with the download and apply paths), then re-probes
the real repo path. Auth/5xx failures must NOT take that path, and a file
that is genuinely gone must still report the same "File not found in repo"
error. Flat root-level models - the common case, and every pre-existing test
- must not spend a listing request or change behavior at all.

Also pins swap_in_update flattening: the live nested model already sits in
its subfolder (the downloader preserves repo layout on disk) while staging
reproduces the repo layout under `.llamaman-update/`, so the swap must
replace by basename - joining the relative path back on would double-nest
(UD-Q4_K_XL/UD-Q4_K_XL/x.gguf) and silently leave the live file stale.
"""

import os
import tempfile
import unittest
from unittest.mock import Mock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import core.model_updates as updates

REPO = "unsloth/Qwen3.8-Flash-Next-GGUF"
STEM = "Qwen3.8-Flash-Next-UD-Q4_K_XL"
SHARD1 = f"{STEM}-00001-of-00004.gguf"
NESTED1 = f"UD-Q4_K_XL/{SHARD1}"
REMOTE_SHA = "6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83"
OTHER_SHA = "1111111111111111111111111111111111111111111111111111111111111111"

# What HF's ?blobs=true listing returns for this repo (trimmed).
def _repo_listing(shard1_sha=REMOTE_SHA):
    files = [
        {"name": "README.md", "size": 42, "sha256": ""},
        {"name": "UD-Q4_K_XL/config.json", "size": 10, "sha256": ""},
    ]
    for i in range(1, 5):
        name = f"UD-Q4_K_XL/{STEM}-{i:05d}-of-00004.gguf"
        files.append({"name": name, "size": 5_000_000_000,
                      "sha256": shard1_sha if i == 1 else f"{i:064x}"})
    # A flat root-level model, like the pre-nesting era / single-file repos.
    files.append({"name": "flat.gguf", "size": 100, "sha256": REMOTE_SHA})
    return files


def _remote(sha=REMOTE_SHA, size=5_000_000_000):
    return {"sha256": sha, "size": size, "commit": "abc123"}


def _head_that_404s_bare(basename, nested_path, sha=REMOTE_SHA, size=5_000_000_000):
    """Stand-in for remote_file_info: bare basename → the real 404 behavior,
    nested repo path → success. Mirrors HF exactly (verified against the
    live repo: /resolve/main/<basename> answers 404, the nested path exists).
    """
    def info(repo_id, filename, token=None):
        if filename == basename:
            raise updates.RemoteFileNotFound(f"File not found in repo: {repo_id}/{filename}")
        if filename == nested_path:
            return _remote(sha, size)
        raise updates.RemoteFileNotFound(f"File not found in repo: {repo_id}/{filename}")
    return info


class CheckNestedMultipartTests(unittest.TestCase):
    """The reported bug: multipart model nested in a quant subfolder."""

    def _check(self, local_size, local_sha, info_fn):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, SHARD1)  # local copy: basename only
            with open(path, "wb") as f:
                f.write(b"x" * local_size)
            with patch.object(updates, "remote_file_info", side_effect=info_fn), \
                 patch("core.downloader.list_repo_files", return_value=_repo_listing()) as listing, \
                 patch("core.model_sources.get_model_sha", return_value=local_sha):
                return updates.check_model_update(path, REPO), listing

    def test_nested_shard_resolves_instead_of_file_not_found(self):
        result, listing = self._check(10, OTHER_SHA,
                                      _head_that_404s_bare(SHARD1, NESTED1))
        self.assertEqual(result["status"], updates.STATUS_UPDATE_AVAILABLE)
        # The verdict is now computed against the REAL repo path, and the
        # response carries it so the UI can show where it looked.
        self.assertEqual(result["filename"], NESTED1)
        listing.assert_called_once()

    def test_nested_shard_up_to_date(self):
        result, _ = self._check(5_000_000_000, REMOTE_SHA,
                                _head_that_404s_bare(SHARD1, NESTED1))
        self.assertEqual(result["status"], updates.STATUS_UP_TO_DATE)

    def test_retry_asks_resolve_filename_for_the_multipart_group(self):
        # resolve_filename expands -00001 to all four shards; the check must
        # pick back out the shard it was asked about, not shard 1 of a
        # different file or the wrong sibling.
        asked = f"{STEM}-00003-of-00004.gguf"
        nested3 = f"UD-Q4_K_XL/{asked}"

        def info(repo_id, filename, token=None):
            if filename == asked:
                raise updates.RemoteFileNotFound("404")
            if filename == nested3:
                return _remote(sha=OTHER_SHA)
            raise updates.RemoteFileNotFound("404")

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, asked)
            with open(path, "wb") as f:
                f.write(b"x" * 10)
            with patch.object(updates, "remote_file_info", side_effect=info), \
                 patch("core.downloader.list_repo_files", return_value=_repo_listing()), \
                 patch("core.model_sources.get_model_sha", return_value=REMOTE_SHA):
                result = updates.check_model_update(path, REPO)
        self.assertEqual(result["filename"], nested3)
        self.assertEqual(result["status"], updates.STATUS_UPDATE_AVAILABLE)


class CheckFlatModelRegressionTests(unittest.TestCase):
    """Single-file root-level models - llamaMan's common case - keep the old
    behavior exactly: one HEAD, no listing request, same verdicts."""

    def _check(self, local_size, local_sha):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "flat.gguf")
            with open(path, "wb") as f:
                f.write(b"x" * local_size)
            with patch.object(updates, "remote_file_info", return_value=_remote(size=100)), \
                 patch("core.downloader.list_repo_files") as listing, \
                 patch("core.model_sources.get_model_sha", return_value=local_sha):
                return updates.check_model_update(path, "org/repo"), listing

    def test_flat_up_to_date_never_touches_the_listing(self):
        result, listing = self._check(100, REMOTE_SHA)
        self.assertEqual(result["status"], updates.STATUS_UP_TO_DATE)
        self.assertEqual(result["filename"], "flat.gguf")
        listing.assert_not_called()

    def test_flat_update_available_never_touches_the_listing(self):
        result, listing = self._check(100, OTHER_SHA)
        self.assertEqual(result["status"], updates.STATUS_UPDATE_AVAILABLE)
        listing.assert_not_called()

    def test_flat_path_field_is_the_basename(self):
        # _resolve_repo_filename must NOT rewrite paths that resolved first
        # try - the apply path and the UI both key off this.
        result, _ = self._check(100, REMOTE_SHA)
        self.assertEqual(result["filename"], "flat.gguf")


class CheckFailureModesTests(unittest.TestCase):
    """The retry must stay narrowly scoped to genuine 404s."""

    def _local(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        path = os.path.join(d.name, SHARD1)
        with open(path, "wb") as f:
            f.write(b"x" * 10)
        return path

    def test_genuinely_deleted_file_keeps_the_original_error(self):
        # 404 on HEAD, and the listing has no copy either -> same message as
        # before the fix, so 'file left the repo' still reads the same.
        def info(repo_id, filename, token=None):
            raise updates.RemoteFileNotFound(f"File not found in repo: {repo_id}/{filename}")
        listing = [{"name": "README.md", "size": 1, "sha256": ""}]
        with patch.object(updates, "remote_file_info", side_effect=info), \
             patch("core.downloader.list_repo_files", return_value=listing):
            result = updates.check_model_update(self._local(), REPO)
        self.assertEqual(result["status"], updates.STATUS_UNKNOWN)
        self.assertIn("File not found in repo", result["detail"])
        self.assertIn(SHARD1, result["detail"])

    def test_auth_failure_does_not_retry_through_the_listing(self):
        # A bad token 401s; listing the repo too would double the requests and
        # bury the auth message under a listing error.
        with patch.object(updates, "remote_file_info",
                          side_effect=RuntimeError("Authentication failed (401). Check your HF token.")), \
             patch("core.downloader.list_repo_files") as listing:
            result = updates.check_model_update(self._local(), "org/private")
        self.assertEqual(result["status"], updates.STATUS_UNKNOWN)
        self.assertIn("Authentication failed", result["detail"])
        listing.assert_not_called()

    def test_listing_failure_surfaces_the_listing_error(self):
        def info(repo_id, filename, token=None):
            raise updates.RemoteFileNotFound("File not found in repo: org/repo/x.gguf")
        with patch.object(updates, "remote_file_info", side_effect=info), \
             patch("core.downloader.list_repo_files",
                   side_effect=RuntimeError("Repository not found: org/repo")):
            result = updates.check_model_update(self._local(), "org/repo")
        self.assertEqual(result["status"], updates.STATUS_UNKNOWN)
        self.assertIn("Repository not found", result["detail"])

    def test_no_repo_short_circuits_before_any_network(self):
        result = updates.check_model_update(self._local(), "")
        self.assertEqual(result["status"], updates.STATUS_NO_REPO)


class SwapFlatteningTests(unittest.TestCase):
    """swap_in_update must land staged files by BASENAME in dest_dir.

    A nested model's live copy already sits inside its subfolder (the
    downloader preserves repo layout: /models/<repo>/UD-Q4_K_XL/x.gguf), and
    staging reproduces that layout under .llamaman-update/. Joining the
    staging-relative path back onto dest_dir double-nests it and leaves the
    live file stale - the update silently does nothing."""

    def test_nested_staged_file_replaces_the_live_basename(self):
        with tempfile.TemporaryDirectory() as d:
            live_dir = os.path.join(d, "UD-Q4_K_XL")   # live nested location
            os.makedirs(live_dir)
            live = os.path.join(live_dir, SHARD1)
            with open(live, "wb") as f:
                f.write(b"old")
            temp = updates.update_temp_dir(live)
            staged = os.path.join(temp, "UD-Q4_K_XL")
            os.makedirs(staged)
            with open(os.path.join(staged, SHARD1), "wb") as f:
                f.write(b"new")

            moved, err = updates.swap_in_update(temp, live_dir)

            self.assertIsNone(err)
            with open(live, "rb") as f:
                self.assertEqual(f.read(), b"new")           # live file replaced
            self.assertFalse(os.path.isdir(
                os.path.join(live_dir, "UD-Q4_K_XL")))       # no double-nest

    def test_multipart_shards_all_land_beside_the_live_shards(self):
        with tempfile.TemporaryDirectory() as d:
            live_dir = os.path.join(d, "UD-Q4_K_XL")
            os.makedirs(live_dir)
            names = [f"{STEM}-{i:05d}-of-00004.gguf" for i in range(1, 5)]
            for n in names:
                with open(os.path.join(live_dir, n), "wb") as f:
                    f.write(b"old")
            temp = updates.update_temp_dir(os.path.join(live_dir, names[0]))
            staged = os.path.join(temp, "UD-Q4_K_XL")
            os.makedirs(staged)
            for n in names:
                with open(os.path.join(staged, n), "wb") as f:
                    f.write(b"new")

            moved, err = updates.swap_in_update(temp, live_dir)

            self.assertIsNone(err)
            self.assertEqual(len(moved), 4)
            for n in names:
                with open(os.path.join(live_dir, n), "rb") as f:
                    self.assertEqual(f.read(), b"new")

    def test_flat_swap_unchanged(self):
        # The pre-existing shape (root-level model, flat staging) still works.
        with tempfile.TemporaryDirectory() as d:
            live = os.path.join(d, "m.gguf")
            with open(live, "wb") as f:
                f.write(b"old")
            temp = updates.update_temp_dir(live)
            os.makedirs(temp)
            with open(os.path.join(temp, "m.gguf"), "wb") as f:
                f.write(b"new")

            moved, err = updates.swap_in_update(temp, d)

            self.assertIsNone(err)
            self.assertEqual(moved, ["m.gguf"])
            with open(live, "rb") as f:
                self.assertEqual(f.read(), b"new")


class RemoteFileInfo404TypeTests(unittest.TestCase):
    """The 404 must raise the retryable type - and it must stay a RuntimeError
    so every existing `except RuntimeError` boundary keeps working."""

    def test_404_raises_retryable_type(self):
        resp = Mock(status_code=404, headers={})
        with patch.object(updates.requests, "head", return_value=resp):
            with self.assertRaises(updates.RemoteFileNotFound) as cm:
                updates.remote_file_info("org/repo", "gone.gguf")
        self.assertIn("File not found in repo", str(cm.exception))

    def test_type_is_still_a_runtime_error(self):
        self.assertTrue(issubclass(updates.RemoteFileNotFound, RuntimeError))


if __name__ == "__main__":
    unittest.main()
