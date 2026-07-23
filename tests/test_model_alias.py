# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import os
import tempfile
import unittest
from unittest.mock import Mock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import core.model_alias as alias
import proxy as proxy_mod


def _storage_with_presets(presets):
    storage = Mock()
    storage.get_all_presets.return_value = presets
    return storage


class AliasResolutionTests(unittest.TestCase):
    def setUp(self):
        alias.invalidate()

    def tearDown(self):
        alias.invalidate()

    def test_resolves_exact_pretty_name_case_insensitively(self):
        storage = _storage_with_presets({
            "/models/Qwen2.5-14B-Instruct-Q4_K_M.gguf": {"pretty_name": "Qwen 2.5 14B"},
        })
        with patch.object(alias, "get_storage", return_value=storage):
            self.assertEqual(alias.resolve_to_stem("Qwen 2.5 14B"),
                             "qwen2.5-14b-instruct-q4_k_m")
            self.assertEqual(alias.resolve_to_stem("qwen 2.5 14b"),
                             "qwen2.5-14b-instruct-q4_k_m")

    def test_strips_ollama_style_tag_suffix(self):
        storage = _storage_with_presets({
            "/models/foo-Q4.gguf": {"pretty_name": "My Model"},
        })
        with patch.object(alias, "get_storage", return_value=storage):
            self.assertEqual(alias.resolve_to_stem("My Model:latest"), "foo-q4")

    def test_unknown_name_passes_through_untouched(self):
        """A filename stem or a share_queue_group must survive canonical_name()
        unchanged - that is what keeps cluster routing working."""
        storage = _storage_with_presets({
            "/models/foo-Q4.gguf": {"pretty_name": "My Model"},
        })
        with patch.object(alias, "get_storage", return_value=storage):
            self.assertEqual(alias.canonical_name("qwen2.5-14b"), "qwen2.5-14b")
            self.assertEqual(alias.canonical_name("foo-q4"), "foo-q4")

    def test_matching_is_exact_never_substring(self):
        storage = _storage_with_presets({
            "/models/foo-Q4.gguf": {"pretty_name": "Qwen 2.5 14B"},
        })
        with patch.object(alias, "get_storage", return_value=storage):
            self.assertEqual(alias.resolve_to_stem("Qwen"), "")
            self.assertEqual(alias.resolve_to_stem("Qwen 2.5 14B Instruct"), "")

    def test_storage_failure_degrades_to_no_aliases(self):
        storage = Mock()
        storage.get_all_presets.side_effect = RuntimeError("db down")
        with patch.object(alias, "get_storage", return_value=storage):
            self.assertEqual(alias.resolve_to_stem("anything"), "")
            self.assertEqual(alias.canonical_name("foo-q4"), "foo-q4")

    def test_presets_without_pretty_name_are_ignored(self):
        """Pre-existing presets have no pretty_name key at all - the seamless
        upgrade case for both storage backends."""
        storage = _storage_with_presets({
            "/models/foo-Q4.gguf": {"ctx_size": 4096, "favorite": True},
        })
        with patch.object(alias, "get_storage", return_value=storage):
            self.assertEqual(alias.resolve_to_stem("foo-q4"), "")
            self.assertEqual(alias.pretty_name_for_path("/models/foo-Q4.gguf"), "")


class ProxyModelMatchTests(unittest.TestCase):
    def setUp(self):
        alias.invalidate()

    def tearDown(self):
        alias.invalidate()

    def test_proxy_accepts_pretty_name_and_still_accepts_filename(self):
        storage = _storage_with_presets({
            "/models/foo-Q4.gguf": {"pretty_name": "My Model"},
        })
        with patch.object(alias, "get_storage", return_value=storage):
            self.assertTrue(proxy_mod._model_matches("/models/foo-Q4.gguf", "My Model"))
            self.assertTrue(proxy_mod._model_matches("/models/foo-Q4.gguf", "foo-Q4"))
            self.assertFalse(proxy_mod._model_matches("/models/foo-Q4.gguf", "Other Model"))


class PrettyNameValidationTests(unittest.TestCase):
    def setUp(self):
        alias.invalidate()

    def tearDown(self):
        alias.invalidate()

    def _validate(self, pretty, model_path, models_dir, presets):
        import api.models as models_api
        import api.presets as presets_api
        storage = _storage_with_presets(presets)
        with patch.object(models_api, "MODELS_DIR", models_dir), \
             patch("config.MODELS_DIR", models_dir), \
             patch.object(alias, "get_storage", return_value=storage), \
             patch.object(presets_api, "get_storage", return_value=storage):
            return presets_api.validate_pretty_name(pretty, model_path)

    def test_empty_name_clears_and_is_valid(self):
        with tempfile.TemporaryDirectory() as d:
            value, err = self._validate("", os.path.join(d, "a.gguf"), d, {})
            self.assertEqual(value, "")
            self.assertEqual(err, "")

    def test_rejects_colon(self):
        with tempfile.TemporaryDirectory() as d:
            _, err = self._validate("My:Model", os.path.join(d, "a.gguf"), d, {})
            self.assertIn("':'", err)

    def test_rejects_duplicate_of_another_models_pretty_name(self):
        with tempfile.TemporaryDirectory() as d:
            other = os.path.join(d, "other.gguf")
            with open(other, "wb") as f:
                f.write(b"gguf")
            _, err = self._validate(
                "Shared Name", os.path.join(d, "a.gguf"), d,
                {other: {"pretty_name": "Shared Name"}},
            )
            self.assertIn("already used as the pretty name", err)

    def test_rejects_name_shadowing_another_models_filename(self):
        with tempfile.TemporaryDirectory() as d:
            other = os.path.join(d, "realmodel.gguf")
            with open(other, "wb") as f:
                f.write(b"gguf")
            _, err = self._validate("realmodel", os.path.join(d, "a.gguf"), d, {})
            self.assertIn("already the filename", err)

    def test_rejects_clash_with_cluster_queue_group(self):
        with tempfile.TemporaryDirectory() as d:
            other = os.path.join(d, "other.gguf")
            with open(other, "wb") as f:
                f.write(b"gguf")
            _, err = self._validate(
                "qwen2.5-14b", os.path.join(d, "a.gguf"), d,
                {other: {"share_queue_group": "qwen2.5-14b"}},
            )
            self.assertIn("cluster queue group", err)

    def test_renaming_own_model_to_its_current_name_is_allowed(self):
        with tempfile.TemporaryDirectory() as d:
            mine = os.path.join(d, "a.gguf")
            with open(mine, "wb") as f:
                f.write(b"gguf")
            value, err = self._validate(
                "My Model", mine, d, {mine: {"pretty_name": "My Model"}},
            )
            self.assertEqual(err, "")
            self.assertEqual(value, "My Model")

    def test_accepts_a_clean_new_name(self):
        with tempfile.TemporaryDirectory() as d:
            mine = os.path.join(d, "a.gguf")
            with open(mine, "wb") as f:
                f.write(b"gguf")
            value, err = self._validate("Qwen 2.5 14B", mine, d, {})
            self.assertEqual(err, "")
            self.assertEqual(value, "Qwen 2.5 14B")


if __name__ == "__main__":
    unittest.main()
