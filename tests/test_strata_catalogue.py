# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Strata's model list read from the installed Strata image
(core/engines/strata.refresh_catalogue_from_image): setup.py's tables are
dumped from the image when its id changes, converted, validated, cached per
image and applied in place; any failure keeps the built-in copy."""

import copy
import json
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import docker

import core.engines.strata as S


def dump_of_builtin():
    """What _DUMP_SCRIPT prints for the setup.py the built-in copy came from."""
    families, sizes, revisions = copy.deepcopy(S._BUILTIN_CATALOGUE)
    fam = {}
    for key, f in families.items():
        hf = f"https://huggingface.co/{f['repo']}/resolve/{revisions[f['repo']]}/" + ("{q}/" if f["subdir"] else "")
        fam[key] = {"title": f["title"], "name": f["served"], "hf": hf, "file": f["file"],
                    "shards": f["shards"], "vision": f["vision"], "experimental": f["experimental"]}
    mod = {k: {"download_gb": m["download_gb"], "ram_gb": m["ram_gb"], "families": list(m["families"])}
           for k, m in sizes.items()}
    return {"families": fam, "models": mod, "revisions": revisions}


def dump_with_new_size():
    d = dump_of_builtin()
    d["models"]["IQ4_XS"] = {"download_gb": 95.0, "ram_gb": 64, "families": ["qwen"]}
    return d


class ConversionTests(unittest.TestCase):

    def test_builtin_round_trips(self):
        self.assertEqual(S.catalogue_from_dump(dump_of_builtin()), S._BUILTIN_CATALOGUE)

    def test_bad_tables_rejected(self):
        for mutate in (
            lambda d: d["revisions"].clear(),                                  # no pinned revision
            lambda d: d["families"]["qwen"].update(file="weights.gguf"),       # unexpected pattern
            lambda d: d["models"]["Q2_0"].update(families=["nope"]),           # unknown family
            lambda d: d["models"]["Q2_0"].pop("ram_gb"),                       # missing field
            lambda d: d.update(models={}),                                     # empty
            lambda d: d.pop("families"),
        ):
            d = dump_of_builtin()
            mutate(d)
            with self.assertRaises(ValueError):
                S.catalogue_from_dump(d)


class RefreshTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._p = [patch("config.STRATA_ENABLED", True), patch("config.STRATA_IMAGE", "strata:latest"),
                   patch("config.DATA_DIR", self._tmp.name)]
        for p in self._p:
            p.start()
        self.client = Mock()
        self.client.images.get.return_value = Mock(id="sha256:aaa")
        self.client.containers.run.return_value = json.dumps(dump_with_new_size()).encode()
        self._dc = patch("core.helpers.get_docker_client", return_value=self.client)
        self._dc.start()
        S._catalogue_state.update(image=None, image_id=None, source="built-in", error=None)

    def tearDown(self):
        self._dc.stop()
        for p in reversed(self._p):
            p.stop()
        S._apply_catalogue(*copy.deepcopy(S._BUILTIN_CATALOGUE))
        S._catalogue_state.update(image=None, image_id=None, source="built-in", error=None)
        self._tmp.cleanup()

    def test_falls_back_to_built_then_pulled_image(self):
        import docker
        have = {"ghcr.io/someone/strata:latest": Mock(id="sha256:bbb")}

        def get(name):
            if name in have:
                return have[name]
            raise docker.errors.ImageNotFound(name)
        self.client.images.get.side_effect = get
        with patch("config.STRATA_IMAGE", "ghcr.io/me/strata:1"), \
             patch("config.STRATA_BUILD_IMAGE", "strata:latest"), \
             patch("api.images.tracked_images", return_value=["ghcr.io/someone/strata:latest"]):
            S.refresh_catalogue_from_image()
            self.assertEqual(S.resolve_image(), "ghcr.io/someone/strata:latest")
            self.assertEqual(self.client.containers.run.call_args.args[0], "ghcr.io/someone/strata:latest")
            have["strata:latest"] = Mock(id="sha256:ccc")       # built later: preferred over pulled
            S.refresh_catalogue_from_image()
            self.assertEqual(S.resolve_image(), "strata:latest")
            have["ghcr.io/me/strata:1"] = Mock(id="sha256:ddd")  # STRATA_IMAGE first
            S.refresh_catalogue_from_image()
            self.assertEqual(S.resolve_image(), "ghcr.io/me/strata:1")
            self.assertEqual(S.StrataEngine().default_image(), "ghcr.io/me/strata:1")

    def test_image_tables_applied_in_place(self):
        sizes_ref = S.SIZES                          # other modules hold these names
        S.refresh_catalogue_from_image()
        self.assertIs(S.SIZES, sizes_ref)
        self.assertIn("IQ4_XS", S.SIZES)
        self.assertEqual(S.catalogue_source()["source"], "image")
        # The new model is offered and recognized like the others.
        self.assertIn("strata/qwen-IQ4_XS", [m["id"] for m in S.catalogue()])
        self.assertEqual(S.model_for_file("/m/Qwen3.8-Flash-Next-GSQ-RCO-IQ4_XS-00001-of-00002.gguf"),
                         ("qwen", "IQ4_XS"))
        kw = self.client.containers.run.call_args.kwargs
        self.assertTrue(kw["network_disabled"])
        self.assertTrue(kw["remove"])

    def test_same_image_not_read_twice_and_cache_reused(self):
        S.refresh_catalogue_from_image()
        S.refresh_catalogue_from_image()
        self.assertEqual(self.client.containers.run.call_count, 1)
        # A restart (state reset) reads the per-image cache, not the image.
        S._apply_catalogue(*copy.deepcopy(S._BUILTIN_CATALOGUE))
        S._catalogue_state.update(image=None, image_id=None, source="built-in", error=None)
        S.refresh_catalogue_from_image()
        self.assertEqual(self.client.containers.run.call_count, 1)
        self.assertIn("IQ4_XS", S.SIZES)

    def test_new_image_is_read(self):
        S.refresh_catalogue_from_image()
        self.client.images.get.return_value = Mock(id="sha256:bbb")
        self.client.containers.run.return_value = json.dumps(dump_of_builtin()).encode()
        S.refresh_catalogue_from_image()
        self.assertEqual(self.client.containers.run.call_count, 2)
        self.assertNotIn("IQ4_XS", S.SIZES)

    def test_unreadable_output_keeps_builtin(self):
        self.client.containers.run.return_value = b"Traceback: something changed\n"
        with self.assertLogs("llamaman", level="WARNING") as logs:
            S.refresh_catalogue_from_image()
        self.assertIn("built-in", logs.output[0])
        self.assertEqual(S.SIZES, S._BUILTIN_CATALOGUE[1])
        src = S.catalogue_source()
        self.assertEqual(src["source"], "built-in")
        self.assertIn("could not read", src["error"])

    def test_changed_table_shape_keeps_builtin(self):
        d = dump_of_builtin()
        d["families"]["qwen"]["file"] = "weights.gguf"
        self.client.containers.run.return_value = json.dumps(d).encode()
        with self.assertLogs("llamaman", level="WARNING"):
            S.refresh_catalogue_from_image()
        self.assertEqual(S.catalogue_source()["source"], "built-in")

    def test_no_image_or_no_docker_keeps_builtin(self):
        S.refresh_catalogue_from_image()                   # image tables in place
        self.assertTrue(S.StrataEngine().describe("cuda")["image_built"])
        self.client.images.get.side_effect = docker.errors.ImageNotFound("x")
        S.refresh_catalogue_from_image()
        self.assertEqual(S.SIZES, S._BUILTIN_CATALOGUE[1])
        self.assertIn("not built", S.catalogue_source()["error"])
        self.assertFalse(S.StrataEngine().describe("cuda")["image_built"])   # hidden in the launch form
        self.client.images.get.side_effect = RuntimeError("socket")
        S.refresh_catalogue_from_image()
        self.assertIn("Docker unavailable", S.catalogue_source()["error"])

    def test_disabled_does_nothing(self):
        with patch("config.STRATA_ENABLED", False):
            S.refresh_catalogue_from_image()
        self.client.images.get.assert_not_called()

    def test_engines_endpoint_reports_source(self):
        S.refresh_catalogue_from_image()
        d = S.StrataEngine().describe("cuda")
        self.assertEqual(d["catalogue"]["source"], "image")


if __name__ == "__main__":
    unittest.main()
