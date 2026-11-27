# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Removing Strata's prepared data for models that are gone (core/strata_cleanup.py)."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import core.strata_cleanup as C

FLASH = "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-0000{}-of-00002.gguf"


def write(path, data=b"x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def flash_config(tag, name):
    return {
        "args": ["--pack", f"/data/Strata-data/packs/{tag}", "--native", "/models/" + FLASH.format(1),
                 "--ple-gguf", "/models/" + FLASH.format(2), "--mtp", "/data/Strata-data/mtp/rt"],
        "tokenizer": f"/data/Strata-data/packs/{tag}/tokenizer",
        "model_name": name,
        "log": f"/data/install/strata-{tag}.log",
        "vision": {"mmproj": "/data/Strata-data/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf",
                   "model": "/models/" + FLASH.format(1)},
    }


class Volume:
    """A Strata data volume like the one Strata's containers see at /data."""

    def __init__(self, root):
        self.root = root

    def setup(self, tag, config, pack_bytes=b"pack"):
        write(os.path.join(self.root, "Strata-data/packs", tag, "experts.bin"), pack_bytes)
        write(os.path.join(self.root, "Strata-data/packs", tag, "tokenizer/tokenizer.json"))
        for n in (f"strata-{tag}.json.bak", f"strata-{tag}.log", f"run-{tag}.sh"):
            write(os.path.join(self.root, "install", n))
        write(os.path.join(self.root, "install", f"strata-{tag}.json"), json.dumps(config).encode())

    def flash_shared(self):
        for rel in ("Strata-data/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf",
                    "Strata-data/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf.done",
                    "Strata-data/mtp/rt/a.bin", "Strata-data/mtp/mtp-q2_0.gguf",
                    "Strata-data/dflash2/qwen3.8-flash-next-dflash2.q4_0.pack",
                    "Strata-data/dflash2/draft_vocab.bin"):
            write(os.path.join(self.root, rel))

    def run(self, **req):
        env = dict(os.environ, LLAMAMAN_STRATA_ROOT=self.root, LLAMAMAN_CLEANUP=json.dumps(req))
        out = subprocess.run([sys.executable, "-I", "-c", C.CLEANUP_SCRIPT], env=env,
                             capture_output=True, text=True, check=True)
        return json.loads(out.stdout.strip().splitlines()[-1])

    def exists(self, rel):
        return os.path.lexists(os.path.join(self.root, rel))


class ScriptTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.v = Volume(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_deleted_models_setup_and_its_familys_shared_files_go(self):
        self.v.setup("iq2_xs", flash_config("iq2_xs", "qwen3.8-flash-next-iq2_xs"))
        self.v.flash_shared()
        plan = self.v.run(tags=["iq2_xs"])
        paths = {i["path"] for i in plan["items"]}
        self.assertEqual(plan["tags"], ["iq2_xs"])
        self.assertIn("/data/Strata-data/packs/iq2_xs", paths)
        self.assertIn("/data/install/strata-iq2_xs.json", paths)
        self.assertIn("/data/install/run-iq2_xs.sh", paths)
        self.assertIn("/data/Strata-data/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf.done", paths)
        self.assertIn("/data/Strata-data/mtp", paths)
        self.assertIn("/data/Strata-data/dflash2/draft_vocab.bin", paths)
        self.assertFalse(plan["applied"])
        self.assertTrue(self.v.exists("Strata-data/packs/iq2_xs"), "a plan removes nothing")

        done = self.v.run(tags=["iq2_xs"], apply=True)
        self.assertEqual(done["errors"], [])
        for rel in ("Strata-data/packs/iq2_xs", "install/strata-iq2_xs.json", "install/strata-iq2_xs.log",
                    "Strata-data/mtp", "Strata-data/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf"):
            self.assertFalse(self.v.exists(rel), rel)
        self.assertTrue(self.v.exists("Strata-data/models"), "top folders stay")

    def test_shared_files_another_setup_still_uses_stay(self):
        self.v.setup("iq2_xs", flash_config("iq2_xs", "qwen3.8-flash-next-iq2_xs"))
        self.v.setup("iq3_s", flash_config("iq3_s", "qwen3.8-flash-next-iq3_s"))
        self.v.flash_shared()
        self.v.run(tags=["iq2_xs"], apply=True)
        self.assertFalse(self.v.exists("Strata-data/packs/iq2_xs"))
        for rel in ("Strata-data/packs/iq3_s", "install/strata-iq3_s.json",
                    "Strata-data/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf", "Strata-data/mtp/rt/a.bin",
                    "Strata-data/dflash2/qwen3.8-flash-next-dflash2.q4_0.pack"):
            self.assertTrue(self.v.exists(rel), rel)

    def test_a_similar_tag_is_not_touched(self):
        self.v.setup("iq2_xs", flash_config("iq2_xs", "qwen3.8-flash-next-iq2_xs"))
        self.v.setup("swift-iq2_xs", flash_config("swift-iq2_xs", "qwen3.8-flash-next-swift-iq2_xs"))
        self.v.run(tags=["iq2_xs"], apply=True)
        self.assertTrue(self.v.exists("install/strata-swift-iq2_xs.json"))
        self.assertTrue(self.v.exists("Strata-data/packs/swift-iq2_xs"))

    def test_orphans_are_setups_whose_mounted_model_is_gone(self):
        self.v.setup("iq2_xs", flash_config("iq2_xs", "qwen3.8-flash-next-iq2_xs"))
        self.v.setup("qwen36-ud-q4_k_xl", {"args": ["--native", "/models/Qwen3.6.gguf"],
                                           "model_name": "qwen3.6-35b-a3b-ud-q4_k_xl"})
        # Strata downloaded this one itself: its model lives on the volume
        self.v.setup("iq3_xxs", {"args": ["--native", "/data/Strata-data/models/Q-IQ3_XXS.gguf"],
                                 "model_name": "qwen3.8-flash-next-iq3_xxs"})
        write(os.path.join(self.v.root, "Strata-data/packs/stale/experts.bin"))
        plan = self.v.run(orphans=True, present=["qwen36-ud-q4_k_xl"])
        self.assertEqual(plan["tags"], ["iq2_xs", "stale"])
        paths = {i["path"] for i in plan["items"]}
        self.assertIn("/data/Strata-data/packs/stale", paths)
        self.assertFalse(any("qwen36" in p or "iq3_xxs" in p for p in paths), paths)

    def test_a_present_tag_is_never_removed(self):
        self.v.setup("iq2_xs", flash_config("iq2_xs", "qwen3.8-flash-next-iq2_xs"))
        plan = self.v.run(tags=["iq2_xs"], present=["iq2_xs"], apply=True)
        self.assertEqual(plan["items"], [])
        self.assertTrue(self.v.exists("Strata-data/packs/iq2_xs"))

    def test_an_unreadable_remaining_config_keeps_shared_files(self):
        self.v.setup("iq2_xs", flash_config("iq2_xs", "qwen3.8-flash-next-iq2_xs"))
        self.v.flash_shared()
        write(os.path.join(self.v.root, "install/strata-broken.json"), b"{not json")
        self.v.run(tags=["iq2_xs"], apply=True)
        self.assertFalse(self.v.exists("Strata-data/packs/iq2_xs"))
        self.assertTrue(self.v.exists("Strata-data/mtp"))
        self.assertTrue(self.v.exists("Strata-data/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf"))

    def test_bad_tags_and_paths_outside_the_volume_are_refused(self):
        outside = os.path.join(self._tmp.name, "..", "keep-me")
        cfg = flash_config("iq2_xs", "qwen3.8-flash-next-iq2_xs")
        cfg["args"] += ["--x", "/data/../keep-me", "--y", "/etc/passwd"]
        self.v.setup("iq2_xs", cfg)
        plan = self.v.run(tags=["../..", "", "iq2_xs"])
        self.assertEqual(plan["tags"], ["iq2_xs"])
        for i in plan["items"]:
            self.assertTrue(i["path"].startswith(("/data/Strata-data/", "/data/install/")), i)
        self.assertFalse(os.path.exists(outside))

    def test_an_empty_volume(self):
        self.assertEqual(self.v.run(orphans=True)["items"], [])


class LlamamanSideTests(unittest.TestCase):

    def test_tags_for_files(self):
        with tempfile.TemporaryDirectory() as d:
            files = [os.path.join(d, FLASH.format(i)) for i in (1, 2)]
            self.assertEqual(C.tags_for_files(files + [os.path.join(d, "readme.txt")]), {"iq2_xs"})

    def test_model_files_under_a_folder(self):
        with tempfile.TemporaryDirectory() as d:
            write(os.path.join(d, "IQ2_XS", FLASH.format(1)))
            write(os.path.join(d, "IQ2_XS", "notes.md"))
            self.assertEqual([os.path.basename(p) for p in C.model_files_under(d)], [FLASH.format(1)])

    def test_run_asks_the_container_and_forgets_the_setups(self):
        result = {"items": [{"path": "/data/Strata-data/packs/iq2_xs", "bytes": 5}], "bytes": 5,
                  "tags": ["iq2_xs"], "applied": True, "errors": []}
        client = Mock()
        client.containers.run.return_value = (json.dumps(result) + "\n").encode()
        eng = Mock()
        eng.data_mount_source.return_value = "/mnt/e/strata"
        with patch("config.STRATA_ENABLED", True), \
                patch.dict("core.engines.ENGINES", {"strata": eng}), \
                patch("core.helpers.get_docker_client", return_value=client), \
                patch.object(C, "_image", return_value="strata:latest"), \
                patch.object(C, "present_tags", return_value={"iq3_s"}):
            self.assertEqual(C.run(["iq2_xs", "iq3_s"], apply=True), result)
        kw = client.containers.run.call_args.kwargs
        req = json.loads(kw["environment"]["LLAMAMAN_CLEANUP"])
        self.assertEqual(req["tags"], ["iq2_xs"])
        self.assertEqual(req["present"], ["iq3_s"])
        self.assertEqual(kw["volumes"], {"/mnt/e/strata": {"bind": "/data", "mode": "rw"}})
        self.assertTrue(kw["network_disabled"])
        eng.forget_setups.assert_called_once_with(["iq2_xs"])

    def test_nothing_to_do_starts_no_container(self):
        client = Mock()
        with patch("config.STRATA_ENABLED", True), \
                patch("core.helpers.get_docker_client", return_value=client), \
                patch.object(C, "present_tags", return_value={"iq2_xs"}):
            self.assertEqual(C.run(["iq2_xs"], apply=True)["items"], [])
        client.containers.run.assert_not_called()


class ForgetSetupsTests(unittest.TestCase):

    def test_forget_setups_drops_only_those_tags(self):
        import core.engines.strata as S
        eng = S.StrataEngine()
        with tempfile.TemporaryDirectory() as d, patch("config.DATA_DIR", d), \
                patch("config.HOST_STRATA_DATA_DIR", "/v"):
            with open(os.path.join(d, "strata_setups.json"), "w") as f:
                json.dump({"/v|iq2_xs": {}, "/v|iq3_s": {}, "/other|iq2_xs": {}}, f)
            eng.forget_setups(["iq2_xs"])
            with open(os.path.join(d, "strata_setups.json")) as f:
                self.assertEqual(set(json.load(f)), {"/v|iq3_s", "/other|iq2_xs"})


if __name__ == "__main__":
    unittest.main()
