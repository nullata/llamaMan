# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""#110: Strata's single-file engine-path families (setup.py OWN_PATH_FAMILIES,
Qwen3.6-35B-A3B), fine-tunes from the model library included."""

import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import Mock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import core.engines as engines
import core.engines.strata as S
from core.engines import parse_engine, virtual_model_key

QWEN36 = "/strata/qwen36-UD-Q4_K_XL"
PUBLISHED = "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
FINE_TUNE = "Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf"
ENGINE = S.StrataEngine()

# The shape of upstream's tables at the time of writing, trimmed.
FAKE_SETUP = textwrap.dedent('''
    FAMILIES = {"qwen": {"title": "Q", "name": "q", "hf": "https://h/a/resolve/r/{q}/",
                         "file": "Q-{q}-0000{i}-of-00002.gguf"},
                "gyro": {"title": "G", "name": "g", "hf": "https://h/g/resolve/r/", "file": "G-{q}.gguf"}}
    MODELS = {"IQ2_XS": {"download_gb": 68.0, "ram_gb": 48, "families": ("qwen",)}}
    HF_REVISIONS = {"a": "r", "unsloth/Qwen3.6-35B-A3B-MTP-GGUF": "r2"}
    FILES = {"Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf": (22853663008, "x"), "mmproj-F16.gguf": (899283584, "y")}
    OWN_PATH_FAMILIES = {"qwen36": {
        "title": "Qwen3.6-35B-A3B", "arch": "qwen35moe", "name": "qwen3.6-35b-a3b", "tag": "qwen36-",
        "hf": "https://h/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/r2/", "file": "Qwen3.6-35B-A3B-{q}.gguf",
        "mmproj": "mmproj-F16.gguf", "sha256": FILES, "vision": True,
        "models": {"UD-Q4_K_XL": {"about": "4-bit", "ram_gb": 32}}}}
''')


def run_dump(setup_source):
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "setup.py"), "w") as f:
            f.write(setup_source)
        out = subprocess.run([sys.executable, "-c", S._DUMP_SCRIPT], cwd=d, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


class CatalogueTests(unittest.TestCase):

    def test_the_dump_script_reads_upstreams_table(self):
        dump = run_dump(FAKE_SETUP)
        self.assertEqual(list(dump["own_path"]), ["qwen36"])
        self.assertEqual(dump["own_path"]["qwen36"]["models"]["UD-Q4_K_XL"],
                         {"ram_gb": 32, "bytes": 22853663008})
        self.assertNotIn("gyro", dump["families"])      # other single files stay out

    def test_the_older_table_name_is_read_too(self):
        dump = run_dump(FAKE_SETUP.replace("OWN_PATH_FAMILIES", "PREVIEW_FAMILIES"))
        self.assertEqual(list(dump["own_path"]), ["qwen36"])

    def test_an_image_without_the_table_still_dumps(self):
        dump = run_dump(FAKE_SETUP.replace("OWN_PATH_FAMILIES", "SOMETHING_ELSE"))
        self.assertEqual(dump["own_path"], {})

    def test_the_dump_converts(self):
        families, sizes, _ = S.catalogue_from_dump(run_dump(FAKE_SETUP))
        fam = families["qwen36"]
        self.assertEqual((fam["shards"], fam["subdir"], fam["arch"]), (1, False, "qwen35moe"))
        self.assertEqual(fam["models"]["UD-Q4_K_XL"], {"download_gb": 22.9, "ram_gb": 32})
        self.assertFalse(fam["vision"])
        self.assertNotIn("UD-Q4_K_XL", sizes)

    def test_an_own_path_family_that_doesnt_fit_costs_only_itself(self):
        dump = run_dump(FAKE_SETUP)
        dump["own_path"]["qwen36"]["hf"] = "https://h/unpinned/"
        with self.assertLogs("llamaman", level="WARNING"):
            families, _, _ = S.catalogue_from_dump(dump)
        self.assertNotIn("qwen36", families)
        self.assertIn("qwen", families)

    def test_a_split_family_still_needs_its_shard_pattern(self):
        dump = run_dump(FAKE_SETUP)
        dump["families"]["qwen"]["file"] = "Q-{q}.gguf"
        with self.assertRaises(ValueError):
            S.catalogue_from_dump(dump)

    def test_a_dump_from_before_own_path_still_converts(self):
        dump = run_dump(FAKE_SETUP)
        del dump["own_path"]
        self.assertNotIn("qwen36", S.catalogue_from_dump(dump)[0])

    def test_a_cached_dump_from_before_own_path_is_read_again(self):
        # Same image id, older script: the cached tables have no own_path.
        self.assertEqual(S._cache_key("sha256:aaa"), "sha256:aaa#v4")

    def test_same_named_sizes_dont_collide(self):
        self.assertEqual(S.family_sizes("qwen36")["UD-Q4_K_XL"]["download_gb"], 22.9)
        self.assertEqual(S.family_sizes("unsloth")["UD-Q4_K_XL"]["download_gb"], 111.3)
        self.assertEqual(S.model_attr("qwen36", "UD-Q4_K_XL", "shards"), 1)
        self.assertEqual(S.model_attr("unsloth", "UD-Q4_K_XL", "shards"), 4)
        self.assertFalse(S.model_attr("qwen36", "UD-Q4_K_XL", "vision"))
        self.assertTrue(S.model_attr("unsloth", "UD-Q4_K_XL", "vision"))

    def test_a_size_of_another_family_is_refused(self):
        self.assertIsNone(S.parse_model_path("/strata/qwen36-IQ2_XS"))
        self.assertIsNone(S.parse_model_path("/strata/unsloth-Q8_0"))
        self.assertEqual(S.parse_model_path("strata/QWEN36-q8_0"), ("qwen36", "Q8_0"))

    def test_upstreams_names_for_it(self):
        self.assertEqual(S.setup_tag("qwen36", "UD-Q4_K_XL"), "qwen36-ud-q4_k_xl")
        self.assertEqual(S.shard_files("qwen36", "Q8_0"), ["Qwen3.6-35B-A3B-Q8_0.gguf"])
        self.assertEqual(S.repo_files("qwen36", "Q8_0"), ["Qwen3.6-35B-A3B-Q8_0.gguf"])
        self.assertEqual(ENGINE.served_model_names(QWEN36),
                         ["strata/qwen36-ud-q4_k_xl", "qwen3.6-35b-a3b-ud-q4_k_xl"])
        self.assertEqual(virtual_model_key("qwen3.6-35b-a3b-ud-q4_k_xl"), "strata/qwen36-ud-q4_k_xl")

    def test_the_download_plan_uses_the_familys_own_size(self):
        plan = ENGINE.download_plan(QWEN36)
        self.assertEqual(plan["repo_id"], "unsloth/Qwen3.6-35B-A3B-MTP-GGUF")
        self.assertEqual(plan["filename"], PUBLISHED)
        self.assertEqual(plan["revision"], "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d")
        self.assertEqual(plan["download_gb"], 22.9)

    def test_metadata_reports_its_architecture(self):
        meta = ENGINE.model_metadata(QWEN36)
        self.assertEqual(meta["general.architecture"], "qwen35moe")
        self.assertEqual(meta["qwen35moe.context_length"], S.DEFAULT_CONTEXT)
        self.assertEqual(ENGINE.model_metadata("/strata/qwen-IQ2_XS")["general.architecture"], "qwen3.8-flash-next")

    def test_the_library_entry_tells_the_form_what_it_is(self):
        entry = next(m for m in ENGINE.virtual_models() if m["path"] == QWEN36)
        self.assertTrue(entry["own_path"])
        self.assertEqual(entry["max_context"], 262144)
        other = next(m for m in ENGINE.virtual_models() if m["path"] == "/strata/qwen-IQ2_XS")
        self.assertFalse(other["own_path"])
        self.assertIsNone(other["max_context"])


class FileMatchingTests(unittest.TestCase):

    def test_the_published_file(self):
        self.assertEqual(S.model_for_file(f"/m/{PUBLISHED}"), ("qwen36", "UD-Q4_K_XL"))

    def test_a_fine_tune_by_name(self):
        self.assertEqual(S.model_for_file(f"/m/{FINE_TUNE}"), ("qwen36", "UD-Q4_K_XL"))

    def test_a_fine_tune_by_architecture(self):
        with patch("api.models.get_cached_gguf_metadata", return_value={"general.architecture": "qwen35moe"}):
            self.assertEqual(S.model_for_file("/m/Tiel-Q8_0.gguf"), ("qwen36", "Q8_0"))

    def test_another_model_of_that_quant_is_not_it(self):
        with patch("api.models.get_cached_gguf_metadata", return_value={"general.architecture": "llama"}):
            self.assertIsNone(S.model_for_file("/m/llama-3-Q8_0.gguf"))

    def test_an_unreadable_header_is_not_it(self):
        with patch("api.models.get_cached_gguf_metadata", side_effect=OSError("nope")):
            self.assertIsNone(S.model_for_file("/m/llama-3-Q8_0.gguf"))

    def test_a_size_setup_doesnt_know_is_not_it(self):
        self.assertIsNone(S.model_for_file("/m/Tiel-35B-A3B-Q5_K_M.gguf"))

    def test_no_header_is_read_without_a_matching_quant(self):
        with patch("api.models.get_cached_gguf_metadata") as read:
            self.assertIsNone(S.model_for_file("/m/llama-3-Q5_K_M.gguf"))
        read.assert_not_called()

    def test_split_files_still_match_flash_next(self):
        name = "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
        self.assertEqual(S.model_for_file(f"/m/{name}"), ("unsloth", "UD-Q4_K_XL"))

    def test_a_split_fine_tune_of_it_is_not_offered(self):
        # A single-file path: setup wants one file.
        self.assertIsNone(S.model_for_file("/m/Tiel-35B-A3B-Q8_0-00001-of-00002.gguf"))

    def test_the_library_tags_it_and_the_engine_is_accepted(self):
        self.assertEqual(engines.file_engine_models(f"/m/{FINE_TUNE}"),
                         {"strata": "strata/qwen36-UD-Q4_K_XL"})
        name, err = parse_engine({"engine": "strata"}, f"/m/{FINE_TUNE}")
        self.assertEqual((name, err), ("strata", None))


class MountTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.models = self.tmp.name
        for target, value in (("config.MODELS_DIR", self.models), ("config.HOST_MODELS_DIR", "/host/models")):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _file(self, name, sub=""):
        path = os.path.join(self.models, sub, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("gguf")
        return path

    def test_a_fine_tune_is_mounted_under_the_published_name(self):
        path = self._file(FINE_TUNE)
        volumes = ENGINE._shard_volumes(QWEN36, {"engine_source_path": path})
        self.assertEqual(volumes, {f"/host/models/{FINE_TUNE}": {"bind": f"/models/{PUBLISHED}", "mode": "rw"}})

    def test_the_published_name_mounts_only_the_file_too(self):
        # It may sit in MODELS_DIR itself: the whole library must not go in, rw.
        path = self._file(PUBLISHED)
        volumes = ENGINE._shard_volumes(QWEN36, {"engine_source_path": path})
        self.assertEqual(volumes, {f"/host/models/{PUBLISHED}": {"bind": f"/models/{PUBLISHED}", "mode": "rw"}})

    def test_llamamans_own_download_mounts_its_folder(self):
        self._file(PUBLISHED, "strata/qwen36-ud-q4_k_xl")
        volumes = ENGINE._shard_volumes(QWEN36, {})
        self.assertEqual(volumes, {"/host/models/strata/qwen36-ud-q4_k_xl": {"bind": "/models", "mode": "rw"}})

    def test_a_missing_file_blocks_the_launch(self):
        self.assertIn("not downloaded", ENGINE.launch_blocker(QWEN36))
        self.assertIsNone(ENGINE.launch_blocker(self._file(FINE_TUNE)))

    def test_launching_a_fine_tune_from_the_library(self):
        # The real launch path: the file becomes the virtual model, its gate
        # follows the slots, and the file is what gets mounted.
        import api.instances as instances_api
        from core.state import instances, instances_lock
        src = self._file(FINE_TUNE)
        fake = Mock()
        fake.id = "cid"
        with instances_lock:
            saved = dict(instances)
            instances.clear()

        def restore():
            with instances_lock:
                instances.clear()
                instances.update(saved)
        self.addCleanup(restore)
        options, err = ENGINE.parse_options({"strata_parallel": 4, "ctx_size": 32768}, src)
        self.assertIsNone(err)
        with patch("api.instances._run_container", return_value=(fake, None)) as run, \
                patch("api.instances.is_port_available", return_value=True), \
                patch("api.instances.find_available_port", return_value=9005), \
                patch("api.instances.save_state"), patch("api.instances.start_idle_proxy"), \
                patch("api.instances.create_gate") as gate, \
                patch("api.instances._publish_cluster_heartbeat_safe"):
            inst, err = instances_api.launch_instance(
                model_path=src, port=8001, ctx_size=32768, engine="strata",
                max_concurrent=4, engine_options=options)
        self.assertIsNone(err)
        self.assertEqual(inst["model_path"], QWEN36)
        config = inst["config"]
        self.assertEqual((config["engine_source_path"], config["max_concurrent"], config["strata_parallel"]),
                         (src, 4, 4))
        self.assertEqual(gate.call_args.args[1], 4)
        run_config = run.call_args.args[4]
        self.assertEqual(ENGINE._shard_volumes(QWEN36, run_config),
                         {f"/host/models/{FINE_TUNE}": {"bind": f"/models/{PUBLISHED}", "mode": "rw"}})
        self.assertEqual(ENGINE.environment(QWEN36, run_config)["PARALLEL"], "4")

    def test_it_is_passed_as_the_gguf_dir(self):
        path = self._file(FINE_TUNE)
        env = ENGINE.environment(QWEN36, {"engine_source_path": path, "strata_parallel": 1})
        self.assertEqual(env["GGUF_DIR"], "/models")
        self.assertEqual((env["FAMILY"], env["MODEL"], env["VISION"]), ("qwen36", "UD-Q4_K_XL", "no"))


def _write_gguf(path, tensors):
    """A GGUF with one KV and a tensor directory of (name, ggml type id) - no data."""
    import struct
    def string(s):
        b = s.encode()
        return struct.pack("<Q", len(b)) + b
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 1)
    out += string("general.architecture") + struct.pack("<I", 8) + string("qwen35moe")
    for name, ttype in tensors:
        out += string(name) + struct.pack("<I", 2) + struct.pack("<QQ", 2048, 512) \
            + struct.pack("<IQ", ttype, 0)
    with open(path, "wb") as f:
        f.write(out)


Q3_K, Q4_K, Q5_K, Q6_K = 11, 12, 13, 14


class ExpertTypeTests(MountTests):
    """A build whose experts the native kernels don't take is refused at launch,
    not left to kill the engine after setup."""

    def _gguf(self, mtp_gate_up):
        path = os.path.join(self.models, FINE_TUNE)
        tensors = []
        for layer, (gate_up, down) in enumerate(((Q4_K, Q5_K), (Q4_K, Q6_K), (mtp_gate_up, Q5_K))):
            tensors += [(f"blk.{layer}.ffn_gate_exps.weight", gate_up),
                        (f"blk.{layer}.ffn_up_exps.weight", gate_up),
                        (f"blk.{layer}.ffn_down_exps.weight", down),
                        (f"blk.{layer}.attn_q.weight", Q3_K)]
        _write_gguf(path, tensors)
        return path

    def test_the_published_types_launch(self):
        path = self._gguf(Q4_K)
        self.assertEqual(S.unsupported_expert_types(path), {})
        self.assertIsNone(ENGINE.launch_blocker(path))

    def test_q3_k_experts_block_the_launch(self):
        path = self._gguf(Q3_K)
        self.assertEqual(S.unsupported_expert_types(path), {2: ["Q3_K"]})
        why = ENGINE.launch_blocker(path)
        self.assertIn("layer 2 are Q3_K", why)
        self.assertIn("unsloth/Qwen3.6-35B-A3B-MTP-GGUF", why)

    def test_an_unreadable_header_is_left_to_the_engine(self):
        self.assertEqual(S.unsupported_expert_types(self._file(FINE_TUNE)), {})


class SetupFingerprintTests(MountTests):
    """Two fine-tunes of a size are one virtual model with one run config."""

    def setUp(self):
        super().setUp()
        patcher = patch("config.DATA_DIR", self.models)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _ready(self, config):
        ENGINE.on_ready({"model_path": QWEN36, "config": config})

    def test_a_different_file_of_that_size_sets_up_again(self):
        a = {"engine_source_path": self._file("Tiel-35B-A3B-UD-Q4_K_XL.gguf")}
        b = {"engine_source_path": self._file("Other-35B-A3B-UD-Q4_K_XL.gguf")}
        self._ready(a)
        self.assertFalse(ENGINE.needs_reinstall(QWEN36, a))
        self.assertTrue(ENGINE.needs_reinstall(QWEN36, b))
        self.assertEqual(ENGINE.environment(QWEN36, b)["REINSTALL"], "1")

    def test_going_back_to_the_first_sets_up_again(self):
        # B rewrote the one config, so "A was fine once" means nothing now.
        a = {"engine_source_path": self._file("Tiel-35B-A3B-UD-Q4_K_XL.gguf")}
        b = {"engine_source_path": self._file("Other-35B-A3B-UD-Q4_K_XL.gguf")}
        self._ready(a)
        self._ready(b)
        self.assertTrue(ENGINE.needs_reinstall(QWEN36, a))

    def test_a_replaced_file_sets_up_again(self):
        path = self._file(FINE_TUNE)
        config = {"engine_source_path": path}
        self._ready(config)
        os.utime(path, (1, 1))
        self.assertTrue(ENGINE.needs_reinstall(QWEN36, config))

    def test_the_parallel_count_needs_no_setup(self):
        config = {"engine_source_path": self._file(FINE_TUNE), "strata_parallel": 1}
        self._ready(config)
        self.assertFalse(ENGINE.needs_reinstall(QWEN36, {**config, "strata_parallel": 4}))

    def test_flash_next_fingerprints_are_unchanged(self):
        # A new key would re-set-up every existing install on upgrade.
        self.assertNotIn("source", ENGINE.setup_settings("/strata/qwen-IQ2_XS", {}))


class ParallelTests(unittest.TestCase):

    def _enforce(self, path, **config):
        return ENGINE.enforce_capabilities(config, path)["max_concurrent"]

    def test_the_gate_follows_the_slots(self):
        self.assertEqual(self._enforce(QWEN36, max_concurrent=4, strata_parallel=4), 4)
        self.assertEqual(self._enforce(QWEN36, max_concurrent=0, strata_parallel=4), 4)
        self.assertEqual(self._enforce(QWEN36, max_concurrent=8, strata_parallel=2), 2)

    def test_it_defaults_to_one(self):
        self.assertEqual(self._enforce(QWEN36, max_concurrent=4), 1)

    def test_a_bad_value_is_one(self):
        self.assertEqual(self._enforce(QWEN36, max_concurrent=4, strata_parallel="lots"), 1)

    def test_it_is_capped(self):
        self.assertEqual(self._enforce(QWEN36, max_concurrent=99, strata_parallel=99), 8)

    def test_flash_next_stays_at_one(self):
        self.assertEqual(self._enforce("/strata/qwen-IQ2_XS", max_concurrent=4, strata_parallel=4), 1)

    def test_no_model_path_stays_at_one(self):
        self.assertEqual(ENGINE.enforce_capabilities({"max_concurrent": 4, "strata_parallel": 4})["max_concurrent"], 1)

    def test_the_environment(self):
        self.assertEqual(ENGINE.environment(QWEN36, {"strata_parallel": 4})["PARALLEL"], "4")
        self.assertEqual(ENGINE.environment(QWEN36, {})["PARALLEL"], "1")
        self.assertNotIn("PARALLEL", ENGINE.environment("/strata/qwen-IQ2_XS", {}))

    def _patch_config(self, start, parallel):
        """Run the entrypoint's own one-liner on a config file."""
        code = re.search(r'''"\$PY" -c '(import json, sys; p = sys\.argv\[1\]; n = int.*?)' ''',
                         S.ENTRY_SCRIPT.replace("\\\n", " ")).group(1)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cfg.json")
            with open(path, "w") as f:
                json.dump(start, f)
            subprocess.run([sys.executable, "-c", code, path, str(parallel)], check=True)
            with open(path) as f:
                return json.load(f)

    def test_the_entrypoint_writes_it(self):
        self.assertEqual(self._patch_config({"exe": "x"}, 4), {"exe": "x", "parallel": 4})

    def test_the_entrypoint_clears_it_at_one(self):
        self.assertEqual(self._patch_config({"exe": "x", "parallel": 4}, 1), {"exe": "x"})
        self.assertEqual(self._patch_config({"exe": "x", "parallel": 1}, 1), {"exe": "x"})


class EntrypointTests(unittest.TestCase):

    def test_a_setup_rebuilds_the_tokenizer(self):
        # setup keeps a pack whose tokenizer exists: B would get A's.
        script = S.ENTRY_SCRIPT
        wipe = script.index('rm -rf "/data/Strata-data/packs/$tag"')
        self.assertLess(script.index('if [ "${REINSTALL:-0}" = "1" ]'), wipe)
        self.assertLess(wipe, script.index('"$PY" setup.py "$@"'))
        self.assertIn('if [ -n "${PARALLEL:-}" ]; then rm -rf', script)

    def test_the_config_name_matches_upstreams_tag(self):
        # fam["tag"] + model, lowercased: FAMILY-MODEL.
        out = subprocess.run(
            ["sh", "-c", 'FAMILY=qwen36 MODEL=UD-Q4_K_XL; case "$FAMILY" in qwen) prefix="" ;; *) prefix="$FAMILY-" ;; esac; '
                         '''printf '%s%s' "$prefix" "$MODEL" | tr 'A-Z' 'a-z' '''],
            capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout, S.setup_tag("qwen36", "UD-Q4_K_XL"))


class OptionTests(unittest.TestCase):

    def _parse(self, path=QWEN36, **body):
        return ENGINE.parse_options(body, path)

    def test_a_valid_count(self):
        options, err = self._parse(strata_parallel=4)
        self.assertIsNone(err)
        self.assertEqual(options["strata_parallel"], 4)

    def test_the_default_is_one(self):
        for body in ({}, {"strata_parallel": ""}, {"strata_parallel": None}):
            self.assertEqual(self._parse(**body)[0]["strata_parallel"], 1)

    def test_out_of_range_is_refused(self):
        for bad in (0, 9, -1, "lots", 2.5):
            with self.subTest(bad=bad):
                self.assertIn("strata_parallel", self._parse(strata_parallel=bad)[1])

    def test_flash_next_refuses_more_than_one(self):
        path = "/strata/qwen-IQ2_XS"
        self.assertIn("one request at a time", self._parse(path, strata_parallel=2)[1])
        options, err = self._parse(path, strata_parallel=1)
        self.assertIsNone(err)
        self.assertNotIn("strata_parallel", options)

    def test_context_past_its_trained_length_is_refused(self):
        self.assertIn("262144", self._parse(ctx_size=393216)[1])
        self.assertIsNone(self._parse(ctx_size=262144)[1])
        self.assertIsNone(self._parse("/strata/qwen-IQ2_XS", ctx_size=393216)[1])

    def test_flash_next_settings_are_dropped(self):
        options, err = self._parse(strata_kv="int8", strata_low_ram="on", strata_layer_split="18")
        self.assertIsNone(err)
        self.assertEqual((options["strata_kv"], options["strata_low_ram"], options["strata_layer_split"]),
                         ("", "auto", ""))

    def test_images_are_off_for_now(self):
        self.assertIn("no image encoder", self._parse(strata_vision="yes")[1])
        self.assertEqual(self._parse(strata_vision="no")[0]["strata_vision"], "no")

    def test_a_bad_watchdog_is_refused(self):
        self.assertIn("strata_watchdog_s", self._parse(strata_watchdog_s="soon")[1])
        self.assertEqual(self._parse(strata_watchdog_s="30")[0]["strata_watchdog_s"], "30")


class LoadStageTests(unittest.TestCase):

    def test_an_engine_too_old_for_the_family_says_so(self):
        log = ("=== Step 4: the Strata engine ===\n  [X]  this engine (0.2.1) does not run Qwen3.6-35B-A3B yet\n"
               "       update Strata\n\nSetup stopped. Fix the item above and run it again\n")
        stage = S.load_stage_from_log(log)
        self.assertEqual(stage["stage"], "setup failed")
        self.assertIn("can't run Qwen3.6-35B-A3B yet", stage["detail"])

    def test_any_other_failure_points_at_the_log(self):
        self.assertEqual(S.load_stage_from_log("Setup stopped. Fix it")["detail"], "see the log")

    def test_writing_the_start_script_is_not_loading_yet(self):
        stage = S.load_stage_from_log("=== Step 7: writing the start script ===\n")
        self.assertEqual(stage["stage"], "setting up")

    def test_its_own_steps_are_recognised(self):
        self.assertEqual(S.load_stage_from_log("=== Step 5: downloading Qwen3.6-35B-A3B Q8_0 ===\n")["stage"],
                         "downloading")
        self.assertEqual(S.load_stage_from_log("=== Step 6: preparing the model for Strata ===\n")["stage"],
                         "preparing pack")


class FormTests(unittest.TestCase):

    def _read(self, *parts):
        with open(os.path.join(REPO_ROOT, *parts), encoding="utf-8") as f:
            return f.read()

    def test_the_launch_form_has_the_control(self):
        html = self._read("templates", "index.html")
        self.assertIn('id="f-strata-parallel"', html)
        self.assertIn("1.5 GB of VRAM", html)

    def test_the_form_sends_it_and_matches_the_gate(self):
        js = self._read("static", "js", "engines.js")
        self.assertIn("strata_parallel: parallel", js)
        self.assertIn("max_concurrent: parallel", js)

    def test_the_engine_accepts_the_field(self):
        self.assertIn("strata_parallel", ENGINE.launch_fields)
        self.assertIn("strata_parallel", ENGINE.option_keys)


if __name__ == "__main__":
    unittest.main()
