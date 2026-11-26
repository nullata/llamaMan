# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Inkling-2-Mini on Strata: setup.py keeps it in a table of its own
(INKLING_FAMILIES, with its files in strata_control/inkling_family.py), which
the dump script reads next to FAMILIES / OWN_PATH_FAMILIES. It runs like an
own-path family (one file under its published name, --gguf-dir), with
per-size files and repos, a family key with dashes, 1M context and up to 64
requests at once."""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import core.engines.strata as S

FAM = "inkling-2-mini"
SQ20 = f"/strata/{FAM}-SQ-20"
Q4 = f"/strata/{FAM}-UD-Q4_K_XL"

# Upstream's shape at the time of writing, trimmed (terminal/setup.py and
# terminal/strata_control/inkling_family.py / inkling_fit.py).
FAKE_SETUP = textwrap.dedent('''
    FAMILIES = {"qwen": {"title": "Q", "name": "q", "hf": "https://h/a/resolve/r/{q}/",
                         "file": "Q-{q}-0000{i}-of-00002.gguf"}}
    MODELS = {"IQ2_XS": {"download_gb": 68.0, "ram_gb": 48, "families": ("qwen",)}}
    HF_REVISIONS = {"a": "r"}
    OWN_PATH_FAMILIES = {}
    from strata_control import inkling_family, inkling_fit
    HY4_FAMILIES = {"hy4-flash": {"title": "Hy4", "models": {"SQ-64": {"ram_gb": 64}}}}
    INKLING_FAMILIES = {inkling_family.FAMILY: {
        "title": "Inkling-2-Mini", "tag": inkling_family.TAG, "name": inkling_family.FAMILY,
        "arch": "inkling2_loop", "inkling": True, "vision": True,
        "models": {b: {"ram_gb": int(-(-inkling_fit.BUILDS[b]["footprint_gb"] // 1)),
                       "format": inkling_fit.BUILDS[b]["format"]} for b in inkling_fit.ORDER}}}
''')
FAKE_FIT = textwrap.dedent('''
    MAX_CONTEXT = 1048576
    BUILDS = {
        "SQ-20": {"file": "Inkling-2-Mini-SQ-20.strata", "footprint_gb": 19.35, "format": "strata"},
        "UD-Q4_K_XL": {"file": "Inkling-2-Mini-UD-Q4_K_XL.gguf", "footprint_gb": 17.63, "format": "gguf"},
    }
    ORDER = ("SQ-20", "UD-Q4_K_XL")
    def is_gguf_build(b):
        return BUILDS.get(b, {}).get("format") == "gguf"
''')
FAKE_FAMILY = textwrap.dedent('''
    from strata_control import inkling_fit
    FAMILY = "inkling-2-mini"
    TAG = "inkling-2-mini-"
    MAX_PARALLEL = 64
    FILES = {"Inkling-2-Mini-SQ-20.strata": (19_346_685_952, "a"),
             "Inkling-2-Mini-UD-Q4_K_XL.gguf": (17_629_759_456, "b")}
    def build_file(b):
        return inkling_fit.BUILDS[b]["file"]
    def is_gguf_build(b):
        return inkling_fit.is_gguf_build(b)
    def size_of(n):
        return FILES[n][0]
    def sources():
        return [{"kind": "http", "base": "https://huggingface.co/strata-engine/Inkling-2-Mini-Strata/resolve/main", "files": {}}]
    def gguf_sources():
        return [{"kind": "http", "base": "https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/800c946d", "files": {}}]
''')


def run_dump(setup=FAKE_SETUP, family=FAKE_FAMILY, fit=FAKE_FIT):
    with tempfile.TemporaryDirectory() as d:
        os.mkdir(os.path.join(d, "strata_control"))
        for name, src in (("setup.py", setup), ("strata_control/__init__.py", ""),
                          ("strata_control/inkling_family.py", family),
                          ("strata_control/inkling_fit.py", fit)):
            with open(os.path.join(d, name), "w") as f:
                f.write(src)
        out = subprocess.run([sys.executable, "-c", S._DUMP_SCRIPT], cwd=d,
                             capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


class DumpTests(unittest.TestCase):

    def test_the_dump_reads_inkling_families(self):
        dump = run_dump()
        self.assertEqual(list(dump["inkling"]), [FAM])
        ink = dump["inkling"][FAM]
        self.assertEqual(ink["models"]["SQ-20"], {"ram_gb": 20, "file": "Inkling-2-Mini-SQ-20.strata",
                                                  "bytes": 19346685952, "format": "strata"})
        self.assertEqual(ink["models"]["UD-Q4_K_XL"]["format"], "gguf")
        self.assertEqual((ink["max_context"], ink["parallel_max"]), (1048576, 64))
        self.assertNotIn("hy4-flash", json.dumps(dump))           # Hy4-Flash stays out

    def test_an_image_without_inkling_dumps_none(self):
        dump = run_dump(setup=FAKE_SETUP.split("from strata_control")[0])
        self.assertEqual(dump["inkling"], {})

    def test_conversion(self):
        families, _sizes, revisions = S.catalogue_from_dump(run_dump())
        fam = families[FAM]
        self.assertTrue(fam["own_path"] and fam["inkling"])
        self.assertEqual(fam["repo"], "strata-engine/Inkling-2-Mini-Strata")
        self.assertEqual(fam["models"]["SQ-20"], {"download_gb": 19.3, "ram_gb": 20,
                                                  "file": "Inkling-2-Mini-SQ-20.strata"})
        self.assertEqual(fam["models"]["UD-Q4_K_XL"]["repo"], "unsloth/Inkling-2-Mini-GGUF")
        self.assertEqual(revisions["strata-engine/Inkling-2-Mini-Strata"], "main")
        self.assertEqual(revisions["unsloth/Inkling-2-Mini-GGUF"], "800c946d")
        self.assertEqual(revisions["a"], "r")

    def test_a_source_llamaman_cant_fetch_leaves_those_sizes_out(self):
        family = FAKE_FAMILY.replace("https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/800c946d",
                                     "https://mirror.example/inkling")
        families, _, revisions = S.catalogue_from_dump(run_dump(family=family))
        self.assertEqual(list(families[FAM]["models"]), ["SQ-20"])
        self.assertNotIn("unsloth/Inkling-2-Mini-GGUF", revisions)

    def test_a_bad_inkling_entry_keeps_the_rest(self):
        dump = run_dump()
        dump["inkling"][FAM]["models"] = "?"
        families, _, _ = S.catalogue_from_dump(dump)
        self.assertNotIn(FAM, families)
        self.assertIn("qwen", families)


class CatalogueTests(unittest.TestCase):

    def test_ids_with_a_dashed_family(self):
        self.assertEqual(S.parse_model_path(SQ20), (FAM, "SQ-20"))
        self.assertEqual(S.parse_model_path(f"strata/INKLING-2-MINI-ud-q4_k_xl"), (FAM, "UD-Q4_K_XL"))
        for bad in (f"/strata/{FAM}", f"/strata/{FAM}-SQ-99", "/strata/inkling-SQ-20"):
            self.assertIsNone(S.parse_model_path(bad), bad)
        self.assertEqual(S.parse_model_path("/strata/qwen36-Q8_0"), ("qwen36", "Q8_0"))

    def test_tag_and_served_name_match_setup(self):
        # install_inkling2: tag = fam["tag"] + model -> strata-<tag.lower()>.json;
        # model_name = f"{fam['name']}-{model.lower()}"
        self.assertEqual(S.setup_tag(FAM, "SQ-20"), "inkling-2-mini-sq-20")
        by_id = {m["id"]: m for m in S.catalogue()}
        self.assertEqual(by_id[f"strata/{FAM}-SQ-20"]["served_name"], "inkling-2-mini-sq-20")

    def test_published_files_are_recognised(self):
        self.assertEqual(S.model_for_file("/m/Inkling-2-Mini-SQ-15.strata"), (FAM, "SQ-15"))
        self.assertEqual(S.model_for_file("/m/Inkling-2-Mini-UD-IQ1_S.gguf"), (FAM, "UD-IQ1_S"))
        self.assertIsNone(S.model_for_file("/m/other.strata"))

    def test_download_plan_per_size_repo(self):
        eng = S.StrataEngine()
        plan = eng.download_plan(SQ20)
        self.assertEqual((plan["repo_id"], plan["filename"], plan["revision"]),
                         ("strata-engine/Inkling-2-Mini-Strata", "Inkling-2-Mini-SQ-20.strata", "main"))
        plan = eng.download_plan(Q4)
        self.assertEqual((plan["repo_id"], plan["filename"], plan["revision"]),
                         ("unsloth/Inkling-2-Mini-GGUF", "Inkling-2-Mini-UD-Q4_K_XL.gguf",
                          "800c946d7b9c830c86290df49afbbe91db8455f5"))
        self.assertTrue(plan["dest_path"].endswith(os.path.join("strata", "inkling-2-mini-ud-q4_k_xl")))

    def test_virtual_model_limits(self):
        vm = {m["name"]: m for m in S.StrataEngine().virtual_models()}
        m = vm[f"strata/{FAM}-SQ-20"]
        self.assertEqual((m["max_context"], m["parallel_max"], m["vision"]), (1048576, 64, True))
        self.assertEqual(vm["strata/qwen36-Q8_0"]["parallel_max"], 8)


class LaunchTests(unittest.TestCase):

    def setUp(self):
        self.eng = S.StrataEngine()

    def test_options(self):
        opts, err = self.eng.parse_options({"strata_parallel": "64", "ctx_size": 1048576,
                                            "strata_vision": "yes"}, SQ20)
        self.assertIsNone(err)
        self.assertEqual(opts["strata_parallel"], 64)
        _, err = self.eng.parse_options({"strata_parallel": "65"}, SQ20)
        self.assertIn("1 to 64", err)
        _, err = self.eng.parse_options({"ctx_size": 2097152}, SQ20)
        self.assertIn("1048576", err)
        _, err = self.eng.parse_options({"strata_parallel": "9"}, "/strata/qwen36-Q8_0")
        self.assertIn("1 to 8", err)

    def test_concurrency_follows_slots(self):
        self.assertEqual(self.eng.max_concurrency_for({"strata_parallel": 32}, SQ20), 32)
        self.assertEqual(self.eng.max_concurrency_for({"strata_parallel": 32}, "/strata/qwen36-Q8_0"), 8)

    def test_environment(self):
        with patch.object(self.eng, "needs_reinstall", return_value=False), \
                patch.object(self.eng, "_shard_volumes", return_value={}):
            env = self.eng.environment(SQ20, {"ctx_size": 65536, "strata_parallel": 4, "strata_vision": "yes"})
        self.assertEqual((env["FAMILY"], env["MODEL"], env["CONTEXT"], env["PARALLEL"]),
                         (FAM, "SQ-20", "65536", "4"))
        self.assertNotIn("ROUTE_TAIL_SKIP", env)
        # the entrypoint's tag for that FAMILY / MODEL is the one setup writes
        self.assertIn('case "$FAMILY" in qwen) prefix="" ;; *) prefix="$FAMILY-" ;; esac', S.ENTRY_SCRIPT)
        self.assertIn('[ "$FAMILY" = "inkling-2-mini" ]; then set -- "$@" --parallel "$PARALLEL"', S.ENTRY_SCRIPT)

    def test_slots_are_part_of_the_setup(self):
        with patch.object(S.StrataEngine, "_source_identity", return_value="f"):
            a = self.eng.setup_settings(SQ20, {"strata_parallel": 1})
            b = self.eng.setup_settings(SQ20, {"strata_parallel": 8})
            q = self.eng.setup_settings("/strata/qwen36-Q8_0", {"strata_parallel": 8})
        self.assertNotEqual(a, b)
        self.assertNotIn("parallel", q)

    def test_no_llama_cpp_expert_type_check(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "Inkling-2-Mini-UD-Q4_K_XL.gguf")
            open(path, "wb").close()
            with patch.object(S, "unsupported_expert_types", return_value={3: ["NVFP4"]}) as check, \
                    patch.object(S, "model_download", return_value=None):
                self.assertIsNone(self.eng.launch_blocker(path))
            check.assert_not_called()


class LibraryTests(unittest.TestCase):
    """A downloaded .strata build is in the model library (the launch starts
    there) and runs on Strata only."""

    def test_claimed_strata_files_are_listed(self):
        from api.models import discover_models
        with tempfile.TemporaryDirectory() as d:
            folder = os.path.join(d, "strata", "inkling-2-mini-sq-20")
            os.makedirs(folder)
            for name in ("Inkling-2-Mini-SQ-20.strata", "Inkling-2-Mini-Vision.strata", "other.strata"):
                with open(os.path.join(folder, name), "wb") as f:
                    f.write(b"x")
            found = {os.path.basename(m["path"]): m for m in discover_models(d)}
        self.assertEqual(list(found), ["Inkling-2-Mini-SQ-20.strata"])
        m = found["Inkling-2-Mini-SQ-20.strata"]
        self.assertEqual((m["type"], m["quant"], m["engine_only"]), ("strata", "SQ-20", "strata"))
        self.assertEqual(m["engine_models"], {"strata": f"strata/{FAM}-SQ-20"})

    def test_strata_is_the_only_engine_for_them(self):
        from core.engines import file_only_engine, parse_engine
        path = "/models/strata/x/Inkling-2-Mini-SQ-20.strata"
        self.assertEqual(file_only_engine(path), "strata")
        self.assertEqual(parse_engine({}, path), ("strata", None))
        name, err = parse_engine({"engine": "llamacpp"}, path)
        self.assertIn("is a Strata model", err)
        # GGUFs keep llama.cpp as their default, Strata's included
        gguf = "/models/strata/x/Inkling-2-Mini-UD-Q4_K_XL.gguf"
        self.assertIsNone(file_only_engine(gguf))
        self.assertEqual(parse_engine({}, gguf), ("llamacpp", None))


if __name__ == "__main__":
    unittest.main()
