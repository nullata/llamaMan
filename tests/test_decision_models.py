"""Decision models (/v1/systemone): detected from <arch>.decision.type in the
GGUF, refused on the text-generation endpoints, and launched with
--batch-size / --ubatch-size when set."""

import os
import unittest
from unittest.mock import patch

from flask import Flask

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import api.llamaman as llamaman_api
from api.models import decision_info, get_gguf_metadata, model_decision_type
from core.engines import ENGINES
from core.helpers import build_llama_cmd


class DecisionInfoTests(unittest.TestCase):
    def test_qwen_decision(self):
        info = decision_info({"general.architecture": "qwen35", "qwen35.decision.type": "openjev"})
        self.assertEqual(info, {"decision_type": "openjev", "non_causal": False})

    def test_encoder_decision(self):
        self.assertTrue(decision_info({"general.architecture": "modern-bert",
                                       "modern-bert.decision.type": "laya"})["non_causal"])
        self.assertTrue(decision_info({"general.architecture": "x", "x.attention.causal": False,
                                       "x.decision.type": "julia"})["non_causal"])

    def test_not_decision(self):
        self.assertEqual(decision_info({"general.architecture": "llama"}), {"decision_type": ""})
        self.assertEqual(decision_info({}), {"decision_type": ""})

    def test_metadata_summary_and_lookup(self):
        full = {"general.architecture": "bert", "bert.context_length": 8192,
                "bert.decision.type": "laya"}
        with patch("api.models.get_cached_gguf_metadata", return_value=full):
            meta = get_gguf_metadata("/models/laya.gguf")
            self.assertEqual((meta["decision_type"], meta["non_causal"], meta["context_length"]),
                             ("laya", True, 8192))
            self.assertEqual(model_decision_type("/models/laya.gguf"), "laya")
            self.assertEqual(model_decision_type("/strata/qwen-iq2_xs"), "")


class BatchFlagTests(unittest.TestCase):
    def test_flags_emitted_only_when_set(self):
        cmd = build_llama_cmd("/models/m.gguf", 8000, {"batch_size": 8192, "ubatch_size": 8192})
        self.assertEqual(cmd[cmd.index("--batch-size") + 1], "8192")
        self.assertEqual(cmd[cmd.index("--ubatch-size") + 1], "8192")
        cmd = build_llama_cmd("/models/m.gguf", 8000, {"batch_size": None, "ubatch_size": ""})
        self.assertNotIn("--batch-size", cmd)
        self.assertNotIn("--ubatch-size", cmd)

    def test_strata_rejects_batch_sizes(self):
        self.assertTrue(ENGINES["strata"].reject_unsupported_fields({"ubatch_size": 4096}))
        self.assertFalse(ENGINES["strata"].reject_unsupported_fields({"ubatch_size": ""}))


@patch("api.cluster.dispatch_inference", return_value=None)
class TextEndpointsRefuseDecisionTests(unittest.TestCase):
    def _post(self, path, body):
        app = Flask(__name__)
        app.register_blueprint(llamaman_api.bp)
        inst = {"id": "i1", "port": 8000, "status": "healthy",
                "model_path": "/models/openjev.gguf", "config": {}}
        with patch.object(llamaman_api, "_ensure_model_running", return_value=(inst, None)), \
                patch.object(llamaman_api, "model_decision_type", return_value="openjev"), \
                patch.object(llamaman_api, "request_local_worker") as worker:
            r = app.test_client().post(path, json=body)
        worker.assert_not_called()
        return r

    def test_chat_and_completions(self, _):
        msgs = [{"role": "user", "content": "hi"}]
        for path, body in (("/v1/chat/completions", {"model": "openjev", "messages": msgs}),
                           ("/v1/completions", {"model": "openjev", "prompt": "hi"}),
                           ("/api/chat", {"model": "openjev", "messages": msgs, "stream": False})):
            r = self._post(path, body)
            self.assertEqual(r.status_code, 422, path)
            self.assertIn("decision model", r.get_data(as_text=True), path)


if __name__ == "__main__":
    unittest.main()
