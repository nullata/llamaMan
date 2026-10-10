"""llamaMan's API routes /v1/systemone (decision models) like completions:
model by name, auto-launch, gate, request log. The body is forwarded
untouched (no sampling overrides) and the answers + input tokens are
recorded."""

import os
import unittest
from unittest.mock import MagicMock, patch

from flask import Flask

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import api.llamaman as llamaman_api

ANSWER = {"model": "openjev",
          "answers": {"route": {"type": "choice", "choice": "billing",
                                "probabilities": {"billing": 0.99, "shipping": 0.01},
                                "confidence": 0.98}},
          "usage": {"input_tokens": 239, "output_tokens": 0}}
BODY = {"model": "openjev", "state": "charged twice",
        "questions": {"route": {"type": "choice", "instructions": "Which team?",
                                "criteria": {"billing": None, "shipping": None}}}}


def _resp(status=200, data=ANSWER):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = data
    r.__enter__.return_value = r
    return r


@patch("api.cluster.dispatch_inference", return_value=None)
@patch("api.llamaman.get_gate", return_value=None)
@patch("api.llamaman._touch_instance")
class SystemOneRouteTests(unittest.TestCase):
    def _post(self, inst_config, worker_resp, body=BODY):
        app = Flask(__name__)
        app.register_blueprint(llamaman_api.bp)
        handle = MagicMock()
        inst = {"id": "i1", "port": 8000, "status": "healthy", "config": inst_config}
        with patch.object(llamaman_api, "_ensure_model_running", return_value=(inst, None)), \
                patch.object(llamaman_api, "record_request", return_value=handle), \
                patch.object(llamaman_api, "finalize_async"), \
                patch.object(llamaman_api, "request_local_worker", return_value=worker_resp) as worker:
            r = app.test_client().post("/v1/systemone", json=body)
        return r, worker, handle

    def test_forwards_body_unchanged_and_records(self, *_):
        cfg = {"proxy_sampling_override_enabled": True, "proxy_sampling_temperature": 0.1,
               "exclude_from_max_models": True}
        r, worker, handle = self._post(cfg, _resp())
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["answers"]["route"]["choice"], "billing")
        url = worker.call_args.args[0]
        self.assertTrue(url.endswith("/v1/systemone"))
        self.assertEqual(worker.call_args.kwargs["json"], BODY)   # no sampling fields added
        kw = handle.set_response.call_args.kwargs
        self.assertEqual(kw["usage"], ANSWER["usage"])
        self.assertIn("billing", kw["text"])

    def test_upstream_error_passes_through(self, *_):
        err = {"error": {"code": 501, "message": "This model is not a decision model"}}
        r, _, handle = self._post({}, _resp(501, err))
        self.assertEqual(r.status_code, 501)
        self.assertIn("not a decision model", handle.set_response.call_args.kwargs["text"])

    def test_model_required(self, *_):
        r, worker, _ = self._post({}, _resp(), body={"state": "x", "questions": {}})
        self.assertEqual(r.status_code, 400)
        worker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
