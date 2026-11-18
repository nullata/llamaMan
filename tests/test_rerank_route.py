# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Rerank on the main proxy (issue #116).

/v1/rerank, /v1/reranking and /rerank route by `model` like /v1/embeddings
and go to the instance's /v1/rerank (llama-server with --reranking). There is
no llamaman-side reranker check: llama-server's own 501 is passed through.
The embeddings route, now on the same helper, keeps its 422 for instances not
marked as embedding models.
"""

import json
import os
import unittest
from unittest.mock import Mock, patch

from flask import Flask

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import api.llamaman as llamaman
import proxy

RERANK_BODY = {"model": "bge-reranker-v2-m3", "query": "canon 915",
               "documents": ["Can. 915 ...", "carrot cake recipe"]}
RERANK_RESULT = {"model": "bge-reranker-v2-m3", "object": "list",
                 "usage": {"prompt_tokens": 12, "total_tokens": 12},
                 "results": [{"index": 0, "relevance_score": 0.98},
                             {"index": 1, "relevance_score": 0.01}]}


def _inst(**config):
    return {"id": "inst-1", "port": 8001, "status": "healthy",
            "_server_host": "llamaman-inst-1", "_server_port": 8080,
            "config": config}


def _upstream(status=200, payload=None, text=None):
    resp = Mock(status_code=status)
    if text is not None:
        resp.json.side_effect = ValueError("not json")
        resp.text = text
    else:
        resp.json.return_value = payload
        resp.text = json.dumps(payload)
    return resp


class _Case(unittest.TestCase):

    def setUp(self):
        self.app = Flask(__name__)
        self.app.register_blueprint(llamaman.bp)
        for target, value in (
            ("api.cluster.dispatch_inference", Mock(return_value=None)),
            ("api.llamaman._openai_can_evict_admin_instances", Mock(return_value=False)),
            ("api.llamaman.record_request", Mock(return_value=None)),
            ("api.llamaman._touch_instance", Mock()),
        ):
            p = patch(target, value)
            p.start()
            self.addCleanup(p.stop)

    def _post(self, path, body, inst=None, upstream=None, err=None):
        inst = inst if inst is not None else _inst()
        upstream = upstream if upstream is not None else _upstream(payload=RERANK_RESULT)
        with patch.object(llamaman, "_ensure_model_running", return_value=(None if err else inst, err)) as ensure, \
             patch.object(llamaman, "request_local_worker", return_value=upstream) as worker:
            r = self.app.test_client().post(path, json=body)
        return r, ensure, worker


class RerankRouteTests(_Case):

    def test_every_path_goes_to_v1_rerank(self):
        for path in ("/v1/rerank", "/v1/reranking", "/rerank"):
            with self.subTest(path=path):
                r, ensure, worker = self._post(path, RERANK_BODY)
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.get_json(), RERANK_RESULT)
                self.assertEqual(ensure.call_args.args[0], "bge-reranker-v2-m3")
                self.assertEqual(worker.call_args.args[0], "http://llamaman-inst-1:8080/v1/rerank")
                self.assertEqual(worker.call_args.kwargs["json"], RERANK_BODY)

    def test_model_required(self):
        for body in ({"query": "q", "documents": ["a"]}, {"model": "  ", "query": "q"}):
            r, ensure, _ = self._post("/v1/rerank", body)
            self.assertEqual(r.status_code, 400)
            ensure.assert_not_called()

    def test_reranker_needs_no_embedding_flag(self):
        # The reporter's preset sets --reranking in Extra Args only.
        r, _, worker = self._post("/v1/rerank", RERANK_BODY,
                                  inst=_inst(extra_args="--embedding --reranking --pooling rank"))
        self.assertEqual(r.status_code, 200)
        worker.assert_called_once()

    def test_llama_server_refusal_passed_through(self):
        err = {"error": {"code": 501, "message": "This server does not support reranking. "
                         "Start it with `--reranking`", "type": "not_supported_error"}}
        r, _, _ = self._post("/v1/rerank", RERANK_BODY, upstream=_upstream(501, err))
        self.assertEqual(r.status_code, 501)
        self.assertEqual(r.get_json(), err)

    def test_non_json_upstream_is_an_error(self):
        r, _, _ = self._post("/rerank", RERANK_BODY, upstream=_upstream(404, text="Not Found"))
        self.assertEqual(r.status_code, 404)
        self.assertEqual(r.get_json(), {"error": {"message": "Not Found"}})

    def test_tei_list_response_unchanged(self):
        tei = [{"index": 0, "score": 0.98}, {"index": 1, "score": 0.01}]
        r, _, _ = self._post("/rerank", {"model": "m", "query": "q", "texts": ["a", "b"]},
                             upstream=_upstream(payload=tei))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json(), tei)

    def test_typed_model_error_is_503(self):
        err = llamaman.ModelUnavailable("model_stopping", "model 'm' is stopping on this node")
        r, _, worker = self._post("/v1/rerank", RERANK_BODY, err=err)
        self.assertEqual(r.status_code, 503)
        worker.assert_not_called()

    def test_upstream_unreachable_is_502(self):
        with patch.object(llamaman, "_ensure_model_running", return_value=(_inst(), None)), \
             patch.object(llamaman, "request_local_worker", side_effect=ConnectionError("refused")):
            r = self.app.test_client().post("/v1/rerank", json=RERANK_BODY)
        self.assertEqual(r.status_code, 502)

    def test_gated_on_the_instance_port(self):
        for path in ("/v1/rerank", "/v1/reranking", "/rerank"):
            self.assertTrue(proxy._is_inference_request({"REQUEST_METHOD": "POST", "PATH_INFO": path}))


class EmbeddingsUnchangedTests(_Case):

    def test_still_refuses_non_embedding_instance(self):
        r, _, worker = self._post("/v1/embeddings", {"model": "m", "input": "hi"})
        self.assertEqual(r.status_code, 422)
        worker.assert_not_called()

    def test_still_forwards_to_v1_embeddings(self):
        data = {"object": "list", "data": [{"embedding": [0.1]}], "usage": {"prompt_tokens": 1}}
        r, _, worker = self._post("/v1/embeddings", {"model": "m", "input": "hi"},
                                  inst=_inst(embedding_model=True), upstream=_upstream(payload=data))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json(), data)
        self.assertEqual(worker.call_args.args[0], "http://llamaman-inst-1:8080/v1/embeddings")


if __name__ == "__main__":
    unittest.main()
