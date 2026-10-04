# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Streamed responses must be relayed as they arrive.

Strata answers SSE over HTTP/1.0 with neither chunked framing nor a length
(the stream ends when the connection closes). requests'
iter_content(chunk_size=None) reads such a body to EOF before yielding, so
llamaman delivered a whole Strata answer at once. core.helpers.
iter_response_chunks / iter_response_lines read it incrementally, and leave
chunked (llama-server) and length-delimited responses on the old code path.
"""

import os
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

from core.helpers import iter_response_chunks, iter_response_lines


def _utf8_lines(resp):
    resp.encoding = "utf-8"   # as api/llamaman.py does before reading lines
    return iter_response_lines(resp)

GATE = threading.Event()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        # First event, then wait until the client has seen it (or 5s): a
        # buffering reader only gets data after that timeout.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        chunked = self.path == "/chunked"
        if chunked:
            self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send(b):
            if chunked:
                self.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n")
            else:
                self.wfile.write(b)
            self.wfile.flush()
        send("data: {\"t\": \"hé\"}\n\n".encode())
        GATE.wait(5)
        send(b"data: [DONE]\n\n")
        if chunked:
            self.wfile.write(b"0\r\n\r\n")


class _Http10(_Handler):
    protocol_version = "HTTP/1.0"     # like Strata (serve/server.py)


class _Http11(_Handler):
    protocol_version = "HTTP/1.1"     # chunked, like llama-server


def _serve(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class StreamRelayTests(unittest.TestCase):

    def setUp(self):
        GATE.clear()

    def _first_item_delay(self, url, reader):
        t = time.monotonic()
        resp = requests.get(url, stream=True)
        it = reader(resp)
        first = next(it)
        delay = time.monotonic() - t
        GATE.set()
        rest = list(it)
        return delay, first, rest

    def test_eof_delimited_stream_is_not_buffered(self):
        srv = _serve(_Http10)
        try:
            delay, first, rest = self._first_item_delay(
                f"http://127.0.0.1:{srv.server_port}/", iter_response_chunks)
        finally:
            srv.shutdown()
        self.assertLess(delay, 2.0)
        self.assertIn(b"h\xc3\xa9", first)
        self.assertEqual(b"".join(rest), b"data: [DONE]\n\n")

    def test_eof_delimited_lines_are_not_buffered(self):
        srv = _serve(_Http10)
        try:
            delay, first, rest = self._first_item_delay(
                f"http://127.0.0.1:{srv.server_port}/", _utf8_lines)
        finally:
            srv.shutdown()
        self.assertLess(delay, 2.0)
        self.assertEqual(first, 'data: {"t": "hé"}')
        self.assertEqual([x for x in rest if x], ["data: [DONE]"])

    def test_chunked_keeps_requests_code_path(self):
        srv = _serve(_Http11)
        try:
            url = f"http://127.0.0.1:{srv.server_port}/chunked"
            with patch.object(requests.Response, "iter_content", autospec=True,
                              side_effect=requests.Response.iter_content) as ic:
                GATE.set()
                chunks = list(iter_response_chunks(requests.get(url, stream=True)))
            self.assertTrue(ic.called)
            self.assertEqual(b"".join(chunks), "data: {\"t\": \"hé\"}\n\ndata: [DONE]\n\n".encode())
            with patch.object(requests.Response, "iter_lines", autospec=True,
                              side_effect=requests.Response.iter_lines) as il:
                lines = list(_utf8_lines(requests.get(url, stream=True)))
            self.assertTrue(il.called)
            self.assertEqual([x for x in lines if x], ['data: {"t": "hé"}', "data: [DONE]"])
        finally:
            srv.shutdown()


if __name__ == "__main__":
    unittest.main()
