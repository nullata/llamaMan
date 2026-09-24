# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""core/kb.py business logic — backend-free (fake storage + fake embedder).

Guarantees under test:
  - the JSON backend is the unsupported implementation: kb_* raises
    KBNotSupportedError and kb_feature_supported says "json_backend";
  - the resilient wrapper turns primary failures into KBUnavailableError and
    never mirrors KB writes;
  - chunk_text is deterministic, bounded, and overlaps;
  - ingest dedups unchanged content via sha256 (no re-embed);
  - a dims mismatch between stored vectors and the current embedder refuses
    rather than mixing vector spaces;
  - the re-embed job is single-flight.
"""

import os
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

from core import kb
from storage.base import KBNotSupportedError, KBUnavailableError, StorageBackend
from storage.json_backend import JsonBackend


class FakeKBStorage:
    """Scriptable stand-in exposing only the kb_* surface core.kb uses."""

    def __init__(self, meta=None):
        self.meta = dict(meta or {})
        self.settings = {"kb_enabled": True, "kb_embedding_instance": "emb1"}
        self.upserts = []
        self.inserted = []
        self.searches = []
        self.topics = []

    # availability
    def kb_available(self): return True
    def kb_server_version(self): return (11, 8)

    # documents
    def kb_upsert_document(self, topic_id, title, content, source=""):
        import hashlib
        sha = hashlib.sha256(content.encode()).hexdigest()
        self.upserts.append((topic_id, title, sha))
        for i, (t, ti, s) in enumerate(self.upserts[:-1], 1):
            if (t, ti, s) == (topic_id, title, sha):
                return {"id": i, "unchanged": True}
        return {"id": len(self.upserts), "unchanged": False}

    def kb_get_document(self, document_id): return None
    def kb_list_documents(self, topic_id=None, visible_to=None): return []
    def kb_insert_chunks(self, document_id, chunks):
        self.inserted.append((document_id, chunks))
        return len(chunks)
    def kb_search(self, qv, topic_id=None, limit=8, visible_to=None):
        self.searches.append(qv)
        return [{"chunk_id": 1, "document_id": 7, "seq": 1, "text": "hit",
                 "title": "Doc", "topic": "T", "distance": 0.4}]
    def kb_meta_get(self): return dict(self.meta)
    def kb_meta_set(self, **kv): self.meta.update(kv)
    def kb_list_topics(self, visible_to=None):
        return [t for t in self.topics if visible_to is None
                or t.get("owner_key_id", "") in ("", visible_to)]
    def kb_clear_chunks(self): pass
    def get_settings(self): return dict(self.settings)


# --- JSON backend is the unsupported implementation -------------------------


class JsonBackendUnsupported(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.join(REPO_ROOT, "test-data", "kb-json")
        os.makedirs(self.tmp, exist_ok=True)
        self.backend = JsonBackend(
            os.path.join(self.tmp, "s.json"), os.path.join(self.tmp, "p.json"),
            os.path.join(self.tmp, "u.json"), os.path.join(self.tmp, "st.json"))

    def test_kb_methods_raise_not_supported(self):
        for call in (lambda: self.backend.ensure_kb_tables(),
                     lambda: self.backend.kb_list_topics(),
                     lambda: self.backend.kb_create_topic("x"),
                     lambda: self.backend.kb_search([0.1, 0.2]),
                     lambda: self.backend.kb_meta_get(),
                     lambda: self.backend.kb_counts()):
            with self.assertRaises(KBNotSupportedError):
                call()

    def test_kb_available_false_never_raises(self):
        self.assertFalse(self.backend.kb_available())
        self.assertIsNone(self.backend.kb_server_version())

    def test_feature_supported_reason(self):
        with patch("storage.get_storage", return_value=self.backend):
            ok, reason = kb.kb_feature_supported()
        self.assertFalse(ok)
        self.assertEqual(reason, "json_backend")


# --- resilient wrapper: never mirrored, typed errors -------------------------


class ResilientKbPassthrough(unittest.TestCase):
    def _wrapper(self, primary):
        mirror = os.path.join(REPO_ROOT, "test-data", "kb-resilient-mirror")
        os.makedirs(mirror, exist_ok=True)
        from storage.resilient import ResilientBackend
        return ResilientBackend(primary, mirror, node_id="test-node",
                                builder=lambda: primary, enabled=False)

    def test_healthy_passthrough(self):
        fake = FakeKBStorage()
        wrapper = self._wrapper(fake)
        self.assertEqual(wrapper.kb_meta_get(), {})
        wrapper.kb_meta_set(embedding_dims="4")
        self.assertEqual(fake.meta["embedding_dims"], "4")
        self.assertTrue(wrapper.kb_available())

    def test_offline_primary_becomes_kb_unavailable(self):
        wrapper = self._wrapper(None)
        with self.assertRaises(KBUnavailableError):
            wrapper.kb_list_topics()
        # availability probes never raise
        self.assertFalse(wrapper.kb_available())
        self.assertIsNone(wrapper.kb_server_version())


# --- chunker ------------------------------------------------------------------


class Chunking(unittest.TestCase):
    def test_short_text_single_chunk(self):
        self.assertEqual(kb.chunk_text("hello world"), ["hello world"])

    def test_empty(self):
        self.assertEqual(kb.chunk_text(""), [])
        self.assertEqual(kb.chunk_text("  \n\n  "), [])

    def test_determinism(self):
        text = "\n\n".join(f"Paragraph {i} talks about topic {i % 3}. " * 8
                           for i in range(40))
        self.assertEqual(kb.chunk_text(text), kb.chunk_text(text))

    def test_bounds_and_overlap(self):
        para = ("lorem ipsum dolor sit amet " * 30).strip()  # long paragraphs
        text = "\n\n".join(f"{para} section {i}" for i in range(10))
        chunks = kb.chunk_text(text)
        self.assertGreater(len(chunks), 2)
        for c in chunks:
            self.assertLessEqual(len(c), kb.CHUNK_TARGET_CHARS + 64)
        # consecutive chunks share an overlap tail
        overlapped = 0
        for a, b in zip(chunks, chunks[1:]):
            tail_word = a[-40:].split()[-1]
            if tail_word and tail_word in b:
                overlapped += 1
        self.assertGreaterEqual(overlapped, len(chunks) - 1)

    def test_oversized_single_paragraph_hard_split(self):
        text = "x" * 50  # word-free giant block
        chunks = kb.chunk_text("word " * 1 + text)
        self.assertTrue(all(len(c) <= kb.CHUNK_TARGET_CHARS + 64 for c in chunks))


# --- availability gating -------------------------------------------------------


class Gating(unittest.TestCase):
    def test_disabled_master_switch(self):
        fake = FakeKBStorage()
        fake.settings["kb_enabled"] = False
        with patch("storage.get_storage", return_value=fake):
            with self.assertRaises(KBUnavailableError):
                kb.assert_kb_ready()

    def test_search_empty_query_rejected(self):
        fake = FakeKBStorage(meta={"embedding_dims": "4"})
        with patch("storage.get_storage", return_value=fake), \
             patch.object(kb, "embed_texts", return_value=[[0.0] * 4]):
            with self.assertRaises(KBUnavailableError):
                kb.search("   ")

    def test_search_empty_kb_rejected(self):
        fake = FakeKBStorage(meta={})
        with patch("storage.get_storage", return_value=fake):
            with self.assertRaises(KBUnavailableError) as ctx:
                kb.search("anything")
        self.assertIn("empty", str(ctx.exception))

    def test_embedding_dims_mismatch_refuses_search(self):
        fake = FakeKBStorage(meta={"embedding_dims": "4"})
        with patch("storage.get_storage", return_value=fake), \
             patch.object(kb, "embed_texts", return_value=[[0.0] * 8]):
            with self.assertRaises(KBUnavailableError) as ctx:
                kb.search("anything")
        self.assertIn("mismatch", str(ctx.exception))

    def test_mcp_ingest_size_cap(self):
        fake = FakeKBStorage(meta={"embedding_dims": "4"})
        big = "x" * (kb.MCP_INGEST_MAX_BYTES + 1)
        with patch("storage.get_storage", return_value=fake), \
             patch.object(kb, "embed_texts", return_value=[[0.0] * 4]):
            with self.assertRaises(KBUnavailableError) as ctx:
                kb.ingest_document(1, "T", big, max_bytes=kb.MCP_INGEST_MAX_BYTES)
        self.assertIn("too large", str(ctx.exception))


# --- ingest dedup + score clamp ------------------------------------------------


class IngestSearch(unittest.TestCase):
    def test_unchanged_content_skips_embedding(self):
        fake = FakeKBStorage(meta={"embedding_dims": "4"})
        embed_calls = []

        def fake_embed(texts):
            embed_calls.append(len(texts))
            return [[0.0] * 4 for _ in texts]

        with patch("storage.get_storage", return_value=fake), \
             patch.object(kb, "embed_texts", fake_embed):
            r1 = kb.ingest_document(1, "Doc", "some content")
            self.assertFalse(r1["unchanged"])
            self.assertGreaterEqual(r1["chunks"], 1)
            r2 = kb.ingest_document(1, "Doc", "some content")
            self.assertTrue(r2["unchanged"])
            self.assertEqual(r2["chunks"], 0)
        self.assertEqual(embed_calls, [1])  # embedded exactly once

    def test_first_ingest_locks_dims_and_model(self):
        fake = FakeKBStorage(meta={})
        with patch("storage.get_storage", return_value=fake), \
             patch.object(kb, "embed_texts", return_value=[[0.1] * 4]), \
             patch.object(kb, "_current_model_label", return_value="bge-m3"):
            kb.ingest_document(1, "Doc", "content")
        self.assertEqual(fake.meta["embedding_dims"], "4")
        self.assertEqual(fake.meta["embedding_model"], "bge-m3")

    def test_model_label_prefers_model_name_setting(self):
        # Model-name selection is the cluster-wide mode; the recorded label
        # must come from it, not from the legacy per-node instance UUID.
        fake = FakeKBStorage()
        fake.settings["kb_embedding_model"] = "Qwen3-Embedding-0.6B"
        with patch("storage.get_storage", return_value=fake):
            self.assertEqual(kb._current_model_label(), "Qwen3-Embedding-0.6B")

    def test_model_label_falls_back_to_legacy_instance(self):
        fake = FakeKBStorage()  # kb_embedding_instance="emb1", no model name
        with patch("storage.get_storage", return_value=fake), \
             patch.dict("core.state.instances",
                        {"emb1": {"model_name": "bge-m3"}}, clear=True):
            self.assertEqual(kb._current_model_label(), "bge-m3")

    def test_ingest_recovers_from_empty_string_meta(self):
        # Re-embed of an empty KB used to write ('', '') to kb_meta and then
        # the next ingest crashed on int('') inside _lock_or_check_model_meta.
        fake = FakeKBStorage(meta={"embedding_dims": "", "embedding_model": ""})
        with patch("storage.get_storage", return_value=fake), \
             patch.object(kb, "embed_texts", return_value=[[0.1] * 4]), \
             patch.object(kb, "_current_model_label", return_value="bge-m3"):
            kb.ingest_document(1, "Doc", "content")
        self.assertEqual(fake.meta["embedding_dims"], "4")
        self.assertEqual(fake.meta["embedding_model"], "bge-m3")

    def test_reembed_empty_kb_does_not_poison_meta(self):
        # Hitting Re-embed on a KB with zero documents used to unconditionally
        # blank the dims/model lock, bricking the next ingest. Verify meta is
        # left alone when there are no docs to process.
        fake = FakeKBStorage(meta={"embedding_dims": "1024",
                                    "embedding_model": "Qwen3"})
        fake.kb_list_documents = lambda topic_id=None: []
        with patch("storage.get_storage", return_value=fake):
            kb._reembed_worker()
        self.assertEqual(fake.meta["embedding_dims"], "1024")
        self.assertEqual(fake.meta["embedding_model"], "Qwen3")

    def test_score_clamped_to_unit_interval(self):
        fake = FakeKBStorage(meta={"embedding_dims": "4"})
        fake.kb_search = lambda qv, topic_id=None, limit=8, visible_to=None: [
            {"chunk_id": 1, "document_id": 1, "seq": 1, "text": "a",
             "title": "t", "topic": "T", "distance": 2.6}]  # beyond 2 -> clamp
        with patch("storage.get_storage", return_value=fake), \
             patch.object(kb, "embed_texts", return_value=[[0.0] * 4]):
            out = kb.search("q")
        self.assertEqual(out[0]["score"], 0.0)

    def test_topic_name_resolution(self):
        fake = FakeKBStorage(meta={"embedding_dims": "4"})
        fake.topics = [{"id": 3, "name": "Recipes"}]
        with patch("storage.get_storage", return_value=fake), \
             patch.object(kb, "embed_texts", return_value=[[0.0] * 4]):
            out = kb.search("cake", topic_name="recipes", limit=5)
        self.assertEqual(len(out), 1)


# --- re-embed single flight ------------------------------------------------------


class ReembedSingleFlight(unittest.TestCase):
    def test_second_start_rejected(self):
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def fake_worker():
            started.set()
            release.wait(5.0)
            # Mirror the worker's teardown: this IS what the real one does in
            # its finally block; the patch replaces the whole function, so the
            # test owns the state transition deterministically.
            with kb._reembed_lock:
                kb._reembed_job["running"] = False
            finished.set()

        fake = FakeKBStorage(meta={"embedding_dims": "4"})
        with patch("storage.get_storage", return_value=fake):
            with patch.object(kb, "_reembed_worker", fake_worker):
                kb.reembed_all()
                self.assertTrue(started.wait(2.0))
                with self.assertRaises(RuntimeError):
                    kb.reembed_all()
                release.set()
                self.assertTrue(finished.wait(5.0))
        self.assertFalse(kb.reembed_status()["running"])


class EmbedResolver(unittest.TestCase):
    """_resolve_embed_target: local first, peer fallback (model-name mode
    only), legacy per-node UUID still works. No probing of running instances
    via HTTP — the resolver reads state.instances + list_nodes only."""

    def _with_state(self, local_instances, nodes, settings):
        """Patch core.state.instances + storage list_nodes + get_settings."""
        from unittest.mock import MagicMock
        st = MagicMock()
        st.get_settings.return_value = settings
        st.list_nodes.return_value = nodes
        return (patch.dict("core.state.instances",
                           {i["id"]: i for i in local_instances}, clear=True),
                patch("storage.get_storage", return_value=st))

    def test_local_match_by_model_name(self):
        local = [{"id": "a", "model_name": "bge-m3", "status": "healthy",
                  "config": {"embedding_model": True},
                  "_server_host": "h", "_server_port": 9001, "port": 8001}]
        ctx1, ctx2 = self._with_state(
            local, [], {"kb_embedding_model": "bge-m3"})
        with ctx1, ctx2:
            target = kb._resolve_embed_target()
        self.assertEqual(target[0], "local")
        self.assertEqual(target[2], "h")
        self.assertEqual(target[3], 9001)

    def test_local_match_by_legacy_uuid(self):
        local = [{"id": "u-1", "model_name": "bge-m3", "status": "healthy",
                  "config": {"embedding_model": True},
                  "_server_host": "h", "_server_port": 9001, "port": 8001}]
        ctx1, ctx2 = self._with_state(
            local, [], {"kb_embedding_instance": "u-1"})
        with ctx1, ctx2:
            target = kb._resolve_embed_target()
        self.assertEqual(target[0], "local")

    def _opted_in_peer(self, model="bge-m3"):
        return {
            "node_id": "peer1", "advertise_url": "http://peer:5000",
            "snapshot": {
                "kb": {"enabled": True, "mcp": True},
                "instances": [
                    {"id": "p1", "model_name": model, "status": "healthy",
                     "config": {"embedding_model": True}}]}}

    def test_peer_match_when_no_local(self):
        ctx1, ctx2 = self._with_state(
            [], [self._opted_in_peer()], {"kb_embedding_model": "bge-m3"})
        with ctx1, ctx2, \
             patch("core.cluster.get_node_id", return_value="self"):
            target = kb._resolve_embed_target()
        self.assertEqual(target[0], "peer")
        self.assertEqual(target[1]["node_id"], "peer1")

    def test_sleeping_peer_forwarded_not_relaunched(self):
        # Model only asleep on a peer: forward there (the peer wakes it)
        # instead of sticky-relaunching a duplicate instance.
        peer = self._opted_in_peer()
        peer["snapshot"]["instances"][0]["status"] = "sleeping"
        ctx1, ctx2 = self._with_state(
            [], [peer], {"kb_embedding_model": "bge-m3"})
        with ctx1, ctx2, \
             patch("core.cluster.get_node_id", return_value="self"), \
             patch.object(kb, "_launch_on_peer") as launch:
            target = kb._resolve_embed_target()
        self.assertEqual(target[0], "peer")
        self.assertTrue(target[1].get("_kb_waking"))
        launch.assert_not_called()

    def _starting_local(self):
        return [{"id": "a", "model_name": "bge-m3", "status": "starting",
                 "config": {"embedding_model": True},
                 "_server_host": "h", "_server_port": 9001, "port": 8001}]

    def test_local_starting_waits_for_healthy(self):
        # Another request is already loading the model: wait, don't fail.
        ctx1, ctx2 = self._with_state(
            self._starting_local(), [], {"kb_embedding_model": "bge-m3"})
        with ctx1, ctx2, \
             patch("api.instances.wait_for_healthy", return_value=True) as w:
            target = kb._resolve_embed_target()
        self.assertEqual(target[0], "local")
        self.assertEqual((target[2], target[3]), ("h", 9001))
        self.assertEqual(w.call_args[0][:2], ("h", 9001))

    def test_local_starting_that_never_loads_falls_through(self):
        ctx1, ctx2 = self._with_state(
            self._starting_local(), [], {"kb_embedding_model": "bge-m3"})
        with ctx1, ctx2, \
             patch("api.instances.wait_for_healthy", return_value=False), \
             patch("core.cluster.get_node_id", return_value="self"):
            with self.assertRaises(KBUnavailableError):
                kb._resolve_embed_target()

    def test_starting_peer_forwarded_not_relaunched(self):
        peer = self._opted_in_peer()
        peer["snapshot"]["instances"][0]["status"] = "starting"
        ctx1, ctx2 = self._with_state(
            [], [peer], {"kb_embedding_model": "bge-m3"})
        with ctx1, ctx2, \
             patch("core.cluster.get_node_id", return_value="self"), \
             patch.object(kb, "_launch_on_peer") as launch:
            target = kb._resolve_embed_target()
        self.assertEqual(target[0], "peer")
        self.assertTrue(target[1].get("_kb_waking"))
        launch.assert_not_called()

    def test_healthy_peer_preferred_over_sleeping_peer(self):
        sleeping = self._opted_in_peer()
        sleeping["node_id"] = "sleepy"
        sleeping["snapshot"]["instances"][0]["status"] = "sleeping"
        healthy = self._opted_in_peer()
        ctx1, ctx2 = self._with_state(
            [], [sleeping, healthy], {"kb_embedding_model": "bge-m3"})
        with ctx1, ctx2, \
             patch("core.cluster.get_node_id", return_value="self"):
            target = kb._resolve_embed_target()
        self.assertEqual(target[1]["node_id"], "peer1")
        self.assertFalse(target[1].get("_kb_waking"))

    def test_peer_without_opt_in_skipped(self):
        # kb.enabled/mcp both false in the snapshot's kb block — this peer
        # is not in the group and must NOT be picked, even though it has a
        # matching instance running.
        peer = self._opted_in_peer()
        peer["snapshot"]["kb"] = {"enabled": False, "mcp": False}
        ctx1, ctx2 = self._with_state(
            [], [peer], {"kb_embedding_model": "bge-m3"})
        with ctx1, ctx2, \
             patch("core.cluster.get_node_id", return_value="self"):
            with self.assertRaises(KBUnavailableError):
                kb._resolve_embed_target()

    def test_no_match_raises(self):
        ctx1, ctx2 = self._with_state(
            [], [], {"kb_embedding_model": "bge-m3"})
        with ctx1, ctx2:
            with self.assertRaises(KBUnavailableError) as e:
                kb._resolve_embed_target()
        self.assertIn("no live embedding instance", str(e.exception))

    def test_no_setting_raises(self):
        ctx1, ctx2 = self._with_state([], [], {})
        with ctx1, ctx2:
            with self.assertRaises(KBUnavailableError) as e:
                kb._resolve_embed_target()
        self.assertIn("no embedding model", str(e.exception))

    def test_legacy_uuid_never_resolves_peer(self):
        # Peer has the instance but the setting is a legacy UUID — must NOT
        # cross to a peer (UUIDs are per-node ephemeral).
        peer_node = {
            "node_id": "peer1", "advertise_url": "http://peer:5000",
            "snapshot": {"instances": [
                {"id": "u-legacy", "model_name": "bge-m3", "status": "healthy",
                 "config": {"embedding_model": True}}]}}
        ctx1, ctx2 = self._with_state(
            [], [peer_node], {"kb_embedding_instance": "u-legacy"})
        with ctx1, ctx2:
            with self.assertRaises(KBUnavailableError):
                kb._resolve_embed_target()


class LocalWake(unittest.TestCase):
    """A sleeping local instance is woken in-process before an embed call.
    The wake call is `api.instances.relaunch_inactive_instance` (same helper
    the crash auto-restart uses), which blocks until healthy."""

    def test_sleeping_local_is_woken(self):
        from core.state import instances as real_instances
        sleeping = {"id": "a", "model_name": "bge-m3", "status": "sleeping",
                    "config": {"embedding_model": True},
                    "_server_host": "h", "_server_port": 9001, "port": 8001}
        after_wake = dict(sleeping); after_wake["status"] = "healthy"
        st = MagicMock()
        st.get_settings.return_value = {"kb_embedding_model": "bge-m3"}
        st.list_nodes.return_value = []

        def wake(inst_id):
            # relaunch_inactive_instance mutates the real state.instances
            # dict — mirror that here so the resolver's re-pick sees healthy.
            real_instances[inst_id] = after_wake
            return True

        with patch.dict("core.state.instances", {"a": sleeping}, clear=True), \
             patch("storage.get_storage", return_value=st), \
             patch("api.instances.relaunch_inactive_instance",
                   side_effect=wake) as wake_call:
            target = kb._resolve_embed_target()
        self.assertEqual(target[0], "local")
        self.assertEqual(wake_call.call_count, 1)


class StickyStamp(unittest.TestCase):
    """Every successful _embed_local stamps kb_meta sticky_embedding_* so a
    future cluster-wide miss can relaunch here."""

    def test_stamped_on_success(self):
        r_ok = MagicMock()
        r_ok.ok = True; r_ok.status_code = 200
        r_ok.raise_for_status = lambda: None
        r_ok.json = lambda: {"data": [{"index": 0, "embedding": [0.1] * 4}]}
        st = MagicMock()
        with patch("requests.post", return_value=r_ok), \
             patch("core.request_log.record_request", return_value=MagicMock()), \
             patch("core.request_log.finalize_async"), \
             patch("core.cluster.get_node_id", return_value="node-a"), \
             patch("storage.get_storage", return_value=st):
            kb._embed_local({"id": "i", "model_name": "bge-m3",
                              "model_path": "/m/bge.gguf",
                              "config": {"embedding_model": True}},
                             "h", 9001, ["hi"])
        # kb_meta_set called with both sticky keys
        calls = [c for c in st.kb_meta_set.call_args_list
                 if "sticky_embedding_node" in c.kwargs]
        self.assertTrue(calls, "sticky was not stamped")
        self.assertEqual(calls[-1].kwargs["sticky_embedding_node"], "node-a")
        self.assertIn("bge.gguf",
                      calls[-1].kwargs["sticky_embedding_config"])


class Autolaunch(unittest.TestCase):
    """Cluster-wide miss with a valid sticky pointer triggers a cross-node
    launch on the sticky node (if it's opted in), polls until healthy, then
    returns ('peer', node)."""

    def setUp(self):
        # Reset the module-level rate-limit log between tests so budget
        # exhaustion in one doesn't cascade to another.
        kb._autolaunch_log.clear()

    def _sticky_node(self):
        return {
            "node_id": "peer1", "advertise_url": "http://peer:5000",
            "snapshot": {"kb": {"enabled": True, "mcp": True},
                          "instances": []}}

    def _settings(self, model="bge-m3"):
        return {"kb_embedding_model": model}

    def _sticky(self):
        return ("peer1", {"model_path": "/models/bge.gguf",
                          "config": {"embedding_model": True}})

    def test_autolaunch_succeeds_and_returns_peer(self):
        node = self._sticky_node()
        st = MagicMock()
        st.get_settings.return_value = self._settings()
        st.list_nodes.return_value = [node]

        launched = {"posted": False, "polls": 0}
        r_post = MagicMock(); r_post.status_code = 202
        r_post.json = lambda: {"id": "new-instance-id"}
        r_healthy = MagicMock(); r_healthy.status_code = 200
        r_healthy.json = lambda: {"status": "healthy"}

        def fake_cluster_request(node, method, path, **kw):
            if method == "POST" and path == "/api/instances":
                launched["posted"] = True
                return r_post
            if method == "GET" and path.startswith("/api/instances/"):
                launched["polls"] += 1
                return r_healthy
            raise AssertionError(f"unexpected {method} {path}")

        with patch.dict("core.state.instances", {}, clear=True), \
             patch("storage.get_storage", return_value=st), \
             patch("core.cluster.get_node_id", return_value="self"), \
             patch.object(kb, "_read_sticky", return_value=self._sticky()), \
             patch("core.cluster.cluster_request",
                   side_effect=fake_cluster_request), \
             patch("time.sleep"):
            target = kb._resolve_embed_target()
        self.assertEqual(target[0], "peer")
        self.assertTrue(launched["posted"])
        self.assertGreaterEqual(launched["polls"], 1)

    def test_autolaunch_timeout_raises_and_burns_budget(self):
        node = self._sticky_node()
        st = MagicMock()
        st.get_settings.return_value = self._settings()
        st.list_nodes.return_value = [node]

        r_post = MagicMock(); r_post.status_code = 202
        r_post.json = lambda: {"id": "x"}
        r_starting = MagicMock(); r_starting.status_code = 200
        r_starting.json = lambda: {"status": "starting"}

        # Advance time steadily inside the polling loop so the deadline
        # burns out within a few iterations, without wall-clock waiting.
        clock = [1000.0]
        def _time():
            return clock[0]
        def _sleep(_):
            clock[0] += 100  # each sleep advances 100s

        def cluster_request(node, method, path, **kw):
            return r_post if method == "POST" else r_starting

        with patch.dict("core.state.instances", {}, clear=True), \
             patch("storage.get_storage", return_value=st), \
             patch("core.cluster.get_node_id", return_value="self"), \
             patch.object(kb, "_read_sticky", return_value=self._sticky()), \
             patch("core.cluster.cluster_request",
                   side_effect=cluster_request), \
             patch("core.kb.time.time", side_effect=_time), \
             patch("core.kb.time.sleep", side_effect=_sleep):
            with self.assertRaises(KBUnavailableError):
                kb._resolve_embed_target()
        self.assertIn("bge-m3", kb._autolaunch_log)
        self.assertEqual(len(kb._autolaunch_log["bge-m3"]), 1)

    def test_autolaunch_skipped_when_sticky_node_not_opted_in(self):
        node = self._sticky_node()
        node["snapshot"]["kb"] = {"enabled": False, "mcp": False}
        st = MagicMock()
        st.get_settings.return_value = self._settings()
        st.list_nodes.return_value = [node]
        with patch.dict("core.state.instances", {}, clear=True), \
             patch("storage.get_storage", return_value=st), \
             patch("core.cluster.get_node_id", return_value="self"), \
             patch.object(kb, "_read_sticky", return_value=self._sticky()), \
             patch("core.cluster.cluster_request") as cr:
            with self.assertRaises(KBUnavailableError):
                kb._resolve_embed_target()
        cr.assert_not_called()

    def test_autolaunch_budget_burns_out_then_clears_sticky(self):
        node = self._sticky_node()
        st = MagicMock()
        st.get_settings.return_value = self._settings()
        st.list_nodes.return_value = [node]

        r_bad = MagicMock(); r_bad.status_code = 500; r_bad.text = "fail"
        def cluster_request(node, method, path, **kw):
            return r_bad

        with patch.dict("core.state.instances", {}, clear=True), \
             patch("storage.get_storage", return_value=st), \
             patch("core.cluster.get_node_id", return_value="self"), \
             patch.object(kb, "_read_sticky", return_value=self._sticky()), \
             patch("core.cluster.cluster_request",
                   side_effect=cluster_request), \
             patch.object(kb, "_clear_sticky") as clear:
            # Three failures reach the budget cap. On the LAST one, sticky
            # is cleared so the next miss doesn't chase a phantom target.
            for _ in range(kb._AUTOLAUNCH_MAX):
                with self.assertRaises(KBUnavailableError):
                    kb._resolve_embed_target()
        self.assertTrue(clear.called)


class PeerForward(unittest.TestCase):
    """embed_texts hands off to a peer over cluster_request when the
    resolver returns ('peer', node)."""

    def test_peer_forward_returns_vectors(self):
        peer = {"node_id": "peer1", "advertise_url": "http://peer:5000"}
        r = MagicMock()
        r.status_code = 200
        r.json = lambda: {"vectors": [[0.1] * 4, [0.2] * 4]}
        with patch.object(kb, "_resolve_embed_target",
                          return_value=("peer", peer)), \
             patch("core.cluster.cluster_request",
                   return_value=r) as fwd:
            out = kb.embed_texts(["a", "b"])
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0], [0.1] * 4)
        # POST /api/kb/_peer_embed with the texts payload.
        _, kw = fwd.call_args[0], fwd.call_args[1]
        self.assertEqual(fwd.call_args[0][1], "POST")
        self.assertEqual(fwd.call_args[0][2], "/api/kb/_peer_embed")
        self.assertEqual(kw["json"], {"texts": ["a", "b"]})

    def test_peer_forward_to_waking_peer_allows_model_load(self):
        from config import MODEL_LOAD_TIMEOUT
        peer = {"node_id": "peer1", "advertise_url": "http://peer:5000",
                "_kb_waking": True}
        r = MagicMock()
        r.status_code = 200
        r.json = lambda: {"vectors": [[0.1] * 4]}
        with patch.object(kb, "_resolve_embed_target",
                          return_value=("peer", peer)), \
             patch("core.cluster.cluster_request", return_value=r) as fwd:
            kb.embed_texts(["a"])
        _, read = fwd.call_args[1]["timeout"]
        self.assertGreaterEqual(read, MODEL_LOAD_TIMEOUT)

    def test_peer_forward_non_200_raises(self):
        peer = {"node_id": "peer1", "advertise_url": "http://peer:5000"}
        r = MagicMock()
        r.status_code = 503
        with patch.object(kb, "_resolve_embed_target",
                          return_value=("peer", peer)), \
             patch("core.cluster.cluster_request", return_value=r):
            with self.assertRaises(KBUnavailableError):
                kb.embed_texts(["a"])

    def test_peer_forward_wrong_vector_count_raises(self):
        peer = {"node_id": "peer1", "advertise_url": "http://peer:5000"}
        r = MagicMock()
        r.status_code = 200
        r.json = lambda: {"vectors": [[0.1] * 4]}  # 1 vector for 2 inputs
        with patch.object(kb, "_resolve_embed_target",
                          return_value=("peer", peer)), \
             patch("core.cluster.cluster_request", return_value=r):
            with self.assertRaises(KBUnavailableError):
                kb.embed_texts(["a", "b"])


class EmbedRequestLog(unittest.TestCase):
    """Every embedding POST is booked in request_log so embedding traffic
    shows up alongside inference in the same stream. Bypasses the per-
    instance proxy (direct llama-server call), so record_request is called
    explicitly by the KB layer."""

    def _mock_ok_response(self, payload):
        r = MagicMock()
        r.ok = True
        r.status_code = 200
        r.raise_for_status = lambda: None
        r.json = lambda: payload
        return r

    def test_openai_shape_records_each_batch(self):
        r_ok = self._mock_ok_response({
            "data": [{"index": 0, "embedding": [0.1] * 4}]})
        with patch("requests.post", return_value=r_ok) as post, \
             patch("core.request_log.record_request",
                   return_value=MagicMock()) as rec, \
             patch("core.request_log.finalize_async") as fin:
            kb._embed_openai_shape("http://host:9000", ["hello"], (5, 30),
                                    "inst-42", "bge-m3")
        self.assertEqual(post.call_count, 1)
        self.assertEqual(rec.call_count, 1)
        args, kwargs = rec.call_args
        self.assertEqual(kwargs["endpoint"], "kb-embed")
        self.assertEqual(kwargs["path"], "/v1/embeddings")
        self.assertEqual(kwargs["inst_id"], "inst-42")
        self.assertEqual(kwargs["model"], "bge-m3")
        self.assertEqual(fin.call_count, 1)

    def test_llama_native_records_each_call(self):
        r_ok = self._mock_ok_response({"embedding": [0.1] * 4})
        with patch("requests.post", return_value=r_ok) as post, \
             patch("core.request_log.record_request",
                   return_value=MagicMock()) as rec, \
             patch("core.request_log.finalize_async") as fin:
            kb._embed_llama_native("http://host:9000", ["a", "b"], (5, 30),
                                    "inst-42", "bge-m3")
        self.assertEqual(post.call_count, 2)
        self.assertEqual(rec.call_count, 2)
        self.assertEqual(fin.call_count, 2)
        self.assertEqual(rec.call_args_list[0][1]["path"], "/embedding")


class UpsertDedupRequiresChunks(unittest.TestCase):
    """MariaDBBackend.kb_upsert_document's real SQL, run against SQLite: the
    sha is stored before embedding, so identical content only counts as
    unchanged once the document actually has chunks."""

    def setUp(self):
        from sqlalchemy import create_engine, text
        from sqlalchemy.pool import StaticPool
        from storage.mariadb_backend import MariaDBBackend
        self.text = text
        self.b = MariaDBBackend.__new__(MariaDBBackend)
        self.b._engine = create_engine("sqlite://", poolclass=StaticPool,
                                       connect_args={"check_same_thread": False})
        self.b.ensure_kb_tables = lambda: None
        with self.b._engine.begin() as c:
            c.execute(text("CREATE TABLE kb_documents (id INTEGER PRIMARY KEY "
                           "AUTOINCREMENT, topic_id INT, title TEXT, content TEXT, "
                           "sha256 TEXT, source TEXT)"))
            c.execute(text("CREATE TABLE kb_chunks (id INTEGER PRIMARY KEY "
                           "AUTOINCREMENT, document_id INT)"))

    def _add_chunk(self, doc_id):
        with self.b._engine.begin() as c:
            c.execute(self.text("INSERT INTO kb_chunks (document_id) VALUES (:d)"),
                      {"d": doc_id})

    def test_resubmit_after_failed_embed_is_not_unchanged(self):
        first = self.b.kb_upsert_document(1, "Doc", "body")
        self.assertFalse(first["unchanged"])
        # Embed failed: no chunks written. The retry must embed again.
        retry = self.b.kb_upsert_document(1, "Doc", "body")
        self.assertFalse(retry["unchanged"])
        self.assertEqual(retry["id"], first["id"])

    def test_resubmit_after_successful_embed_is_unchanged(self):
        first = self.b.kb_upsert_document(1, "Doc", "body")
        self._add_chunk(first["id"])
        again = self.b.kb_upsert_document(1, "Doc", "body")
        self.assertTrue(again["unchanged"])


if __name__ == "__main__":
    unittest.main()
