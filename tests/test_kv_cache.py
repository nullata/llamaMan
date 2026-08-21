# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Tests for Flash Attention + KV cache quantization launch controls:
--flash-attn / --cache-type-k / --cache-type-v flag emission, preset
persistence, launch route forwarding, and the preset-merge whitelist that
lets a live preset edit reach a running instance without a relaunch.

--flash-attn is llama.cpp's tri-state ('on'|'off'|'auto', default 'auto');
these tests also pin the legacy-bool coercion path so a config saved before
the tri-state rollout keeps producing an equivalent command line."""

import os
import unittest
from unittest.mock import Mock, patch

from flask import Flask

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))
os.environ.setdefault("LLAMAMAN_NODE_NAME", "test-node")

import api.instances as instances_api
import api.presets as presets_api
from core.helpers import build_llama_cmd, normalize_flash_attn
from core.state import instances, instances_lock


class NormalizeFlashAttnTests(unittest.TestCase):
    """The helper is the single source of truth for flash_attn coercion -
    every boundary (route, launch_instance, preset save, build_llama_cmd)
    routes through it. Pinning it here catches drift that would otherwise
    only surface as a wrong CLI arg buried in a launch."""

    def test_new_string_values_pass_through(self):
        self.assertEqual(normalize_flash_attn("on"), "on")
        self.assertEqual(normalize_flash_attn("off"), "off")
        self.assertEqual(normalize_flash_attn("auto"), "auto")

    def test_case_and_whitespace_tolerant(self):
        self.assertEqual(normalize_flash_attn(" ON "), "on")
        self.assertEqual(normalize_flash_attn("Auto"), "auto")

    def test_legacy_bool_true_maps_to_on(self):
        # Configs and presets from before the tri-state rollout stored a bool.
        # True has to become 'on' so the emitted CLI matches the old behavior
        # (which passed --flash-attn to force it on).
        self.assertEqual(normalize_flash_attn(True), "on")

    def test_legacy_bool_false_maps_to_off(self):
        # False becomes 'off' rather than 'auto' because a user who explicitly
        # unchecked the old toggle meant "don't use flash-attn" - preserving
        # that intent over llama.cpp's new 'auto' default matters on backends
        # where auto might resolve to on.
        self.assertEqual(normalize_flash_attn(False), "off")

    def test_empty_and_unknown_default_to_auto(self):
        self.assertEqual(normalize_flash_attn(None), "auto")
        self.assertEqual(normalize_flash_attn(""), "auto")
        self.assertEqual(normalize_flash_attn("garbage"), "auto")
        self.assertEqual(normalize_flash_attn(42), "auto")


class BuildLlamaCmdKvCacheTests(unittest.TestCase):
    """`build_llama_cmd` emits --flash-attn / --cache-type-k / --cache-type-v
    only when the user picked non-default values, so a config from before this
    feature (or one where the user left the defaults) produces the exact same
    command line as it did previously.

    For --flash-attn specifically: 'auto' (and any missing/empty value) omits
    the flag entirely, matching llama.cpp's own default so the command line
    stays quiet for the common case. 'on'/'off' emit the value explicitly -
    a bare --flash-attn would now consume the next arg as its value."""

    def test_flash_attn_missing_omits_flag(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {})
        self.assertNotIn("--flash-attn", cmd)

    def test_flash_attn_auto_omits_flag(self):
        # 'auto' is llama.cpp's own default when the flag is absent, so
        # emitting it would just be noise in the command line.
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"flash_attn": "auto"})
        self.assertNotIn("--flash-attn", cmd)

    def test_flash_attn_on_emits_on_value(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"flash_attn": "on"})
        self.assertEqual(cmd[cmd.index("--flash-attn") + 1], "on")

    def test_flash_attn_off_emits_off_value(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"flash_attn": "off"})
        self.assertEqual(cmd[cmd.index("--flash-attn") + 1], "off")

    def test_flash_attn_legacy_bool_true_emits_on(self):
        # A config or preset from before the tri-state rollout stored bool;
        # normalize_flash_attn folds it into the string form so the emitted
        # CLI matches what the old code produced.
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"flash_attn": True})
        self.assertEqual(cmd[cmd.index("--flash-attn") + 1], "on")

    def test_flash_attn_legacy_bool_false_emits_off(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"flash_attn": False})
        # Legacy False was an explicit "don't use flash-attn" - map to 'off'
        # rather than the new 'auto' default to preserve user intent.
        self.assertEqual(cmd[cmd.index("--flash-attn") + 1], "off")

    def test_default_f16_cache_types_are_omitted(self):
        # f16 IS llama.cpp's own default - emitting the flag anyway would be
        # noisy in logs for the common case and match no user intent that
        # wasn't already the default.
        cmd = build_llama_cmd("/models/m.gguf", 8080, {
            "cache_type_k": "f16",
            "cache_type_v": "f16",
        })
        self.assertNotIn("--cache-type-k", cmd)
        self.assertNotIn("--cache-type-v", cmd)

    def test_cache_type_k_quantized_emits_flag(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"cache_type_k": "q8_0"})
        self.assertEqual(cmd[cmd.index("--cache-type-k") + 1], "q8_0")

    def test_cache_type_v_quantized_emits_flag(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"cache_type_v": "q4_0"})
        self.assertEqual(cmd[cmd.index("--cache-type-v") + 1], "q4_0")

    def test_cache_type_case_and_whitespace_are_normalized(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {
            "cache_type_k": "  Q8_0  ",
            "cache_type_v": "  BF16 ",
        })
        self.assertEqual(cmd[cmd.index("--cache-type-k") + 1], "q8_0")
        self.assertEqual(cmd[cmd.index("--cache-type-v") + 1], "bf16")

    def test_unknown_cache_type_is_dropped(self):
        # A corrupt preset or hand-crafted request could send a value
        # llama-server would refuse - drop the flag instead of shipping it
        # and letting the container die at startup with an opaque error.
        # The UI dropdown only offers the real types, so a UI-only path
        # can't reach this.
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"cache_type_k": "garbage"})
        self.assertNotIn("--cache-type-k", cmd)

    def test_k_quantized_emitted_without_flash_attn(self):
        # llama-server has NO guard requiring flash-attn for K quantization
        # (only V), so emitting --cache-type-k q8_0 alone must be legal.
        # See src/llama-context.cpp:1018 - the ggml_is_quantized(params.type_v)
        # check only fires on V.
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"cache_type_k": "q8_0"})
        self.assertNotIn("--flash-attn", cmd)
        self.assertEqual(cmd[cmd.index("--cache-type-k") + 1], "q8_0")

    def test_v_quantized_without_flash_attn_still_emitted_verbatim(self):
        # We deliberately DO NOT second-guess the user server-side: if the
        # UI's grey-out is bypassed and a V-quant + no-flash-attn config
        # reaches build_llama_cmd, ship it as-is. llama-server's own error
        # ("quantized V cache was requested, but this requires Flash Attention")
        # is far more actionable than a silently dropped flag would be.
        cmd = build_llama_cmd("/models/m.gguf", 8080, {"cache_type_v": "q4_0"})
        self.assertNotIn("--flash-attn", cmd)
        self.assertEqual(cmd[cmd.index("--cache-type-v") + 1], "q4_0")

    def test_all_three_flags_together(self):
        cmd = build_llama_cmd("/models/m.gguf", 8080, {
            "flash_attn": "on",
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
        })
        self.assertEqual(cmd[cmd.index("--flash-attn") + 1], "on")
        self.assertEqual(cmd[cmd.index("--cache-type-k") + 1], "q8_0")
        self.assertEqual(cmd[cmd.index("--cache-type-v") + 1], "q8_0")

    def test_non_default_non_quantized_types_still_emit(self):
        # f32 and bf16 are non-default (llama.cpp defaults to f16) but not
        # quantized, so the emission logic ("emit when set AND not f16 AND
        # in the whitelist") must include them. If the whitelist ever loses
        # bf16 or f32, or the guard ever accidentally checks "quantized",
        # this test fails.
        cmd = build_llama_cmd("/models/m.gguf", 8080, {
            "cache_type_k": "f32",
            "cache_type_v": "bf16",
        })
        self.assertEqual(cmd[cmd.index("--cache-type-k") + 1], "f32")
        self.assertEqual(cmd[cmd.index("--cache-type-v") + 1], "bf16")


class LaunchInstanceKvCacheFieldsTests(unittest.TestCase):
    """launch_instance() must copy the three kwargs into inst["config"] with
    normalization (tri-state string for flash_attn, lowercase+trim for the
    cache types), so the running instance's config is authoritative for later
    reads - restart, cluster snapshot, live preset merge."""

    def setUp(self):
        with instances_lock:
            self._saved_instances = {inst_id: dict(inst) for inst_id, inst in instances.items()}
            instances.clear()

    def tearDown(self):
        with instances_lock:
            instances.clear()
            instances.update(self._saved_instances)

    @patch("api.instances.save_state")
    @patch("api.instances._run_container")
    @patch("api.instances.is_port_available", return_value=True)
    def test_launch_instance_stores_kv_fields(
        self, _is_port_mock, run_container_mock, _save_state_mock,
    ):
        fake_container = Mock()
        fake_container.id = "abc123"
        run_container_mock.return_value = (fake_container, None)

        inst, err = instances_api.launch_instance(
            model_path="/models/chat.gguf",
            port=8000,
            ctx_size=4096,
            flash_attn="on",
            cache_type_k="q8_0",
            cache_type_v="q4_0",
        )

        self.assertIsNone(err)
        self.assertEqual(inst["config"]["flash_attn"], "on")
        self.assertEqual(inst["config"]["cache_type_k"], "q8_0")
        self.assertEqual(inst["config"]["cache_type_v"], "q4_0")

    @patch("api.instances.save_state")
    @patch("api.instances._run_container")
    @patch("api.instances.is_port_available", return_value=True)
    def test_launch_instance_defaults_are_zero_values(
        self, _is_port_mock, run_container_mock, _save_state_mock,
    ):
        # Callers that don't mention the new kwargs still get the keys in
        # config at their zero values, so downstream reads never KeyError.
        # flash_attn's zero value is 'auto' to match llama.cpp's own default.
        fake_container = Mock()
        fake_container.id = "abc123"
        run_container_mock.return_value = (fake_container, None)

        inst, err = instances_api.launch_instance(
            model_path="/models/chat.gguf",
            port=8000,
            ctx_size=4096,
        )

        self.assertIsNone(err)
        self.assertEqual(inst["config"]["flash_attn"], "auto")
        self.assertEqual(inst["config"]["cache_type_k"], "")
        self.assertEqual(inst["config"]["cache_type_v"], "")

    @patch("api.instances.save_state")
    @patch("api.instances._run_container")
    @patch("api.instances.is_port_available", return_value=True)
    def test_launch_instance_normalizes_cache_types(
        self, _is_port_mock, run_container_mock, _save_state_mock,
    ):
        fake_container = Mock()
        fake_container.id = "abc123"
        run_container_mock.return_value = (fake_container, None)

        inst, err = instances_api.launch_instance(
            model_path="/models/chat.gguf",
            port=8000,
            ctx_size=4096,
            cache_type_k="  Q8_0 ",
            cache_type_v=" BF16  ",
        )

        self.assertIsNone(err)
        self.assertEqual(inst["config"]["cache_type_k"], "q8_0")
        self.assertEqual(inst["config"]["cache_type_v"], "bf16")

    @patch("api.instances.save_state")
    @patch("api.instances._run_container")
    @patch("api.instances.is_port_available", return_value=True)
    def test_launch_instance_folds_legacy_bool_flash_attn(
        self, _is_port_mock, run_container_mock, _save_state_mock,
    ):
        # A restart/adopt path handing back an old config still has bool - the
        # normalization at launch_instance must fold it so the stored config
        # is always the new tri-state string.
        fake_container = Mock()
        fake_container.id = "abc123"
        run_container_mock.return_value = (fake_container, None)

        inst, err = instances_api.launch_instance(
            model_path="/models/chat.gguf",
            port=8000,
            ctx_size=4096,
            flash_attn=True,
        )

        self.assertIsNone(err)
        self.assertEqual(inst["config"]["flash_attn"], "on")


class InstancesCreateRouteKvCacheTests(unittest.TestCase):
    """POST /api/instances must lift flash_attn / cache_type_k / cache_type_v
    out of the JSON body and forward them as kwargs. If the route drops any,
    the user's UI choice vanishes silently at launch."""

    def setUp(self):
        self.app = Flask(__name__)
        self.app.register_blueprint(instances_api.bp)
        self.client = self.app.test_client()
        with instances_lock:
            self._saved_instances = {inst_id: dict(inst) for inst_id, inst in instances.items()}
            instances.clear()

    def tearDown(self):
        with instances_lock:
            instances.clear()
            instances.update(self._saved_instances)

    @patch("api.instances.launch_instance")
    def test_create_route_forwards_kv_fields(self, launch_mock):
        launch_mock.return_value = ({"id": "inst-1"}, None)
        with patch("api.instances._public_instance", side_effect=lambda inst: inst):
            resp = self.client.post(
                "/api/instances",
                json={
                    "model_path": "/models/chat.gguf",
                    "port": 8000,
                    "ctx_size": 4096,
                    "flash_attn": "on",
                    "cache_type_k": "q8_0",
                    "cache_type_v": "q4_0",
                },
            )
        self.assertEqual(resp.status_code, 201)
        kwargs = launch_mock.call_args.kwargs
        self.assertEqual(kwargs["flash_attn"], "on")
        self.assertEqual(kwargs["cache_type_k"], "q8_0")
        self.assertEqual(kwargs["cache_type_v"], "q4_0")

    @patch("api.instances.launch_instance")
    def test_create_route_defaults_omitted_kv_fields(self, launch_mock):
        launch_mock.return_value = ({"id": "inst-1"}, None)
        with patch("api.instances._public_instance", side_effect=lambda inst: inst):
            self.client.post(
                "/api/instances",
                json={
                    "model_path": "/models/chat.gguf",
                    "port": 8000,
                    "ctx_size": 4096,
                },
            )
        kwargs = launch_mock.call_args.kwargs
        # No body value -> the tri-state default 'auto' (matching llama.cpp).
        self.assertEqual(kwargs["flash_attn"], "auto")
        self.assertEqual(kwargs["cache_type_k"], "")
        self.assertEqual(kwargs["cache_type_v"], "")

    @patch("api.instances.launch_instance")
    def test_create_route_accepts_legacy_bool_flash_attn(self, launch_mock):
        # A JS client from before the tri-state rollout, or an older API
        # caller, still sends a bool. The stored config has to end up as the
        # tri-state string so downstream reads don't have to defend against
        # both shapes - normalization happens inside launch_instance, so
        # here we only assert the route forwards the value unchanged.
        launch_mock.return_value = ({"id": "inst-1"}, None)
        with patch("api.instances._public_instance", side_effect=lambda inst: inst):
            self.client.post(
                "/api/instances",
                json={
                    "model_path": "/models/chat.gguf",
                    "port": 8000,
                    "ctx_size": 4096,
                    "flash_attn": True,
                },
            )
        # The route no longer coerces at the boundary - it forwards whatever
        # came in and lets launch_instance's normalize_flash_attn fold it.
        self.assertIs(launch_mock.call_args.kwargs["flash_attn"], True)


class PresetKvCacheTests(unittest.TestCase):
    """The three new keys are shared preset fields (behavior knobs, not
    topology), so they live in the main preset body rather than in
    PRESET_HARDWARE_KEYS - matching ctx_size, which is also a shared knob
    that affects VRAM but describes model behavior, not node hardware."""

    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(presets_api.bp)
        self.client = app.test_client()

    def test_kv_fields_are_NOT_in_hardware_keys(self):
        # If either key accidentally lands in PRESET_HARDWARE_KEYS, the
        # cluster path silently overlays them per-node - so a preset saved
        # on one node would set an override for that node alone and other
        # nodes would keep the shared base indefinitely.
        self.assertNotIn("flash_attn", presets_api.PRESET_HARDWARE_KEYS)
        self.assertNotIn("cache_type_k", presets_api.PRESET_HARDWARE_KEYS)
        self.assertNotIn("cache_type_v", presets_api.PRESET_HARDWARE_KEYS)

    def test_preset_save_persists_kv_fields(self):
        storage = Mock()
        storage.get_preset.return_value = {}
        with patch("api.presets.get_storage", return_value=storage), \
             patch("api.presets._apply_live_preset_changes"):
            resp = self.client.put("/api/presets/models/chat.gguf", json={
                "ctx_size": 4096,
                "flash_attn": "on",
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
            })
        self.assertEqual(resp.status_code, 200)
        _, saved_preset = storage.save_preset.call_args.args
        self.assertEqual(saved_preset["flash_attn"], "on")
        self.assertEqual(saved_preset["cache_type_k"], "q8_0")
        self.assertEqual(saved_preset["cache_type_v"], "q8_0")

    def test_preset_save_normalizes_case_and_whitespace(self):
        storage = Mock()
        storage.get_preset.return_value = {}
        with patch("api.presets.get_storage", return_value=storage), \
             patch("api.presets._apply_live_preset_changes"):
            self.client.put("/api/presets/models/chat.gguf", json={
                "ctx_size": 4096,
                "flash_attn": "  ON ",
                "cache_type_k": " Q4_0 ",
                "cache_type_v": " BF16 ",
            })
        _, saved_preset = storage.save_preset.call_args.args
        self.assertEqual(saved_preset["flash_attn"], "on")
        self.assertEqual(saved_preset["cache_type_k"], "q4_0")
        self.assertEqual(saved_preset["cache_type_v"], "bf16")

    def test_preset_save_defaults_are_auto_and_empty(self):
        # A save that doesn't mention the keys must still write them - a
        # subsequent save from an older form would otherwise leave a stale
        # value behind (same invariant as split_mode/tensor_split).
        # flash_attn's default is 'auto', matching llama.cpp's own default.
        storage = Mock()
        storage.get_preset.return_value = {
            "flash_attn": "on", "cache_type_k": "q8_0", "cache_type_v": "q8_0",
        }
        with patch("api.presets.get_storage", return_value=storage), \
             patch("api.presets._apply_live_preset_changes"):
            self.client.put("/api/presets/models/chat.gguf", json={"ctx_size": 4096})
        _, saved_preset = storage.save_preset.call_args.args
        self.assertEqual(saved_preset["flash_attn"], "auto")
        self.assertEqual(saved_preset["cache_type_k"], "")
        self.assertEqual(saved_preset["cache_type_v"], "")

    def test_preset_save_folds_legacy_bool_flash_attn(self):
        # A legacy JS client or an older API caller still sends bool. The
        # saved preset must land as the tri-state string so live-merge and
        # restart paths never see a bool leaking through.
        storage = Mock()
        storage.get_preset.return_value = {}
        with patch("api.presets.get_storage", return_value=storage), \
             patch("api.presets._apply_live_preset_changes"):
            self.client.put("/api/presets/models/chat.gguf", json={
                "ctx_size": 4096,
                "flash_attn": True,
            })
        _, saved_preset = storage.save_preset.call_args.args
        self.assertEqual(saved_preset["flash_attn"], "on")


class InstancesRestartRouteKvCacheTests(unittest.TestCase):
    """POST /api/instances/<id>/restart forwards the stopped instance's
    kv-cache config back into launch_instance. Missing any of the three
    would mean a restart silently reverts to defaults - which would look
    like the feature "forgot" itself the first time an operator stops and
    starts an instance to apply an unrelated change."""

    def setUp(self):
        self.app = Flask(__name__)
        self.app.register_blueprint(instances_api.bp)
        self.client = self.app.test_client()
        with instances_lock:
            self._saved_instances = {inst_id: dict(inst) for inst_id, inst in instances.items()}
            instances.clear()

    def tearDown(self):
        with instances_lock:
            instances.clear()
            instances.update(self._saved_instances)

    @patch("api.instances.save_state")
    @patch("api.instances.release_instance_reservations")
    @patch("api.instances.is_port_available", return_value=True)
    @patch("api.instances._admin_ui_enforces_eviction", return_value=False)
    @patch("api.instances._would_ui_launch_exceed_limit", return_value=False)
    @patch("api.instances._merge_preset_into_config", side_effect=lambda _p, cfg: cfg)
    @patch("api.instances.launch_instance")
    def test_restart_route_preserves_kv_fields(
        self, launch_mock, _merge_mock, _would_exceed_mock, _admin_mock,
        _is_port_mock, _release_mock, _save_state_mock,
    ):
        # _merge_preset_into_config is patched to identity so this isolates
        # the restart wiring from the merge behavior (which has its own tests
        # below) - here we care that config -> launch_instance is complete.
        launch_mock.return_value = ({"id": "inst-1"}, None)

        with instances_lock:
            instances["inst-1"] = {
                "id": "inst-1",
                "model_path": "/models/chat.gguf",
                "port": 8000,
                "status": "stopped",
                "stats": {},
                "config": {
                    "n_gpu_layers": -1,
                    "ctx_size": 4096,
                    "flash_attn": "on",
                    "cache_type_k": "q8_0",
                    "cache_type_v": "q4_0",
                },
            }

        with patch("api.instances._public_instance", side_effect=lambda inst: inst):
            resp = self.client.post("/api/instances/inst-1/restart", json={})

        self.assertIn(resp.status_code, (200, 201))
        kwargs = launch_mock.call_args.kwargs
        self.assertEqual(kwargs["flash_attn"], "on")
        self.assertEqual(kwargs["cache_type_k"], "q8_0")
        self.assertEqual(kwargs["cache_type_v"], "q4_0")


class MergePresetIntoConfigKvCacheTests(unittest.TestCase):
    """_merge_preset_into_config is the "live preset apply" path - if any of
    the three new keys isn't in the whitelist, a preset edit silently doesn't
    take effect for them, and the user thinks it did."""

    def test_merge_preset_overlays_kv_fields(self):
        base_config = {"n_gpu_layers": -1, "ctx_size": 4096}
        preset = {"flash_attn": "on", "cache_type_k": "q8_0", "cache_type_v": "q4_0"}

        storage = Mock()
        storage.get_preset.return_value = preset
        with patch("storage.get_storage", return_value=storage):
            merged = instances_api._merge_preset_into_config("/models/chat.gguf", base_config)

        self.assertEqual(merged["flash_attn"], "on")
        self.assertEqual(merged["cache_type_k"], "q8_0")
        self.assertEqual(merged["cache_type_v"], "q4_0")

    def test_merge_preset_overrides_base_kv_fields(self):
        # A preset edit AFTER launch has to win over the value baked in at
        # launch, otherwise the reaper's re-read behavior on live instances
        # doesn't reach these fields.
        base_config = {"flash_attn": "off", "cache_type_k": "", "cache_type_v": ""}
        preset = {"flash_attn": "on", "cache_type_k": "q8_0", "cache_type_v": "q8_0"}

        storage = Mock()
        storage.get_preset.return_value = preset
        with patch("storage.get_storage", return_value=storage):
            merged = instances_api._merge_preset_into_config("/models/chat.gguf", base_config)

        self.assertEqual(merged["flash_attn"], "on")
        self.assertEqual(merged["cache_type_k"], "q8_0")
        self.assertEqual(merged["cache_type_v"], "q8_0")

    def test_merge_preset_leaves_base_when_preset_omits_kv_fields(self):
        # A pre-feature preset has none of the keys. The merged config must
        # keep the base's values rather than clobbering them with defaults,
        # which would silently disable Flash Attention on the next reap for
        # a running instance that was launched with it on.
        base_config = {"flash_attn": "on", "cache_type_k": "q8_0", "cache_type_v": "q8_0"}
        preset = {"ctx_size": 4096}  # pre-feature preset

        storage = Mock()
        storage.get_preset.return_value = preset
        with patch("storage.get_storage", return_value=storage):
            merged = instances_api._merge_preset_into_config("/models/chat.gguf", base_config)

        self.assertEqual(merged["flash_attn"], "on")
        self.assertEqual(merged["cache_type_k"], "q8_0")
        self.assertEqual(merged["cache_type_v"], "q8_0")


if __name__ == "__main__":
    unittest.main()
