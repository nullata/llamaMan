# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""The Engine interface: everything that differs between inference backends
when llamaman spawns, reaches and talks to a sibling container.

An engine is stateless - one module-level instance per backend, looked up by
name through core.engines.get_engine(). It answers questions from an instance
config; it never touches Docker itself. `container_spec()` returns the
containers.run(**kwargs) dict minus GPU attachment, and api/instances.py
resolves `gpu_attachment()` into devices with its host-side helpers (vendor
detection, render GIDs), so a spec can be built and compared in tests without
a Docker daemon.
"""

import json

# llama.cpp-only launch fields and the values that mean "not used". An engine
# with an explicit launch_fields set rejects any of these that is set to
# something else (reject_unsupported_fields), so an API caller can't believe
# it configured e.g. GPU layers on an engine that ignores them. Dependent
# fields (spec_type, mmproj_path, dry_multiplier, ...) are not listed: they
# only take effect when their switch is on, and the switch is listed.
LLAMACPP_ONLY_FIELDS = {
    "n_gpu_layers": (-1, "-1"),
    "n_cpu_moe_layers": (0, "0"),
    "threads": (0, "0"),
    "threads_batch": (0, "0"),
    "parallel": (0, "0"),
    "extra_args": (),
    "spec_enabled": (),
    "mmproj_enabled": (),
    "split_mode": ("layer",),
    "tensor_split": (),
    "flash_attn": ("auto",),
    "reasoning_format": ("auto",),
    "load_mode": ("auto",),
    "cache_type_k": ("f16",),
    "cache_type_v": ("f16",),
    "dry_enabled": (),
    "embedding_model": (),
}


def _is_unset(value, defaults) -> bool:
    if value is None or value is False:
        return True
    if isinstance(value, str):
        value = value.strip().lower()
        if value == "":
            return True
    return value in defaults


class Engine:
    # Registry key, stored as config["engine"] / preset["engine"].
    name = ""
    # Human-readable name for the UI.
    label = ""
    # Port the server listens on inside its container. Also the port published
    # on the host (bare-metal) and the one reached by container name (Docker).
    internal_port = 8080
    # ctx_size used when nothing (request, preset) says otherwise.
    default_ctx_size = 4096

    # What the engine can do. Read by the API (validation), the UI (which
    # fields to show) and the cluster snapshot (which nodes can launch it).
    #   max_concurrency        0 = no engine-imposed cap; N = requests the
    #                          server can run at once (the gate is forced to N)
    #   supports_gpu_layers    --n-gpu-layers style partial offload
    #   supports_spec_decoding draft-model speculative decoding
    #   supports_embeddings    can be launched as an embedding model
    #   virtual_models         models are a fixed catalogue, not files on disk
    #   single_instance_per_model  one live instance per model per node
    capabilities = {
        "max_concurrency": 0,
        "supports_gpu_layers": False,
        "supports_spec_decoding": False,
        "supports_embeddings": False,
        "virtual_models": False,
    }

    # Launch-form fields this engine reads. None means "every field" (the
    # llama.cpp form as it has always been). An engine that lists fields gets
    # every other llama.cpp-specific field rejected by the API when set to a
    # non-default value - see core.engines.reject_unsupported_fields.
    launch_fields: frozenset | None = None

    # Engine-specific launch fields, stored in the instance config and the
    # preset under these keys (validated by parse_options).
    option_keys: tuple = ()

    def owns_model_path(self, model_path: str | None) -> bool:
        """True when model_path is one of this engine's virtual models, so a
        request naming it runs on this engine without an explicit engine."""
        return False

    def parse_options(self, body: dict, model_path: str) -> tuple[dict, str | None]:
        """Validate the option_keys fields of an API body: (options, error)."""
        return {}, None

    def reject_unsupported_fields(self, body: dict) -> str | None:
        """An error naming the llama.cpp-only fields a body sets that this
        engine would ignore, or None."""
        if self.launch_fields is None:
            return None
        bad = sorted(k for k, defaults in LLAMACPP_ONLY_FIELDS.items()
                     if k not in self.launch_fields and not _is_unset(body.get(k), defaults))
        if not bad:
            return None
        return f"{self.label} does not support: {', '.join(bad)}"

    def enforce_capabilities(self, config: dict) -> dict:
        """Clamp a config (instance or preset) to what the engine can do, in
        place. Returns it for chaining."""
        cap = int(self.capabilities.get("max_concurrency") or 0)
        if cap > 0:
            try:
                mc = int(config.get("max_concurrent") or 0)
            except (TypeError, ValueError):
                mc = 0
            if mc <= 0 or mc > cap:
                config["max_concurrent"] = cap
        if not self.capabilities.get("supports_embeddings") and config.get("embedding_model"):
            config["embedding_model"] = False
        if not self.capabilities.get("supports_spec_decoding") and config.get("spec_enabled"):
            config["spec_enabled"] = False
        return config

    # ------------------------------------------------------------------ image
    def default_image(self) -> str:
        raise NotImplementedError

    def image(self, config: dict) -> str:
        return (config.get("image") or "").strip() or self.default_image()

    def image_missing_message(self, image_name: str) -> str:
        return (f"Docker image '{image_name}' not found. Pull it in the Docker Images tab, "
                f"or run: docker pull {image_name}")

    # --------------------------------------------------------------- container
    def command(self, model_path: str, config: dict) -> list[str]:
        """Arguments passed to the image's entrypoint."""
        return []

    def environment(self, model_path: str, config: dict) -> dict:
        return {}

    def volumes(self, model_path: str, config: dict) -> dict:
        return {}

    def ulimits(self, config: dict) -> list:
        return []

    def gpu_attachment(self, config: dict, vendor: str | None) -> str:
        """How api/instances should attach GPUs:
        'none'   - no devices (CPU only)
        'vendor' - whatever the detected vendor needs (CUDA device_requests,
                   ROCm /dev/kfd+/dev/dri, Intel/Vulkan /dev/dri)
        'nvidia' - CUDA device_requests regardless of the detected vendor
        """
        return "vendor"

    def container_spec(self, *, inst_id: str, container_name: str,
                       model_path: str, server_port: int, config: dict) -> dict:
        """containers.run(**kwargs) for this instance, without GPU devices."""
        from config import LLAMA_NETWORK

        kwargs = dict(
            image=self.image(config),
            # None, not [], for an entrypoint that takes no arguments.
            command=self.command(model_path, config) or None,
            name=container_name,
            network=LLAMA_NETWORK,
            volumes=self.volumes(model_path, config),
            ports={self.internal_port: server_port},
            detach=True,
            labels={
                "llamaman.instance_id": inst_id,
                "llamaman.model_path": model_path,
                "llamaman.port": str(server_port),
                "llamaman.config": json.dumps(config),
            },
        )
        memory_limit = config.get("memory_limit")
        if memory_limit:
            kwargs["mem_limit"] = memory_limit
        env = self.environment(model_path, config)
        if env:
            kwargs["environment"] = env
        ulimits = self.ulimits(config)
        if ulimits:
            kwargs["ulimits"] = ulimits
        return kwargs

    # --------------------------------------------------------------- readiness
    def load_timeout(self) -> int:
        """Seconds to wait for a freshly started container to become ready."""
        from config import MODEL_LOAD_TIMEOUT
        return MODEL_LOAD_TIMEOUT

    def on_ready(self, inst: dict) -> None:
        """Called when an instance of this engine first reports healthy."""

    def load_stage(self, inst: dict) -> dict | None:
        """While starting: {"stage", "detail", "percent"} for the instance
        card, or None when the engine can't tell."""
        return None

    # ------------------------------------------------------------------ models
    def display_name(self, model_path: str) -> str:
        """The instance card's model name."""
        from pathlib import Path
        return Path(model_path).name

    def served_model_names(self, model_path: str) -> list[str]:
        """Lowercase names a request's `model` field may carry for this
        instance, beyond the llamaman-wide filename rules."""
        return []

    # ------------------------------------------------------------ availability
    def availability(self, vendor: str | None) -> tuple[bool, str]:
        """(can this node launch it, reason when it can't)."""
        return True, ""

    def describe(self, vendor: str | None) -> dict:
        """JSON-able summary for the API, the UI and the cluster snapshot."""
        ok, reason = self.availability(vendor)
        return {
            "name": self.name,
            "label": self.label,
            "available": ok,
            "reason": reason,
            "capabilities": dict(self.capabilities),
            "launch_fields": sorted(self.launch_fields) if self.launch_fields is not None else None,
        }
