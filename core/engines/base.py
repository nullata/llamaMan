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


class Engine:
    # Registry key, stored as config["engine"] / preset["engine"].
    name = ""
    # Human-readable name for the UI.
    label = ""
    # Port the server listens on inside its container. Also the port published
    # on the host (bare-metal) and the one reached by container name (Docker).
    internal_port = 8080

    # What the engine can do. Read by the API (validation), the UI (which
    # fields to show) and the cluster snapshot (which nodes can launch it).
    #   max_concurrency        0 = no engine-imposed cap; N = requests the
    #                          server can run at once (the gate is forced to N)
    #   supports_gpu_layers    --n-gpu-layers style partial offload
    #   supports_spec_decoding draft-model speculative decoding
    #   supports_embeddings    can be launched as an embedding model
    #   virtual_models         models are a fixed catalogue, not files on disk
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
            command=self.command(model_path, config),
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
