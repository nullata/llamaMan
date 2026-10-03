# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""llama.cpp's llama-server - the engine llamaman has always run.

Everything here is the pre-engine-abstraction launch path moved behind the
Engine interface, unchanged: the flags still come from
core.helpers.build_llama_cmd, the mounts and limits are the ones
api/instances._run_container used to build inline.
tests/fixtures/llamacpp_container_spec.json holds the containers.run kwargs
captured before the move; tests/test_engines.py compares against it.
"""

from core.engines.base import Engine


class LlamaCppEngine(Engine):
    name = "llamacpp"
    label = "llama.cpp"
    internal_port = 8080
    capabilities = {
        "max_concurrency": 0,
        "supports_gpu_layers": True,
        "supports_spec_decoding": True,
        "supports_embeddings": True,
        "virtual_models": False,
    }
    launch_fields = None  # the whole launch form

    def default_image(self) -> str:
        from config import LLAMA_IMAGE
        return LLAMA_IMAGE

    def command(self, model_path: str, config: dict) -> list[str]:
        from core.helpers import build_llama_cmd
        return build_llama_cmd(model_path, self.internal_port, config)

    def volumes(self, model_path: str, config: dict) -> dict:
        # SOURCE must be a path on the Docker HOST (the daemon's filesystem).
        # When llamaman itself runs in Docker, HOST_MODELS_DIR / HOST_LOGS_DIR
        # are the real host paths; they default to MODELS_DIR / LOGS_DIR for
        # bare-metal.
        from config import HOST_LOGS_DIR, HOST_MODELS_DIR, LOGS_DIR, MODELS_DIR
        return {
            HOST_MODELS_DIR: {"bind": MODELS_DIR, "mode": "ro"},
            HOST_LOGS_DIR: {"bind": LOGS_DIR, "mode": "rw"},
        }

    def container_spec(self, **kw) -> dict:
        kwargs = super().container_spec(**kw)
        threads = kw["config"].get("threads")
        if threads:
            kwargs["nano_cpus"] = int(float(threads) * 1e9)
        return kwargs

    def gpu_attachment(self, config: dict, vendor: str | None) -> str:
        try:
            n_gpu_layers = int(config.get("n_gpu_layers", -1))
        except (TypeError, ValueError):
            n_gpu_layers = -1
        # CPU-only: attach no GPU devices at all. Besides honoring the user's
        # intent, this avoids Docker's CDI GPU discovery, which errors on hosts
        # without a configured GPU runtime (e.g. WSL without the NVIDIA
        # container toolkit).
        return "none" if n_gpu_layers == 0 else "vendor"
