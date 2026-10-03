# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Launch configs for the llama.cpp container-spec snapshot test.

tests/fixtures/llamacpp_container_spec.json holds the docker
containers.run(**kwargs) that _run_container produced for each of these,
captured from the code as it was BEFORE the engine abstraction (core/engines)
was introduced. Do not regenerate it to make a failing test pass: a diff
means the llama.cpp launch changed.
"""

VENDORS = ["cuda", "rocm", "intel", "vulkan", None]

CASES = {
    "minimal": ("/models/chat.gguf", {"n_gpu_layers": -1, "ctx_size": 4096}),
    "empty_config": ("/models/chat.gguf", {}),
    "cpu_only": ("/models/chat.gguf", {"n_gpu_layers": 0, "ctx_size": 2048}),
    "gpu_pinned": ("/models/sub/m-Q4_K_M.gguf", {
        "n_gpu_layers": 40, "ctx_size": 8192, "gpu_devices": "0,1",
        "split_mode": "row", "tensor_split": "3,2",
    }),
    "gpu_all_literal": ("/models/m.gguf", {"gpu_devices": "all", "ctx_size": 4096}),
    "limits": ("/models/m.gguf", {
        "ctx_size": 4096, "threads": 6, "threads_batch": 12,
        "memory_limit": "32g", "parallel": 4,
    }),
    "image_override": ("/models/m.gguf", {"ctx_size": 4096, "image": "registry.example.org/custom/llama-server:pinned"}),
    "kitchen_sink": ("/models/big/q-00001-of-00002.gguf", {
        "n_gpu_layers": -1, "n_cpu_moe_layers": 8, "ctx_size": 65536,
        "flash_attn": "on", "cache_type_k": "q8_0", "cache_type_v": "q4_0",
        "reasoning_format": "deepseek", "load_mode": "mlock",
        "dry_enabled": True, "dry_multiplier": 0.8, "dry_base": 1.75,
        "dry_allowed_length": 2, "dry_penalty_last_n": 512,
        "share_queue_group": "chat", "spec_enabled": True,
        "spec_type": "draft-simple", "spec_draft_model": "/models/d.gguf",
        "spec_draft_n_max": 16, "spec_draft_n_min": 2,
        "mmproj_enabled": True, "mmproj_path": "/models/mmproj.gguf",
        "mmproj_offload": False, "embedding_model": False,
        "extra_args": "--top-k 20 --jinja", "max_concurrent": 2,
        "idle_timeout_min": 10,
    }),
    "embedding": ("/models/emb.gguf", {"ctx_size": 512, "embedding_model": True, "n_cpu_moe_layers": -1}),
}


def normalize_spec(kwargs: dict, repo_root: str, llama_image: str) -> dict:
    """JSON-roundtrip a containers.run kwargs dict and swap the two
    host-dependent values (the repo checkout path baked into MODELS_DIR /
    LOGS_DIR by the test env, and the vendor-resolved LLAMA_IMAGE) for
    placeholders, so the snapshot compares equal on any machine."""
    import json
    text = json.dumps(kwargs, sort_keys=True)
    text = text.replace(json.dumps(repo_root)[1:-1], "<REPO>")
    # Whole-value match only (quotes included), so an image that merely
    # starts with the default's name is left alone.
    text = text.replace(json.dumps(llama_image), '"<LLAMA_IMAGE>"')
    return json.loads(text)
