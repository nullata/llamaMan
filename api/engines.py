# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

from flask import Blueprint, jsonify, request

from core.engines import ENGINES, describe_engines
from core.gpu import get_vendor

bp = Blueprint("engines", __name__)


def engines_snapshot() -> dict:
    """This node's inference engines: availability (with the reason when an
    engine can't run here), capabilities, the launch fields each reads, and
    each virtual-model engine's catalogue. Served at /api/engines and
    published in the cluster heartbeat so a peer's UI only offers an engine
    on nodes that can launch it."""
    vendor = get_vendor()
    engines = describe_engines(vendor)
    for d in engines:
        eng = ENGINES[d["name"]]
        if eng.capabilities.get("virtual_models"):
            d["models"] = eng.virtual_models()
    return {"vendor": vendor, "engines": engines}


@bp.route("/api/engines", methods=["GET"])
def api_engines():
    return jsonify(engines_snapshot())


@bp.route("/api/engines/<engine>/download", methods=["POST"])
def api_engine_model_download(engine):
    """Pre-download a virtual model's files with llamaman's downloader.

    For Strata: the GGUF shards, from the Hugging Face revision Strata pins,
    into MODELS_DIR/strata/<tag>/ - from where they are bind-mounted into the
    container so its setup skips its own download. It is an ordinary download
    (Downloads tab: progress, pause/resume, retry, HF tokens, speed limits)
    that skips update-check provenance: a pinned file must not be "updated".

    body: {"model": "strata/qwen-IQ2_XS", "hf_token_id"?, "hf_token"?,
           "speed_limit_mbps"?}
    """
    from api.downloads import resolve_request_token, start_download
    from core.helpers import public_dict

    eng = ENGINES.get((engine or "").strip().lower())
    if eng is None:
        return jsonify({"error": f"unknown engine '{engine}'"}), 404
    body = request.get_json(force=True) or {}
    model_path = eng.canonical_model_path(str(body.get("model") or ""))
    plan = eng.download_plan(model_path) if model_path else None
    if plan is None:
        return jsonify({"error": f"'{body.get('model')}' is not a downloadable {eng.label} model"}), 400
    ok, reason = eng.availability(get_vendor())
    if not ok:
        return jsonify({"error": reason}), 400

    existing = next((m for m in eng.virtual_models() if m["path"] == model_path), {})
    if existing.get("local_shards"):
        return jsonify({"error": f"{plan['model_id']} is already downloaded"}), 409
    current = existing.get("download") or {}
    if current.get("status") in ("downloading", "paused", "failed"):
        return jsonify({
            "error": f"{plan['model_id']} already has a {current['status']} download; "
                     "resume, retry or cancel it in the Downloads tab",
            "download_id": current["id"],
        }), 409

    token, token_id, err = resolve_request_token(body)
    if err:
        return jsonify({"error": err}), 400
    dl, err, code = start_download(
        plan["repo_id"], plan["filename"], token, token_id,
        float(body.get("speed_limit_mbps", 0) or 0),
        dest_path=plan["dest_path"], revision=plan["revision"],
        record_source=False, extra={"engine_model": plan["model_id"]},
    )
    if err:
        return jsonify({"error": err}), code
    return jsonify(public_dict(dl)), 201
