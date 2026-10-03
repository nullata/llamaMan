# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

from flask import Blueprint, jsonify

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
