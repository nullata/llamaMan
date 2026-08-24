# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

import hashlib
import os

from flask import Flask, jsonify, make_response, render_template

from config import SECRET_KEY, SESSION_COOKIE_NAME, VERSION, logger
from core.migrations import run_pending_migrations
from core.state import load_state
from proxy import start_idle_proxy
from core.monitoring import start_background_poller
from storage import get_storage

import api.auth as auth
import api.models as models
import api.presets as presets
import api.instances as instances
import api.downloads as downloads
import api.system_info as system_info
import api.llamaman as llamaman
import api.settings as settings
import api.api_keys as api_keys
import api.images as images
import api.restore as restore
import api.request_log as request_log
import api.cluster as cluster


def create_app() -> Flask:
    application = Flask(__name__)

    # Namespaced cookie name so we coexist with any other Flask app on the same
    # host - cookies ignore the port, so two apps both using the default "session"
    # name overwrite each other's login.
    application.config["SESSION_COOKIE_NAME"] = SESSION_COOKIE_NAME

    # Secret key for session cookies - derived from SECRET_KEY env var,
    # or auto-generated from machine-id for zero-config single-user setups.
    if SECRET_KEY:
        application.secret_key = SECRET_KEY
    else:
        seed = "llamaman"
        try:
            with open("/etc/machine-id", "r") as f:
                seed = f.read().strip()
        except FileNotFoundError:
            pass
        application.secret_key = hashlib.sha256(seed.encode()).hexdigest()

    application.register_blueprint(auth.bp)
    application.register_blueprint(models.bp)
    application.register_blueprint(presets.bp)
    application.register_blueprint(instances.bp)
    application.register_blueprint(downloads.bp)
    application.register_blueprint(system_info.bp)
    application.register_blueprint(llamaman.bp)
    application.register_blueprint(settings.bp)
    application.register_blueprint(api_keys.bp)
    application.register_blueprint(images.bp)
    application.register_blueprint(restore.bp)
    application.register_blueprint(request_log.bp)
    application.register_blueprint(cluster.bp)

    auth.init_auth(application)

    # One handler covers every blueprint, so no route needs to know the storage
    # backend can be degraded. Raised only by the writes that cannot be replayed
    # safely against a shared database (see storage/resilient.py).
    from storage.resilient import StorageDegradedError

    @application.errorhandler(StorageDegradedError)
    def _storage_degraded(err):
        logger.warning("Rejected write while database offline: %s", err)
        return jsonify({
            "error": "Database offline - this change cannot be saved right now.",
            "detail": str(err),
            "degraded": True,
        }), 503

    # Expose the app version to every template so the footer (and any future
    # per-page version display) reads it without each route threading it in.
    @application.context_processor
    def _inject_version():
        return {"llamaman_version": VERSION}

    @application.route("/")
    def index():
        resp = make_response(render_template("index.html"))
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @application.route("/logging")
    def logging_page():
        resp = make_response(render_template("logging.html"))
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @application.route("/health")
    def health():
        return jsonify({"status": "ok"})

    return application


# ---------------------------------------------------------------------------
# Startup - runs on import (works for both gunicorn and python app.py)
# ---------------------------------------------------------------------------

# Run any pending schema migrations before anything reads timestamp-affected
# tables. Aborts startup on failure - serving traffic on a half-migrated
# schema is worse than a hard crash that surfaces in logs.
#
# Skipped when storage booted degraded (database unreachable, serving from the
# local mirror): there is no schema to migrate from here, and the mirror's copy
# of the recorded version can be ahead of what this node has actually applied.
# The reconnect path re-runs them against the real backend before it resumes.
_storage = get_storage()
if _storage.is_degraded():
    logger.warning("Storage is degraded at startup - deferring schema migrations")
else:
    run_pending_migrations(_storage)

# Load persisted state (instances, downloads) and collect proxies to restore
_deferred_proxies = load_state()

# Start background health/download poller
start_background_poller()

# Create the Flask app
app = create_app()

# Keep the subprocess-facing settings mirror in sync from boot onward so any
# download subprocess spawned right after startup sees the current values.
from api.settings import snapshot_subprocess_settings as _snapshot_subprocess_settings
try:
    _snapshot_subprocess_settings()
except Exception as _e:
    logger.warning("subprocess_settings snapshot at boot failed: %s", _e)

# Restore idle proxies from previous state
for _inst_id, _proxy_port, _internal_port in _deferred_proxies:
    try:
        start_idle_proxy(_inst_id, _proxy_port, _internal_port)
    except Exception as _e:
        logger.warning("Failed to restore proxy for %s: %s", _inst_id, _e)

# The client-facing inference port (default 42069, OpenWebUI's
# OLLAMA_BASE_URL=http://llamaman:42069) used to run a separate werkzeug
# make_server here in a daemon thread - same Flask app, different port.
# That meant requests on 42069 were served by werkzeug's threaded WSGI dev
# server while requests on 5000 went through gunicorn's gthread workers,
# and a cluster relay that spanned both runtimes was a mess to reason
# about (different concurrency models, different timeout behavior, no
# shared thread pool). Both ports are now bound by gunicorn (see
# gunicorn.conf.py); no in-process server needed here.


if __name__ == "__main__":
    # Direct execution (no gunicorn): run the Flask dev server on port 5000.
    app.run(host="0.0.0.0", port=5000, debug=False)
