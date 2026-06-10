# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

from flask import Blueprint, jsonify, make_response, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from storage import get_storage

bp = Blueprint("auth", __name__)

# Cached after first user is created so we don't hit disk on every request
_has_users: bool | None = None


def _check_has_users() -> bool:
    global _has_users
    if _has_users is True:
        return True
    result = get_storage().user_count() > 0
    if result:
        _has_users = True
    return result


def verify_bearer_token(auth_header: str, strict: bool = False) -> str | None:
    """Validate a bearer token from an Authorization header.

    Returns None on success, or an error string if auth fails.

    strict=True:  require a valid token, period (used when require_auth is on).
    strict=False: allow the request if no API keys exist yet (management endpoints).
    """
    storage = get_storage()

    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        # Bearer is for clients only - an API key. The cluster secret is NOT a
        # client credential; peers authenticate via the X-Cluster-Secret header
        # (see is_cluster_peer_request / before_request).
        if storage.verify_api_key(token):
            return None
        return "Invalid API key"

    if strict:
        return "API key required"

    # Non-strict: allow if no keys have been created yet
    if bool(storage.get_api_keys()):
        return "API key required"
    return None


def is_require_auth_enabled() -> bool:
    """Check if the require_auth setting is on (default: True)."""
    return get_storage().get_settings().get("require_auth", True) is not False


def is_cluster_peer_request() -> bool:
    """True if the request carries a valid cluster secret in X-Cluster-Secret -
    i.e. it's a trusted node-to-node call, not a client request."""
    from core.cluster import CLUSTER_SECRET_HEADER, verify_cluster_secret
    return verify_cluster_secret(request.headers.get(CLUSTER_SECRET_HEADER, ""))


def init_auth(app):
    """Register a before_request hook that enforces login on all routes.

    Auth is completely disabled until the first user account is created
    via /setup. Until then the app works exactly like it did before auth
    was added - no redirects, no 401s, nothing blocked.
    """

    @app.before_request
    def require_login():
        # Always allow auth pages, health check, and static assets
        if request.endpoint in ("auth.login", "auth.setup", "health", "static", None):
            return

        # Trusted node-to-node call (valid X-Cluster-Secret) - allow on any
        # endpoint. This is the ONLY thing the cluster secret authorizes; it is
        # never a client bearer.
        if is_cluster_peer_request():
            return

        # Llamaman blueprint: when require_auth is on, demand a valid token
        bp_name = request.blueprints[0] if request.blueprints else None
        if bp_name == "llamaman":
            if not is_require_auth_enabled():
                return
            error = verify_bearer_token(
                request.headers.get("Authorization", ""), strict=True)
            if error:
                return jsonify({"error": error}), 401
            return

        # Check session (works for both UI pages and API calls from dashboard)
        if session.get("user"):
            return

        # API endpoints: check bearer token
        if request.path.startswith("/api/") or request.path.startswith("/v1/"):
            strict = is_require_auth_enabled()
            error = verify_bearer_token(
                request.headers.get("Authorization", ""), strict=strict)
            if error:
                return jsonify({"error": error}), 401
            return

        # Browser pages: redirect to setup or login
        if not _check_has_users():
            return redirect(url_for("auth.setup"))
        return redirect(url_for("auth.login"))


@bp.route("/login", methods=["GET", "POST"])
def login():
    storage = get_storage()

    if storage.user_count() == 0:
        return redirect(url_for("auth.setup"))

    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = storage.get_user(username)
        if user and check_password_hash(user["password_hash"], password):
            session["user"] = username
            return redirect(url_for("index"))
        error = "Invalid username or password"

    resp = make_response(render_template("login.html", error=error, setup=False))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@bp.route("/setup", methods=["GET", "POST"])
def setup():
    storage = get_storage()

    if storage.user_count() > 0:
        return redirect(url_for("auth.login"))

    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")

        if not username:
            error = "Username is required"
        elif len(password) < 4:
            error = "Password must be at least 4 characters"
        elif password != confirm:
            error = "Passwords do not match"
        else:
            global _has_users
            storage.save_user(username, generate_password_hash(password))
            _has_users = True
            session["user"] = username
            return redirect(url_for("index"))

    resp = make_response(render_template("login.html", error=error, setup=True))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@bp.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("auth.login"))
