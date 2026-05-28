# Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

# Gunicorn configuration for LlamaMan
#
# Usage:
#   gunicorn app:app
#
# Or with explicit config:
#   gunicorn -c gunicorn.conf.py app:app

import os

# Bind to both the management/cluster port (5000) and the client-facing
# inference port (42069, OpenWebUI's OLLAMA_BASE_URL). One Flask app, one
# worker pool, both ports - replaces the old werkzeug make_server-in-a-thread
# setup that ran two different WSGI runtimes side by side. Override either
# with GUNICORN_BIND (comma-separated) or LLAMAMAN_PROXY_PORT.
_default_inference_port = os.environ.get("LLAMAMAN_PROXY_PORT", "42069")
_default_bind = f"0.0.0.0:5000,0.0.0.0:{_default_inference_port}"
bind = [b.strip() for b in os.environ.get("GUNICORN_BIND", _default_bind).split(",") if b.strip()]

# IMPORTANT: Must use exactly 1 worker. The app uses in-memory state
# (instances, downloads, locks) that cannot be shared across processes.
# Use threads for concurrency instead.
workers = 1
# In a cluster, each cross-node forwarded inference request holds a thread for
# the ENTIRE generation (it lands on this UI/API port via the peer's advertise
# URL), and it competes with dashboard polls, reachability probes, and the
# cross-node management proxy. 8 starved easily and made nodes flap offline, so
# this is sized generously - threads are cheap here (the work is I/O-bound,
# mostly waiting on llama-server).
threads = 32

# Worker class - gthread supports threading within a single worker
worker_class = "gthread"

# Timeout - model launches can take a while
timeout = int(os.environ.get("GUNICORN_TIMEOUT", 300))

# Graceful shutdown timeout
graceful_timeout = 30

# Logging
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("GUNICORN_LOG_LEVEL", "info")

# IMPORTANT: preload_app MUST be False.  With preload_app=True the module-level
# code (load_state, background poller, proxy thread) runs in the *master*
# process.  The single worker is a fork with its own copy of memory, so it
# never sees status updates from the poller, instances launched by the proxy,
# or any state changes.  Worker crashes also reset state to the master's
# original snapshot instead of reading from disk.
preload_app = False
