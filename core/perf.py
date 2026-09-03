# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Opt-in phase timing for diagnosing slow request paths.

Enabled with LLAMAMAN_PERF_LOG=1 (see config.PERF_LOG). When off — the default —
phase() is a near-noop: one boolean check and a contextlib no-op, no clock call,
no log line, no allocation. Nothing on the measured paths changes behavior; this
only observes.

Usage:

    from core.perf import phase

    with phase("stop_instance", inst_id=iid):
        with phase("stop_instance.stop_container", cid=cid):
            stop_container(cid)
        with phase("stop_instance.save_state"):
            save_state()

Emits one INFO line per phase when it completes:

    perf stop_instance.save_state 412ms  inst=a1b2 cid=...

The nested phases are deliberately separate log lines rather than one rollup:
the point is to see which phase dominates *live* in the log while reproducing,
and a rollup would hide a slow child behind a fast sibling's overlap.
"""

import contextlib
import time

from config import PERF_LOG, logger


def _fmt_fields(fields: dict) -> str:
    if not fields:
        return ""
    return "  " + "  ".join(f"{k}={v}" for k, v in fields.items())


@contextlib.contextmanager
def phase(name: str, **fields):
    """Time a block and log it when PERF_LOG is on. Never raises, never logs when off."""
    if not PERF_LOG:
        yield
        return
    t0 = time.monotonic()
    err = ""
    try:
        yield
    except BaseException as e:
        err = f"  error={type(e).__name__}"
        raise
    finally:
        ms = (time.monotonic() - t0) * 1000.0
        try:
            logger.info("perf %s %.0fms%s%s", name, ms, _fmt_fields(fields), err)
        except Exception:
            pass  # observability must never break the measured path
