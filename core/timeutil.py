# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Tiny timestamp helpers.

The wire/storage format for timestamps is ISO 8601 in UTC with millisecond
precision and a trailing "Z" (e.g. "2026-05-05T14:32:18.123Z"). SQL columns
are DATETIME(3); on-disk JSON records and API payloads are strings.

Use these helpers everywhere a timestamp crosses a boundary so the format
stays consistent.
"""

from datetime import datetime, timezone


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_iso() -> str:
    return to_iso(now_utc())


def to_iso(dt: datetime) -> str:
    """Format a datetime as ISO 8601 UTC with ms precision and 'Z' suffix.

    Naive datetimes are assumed to be UTC.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_iso(s: str) -> datetime:
    """Parse an ISO 8601 string. Accepts both '...Z' and '...+00:00'."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def epoch_ms_to_iso(ms: int | float) -> str:
    return to_iso(datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc))


def epoch_s_to_iso(s: int | float) -> str:
    return to_iso(datetime.fromtimestamp(int(s), tz=timezone.utc))
