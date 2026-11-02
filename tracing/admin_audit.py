"""Bounded, content-free operational records for the administrator workspace."""

import contextvars
import logging
from contextlib import closing, contextmanager
from datetime import datetime, timezone

from auth.database import _connection

logger = logging.getLogger(__name__)
_trace = contextvars.ContextVar("admin_audit_trace", default=None)


@contextmanager
def trace_scope(trace_id: str, actor_id: str, session_id: str):
    token = _trace.set({"trace_id": trace_id, "actor_id": actor_id, "session_id": session_id})
    try:
        yield
    finally:
        _trace.reset(token)


def current_trace():
    return _trace.get()


def record_event(*, kind: str, action: str, outcome: str, actor_id: str = "", target_id: str = "", session_id: str = "", trace_id: str = "", duration_ms: float | None = None, detail: str = "") -> None:
    """Write only caller-selected identifiers and fixed status codes, never request content."""
    try:
        with closing(_connection()) as connection, connection:
            connection.execute(
                """INSERT INTO admin_events
                   (created_at, kind, action, outcome, actor_id, target_id, session_id, trace_id, duration_ms, detail)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (datetime.now(timezone.utc).isoformat(), kind, action, outcome, actor_id,
                 target_id, session_id, trace_id, duration_ms, detail),
            )
    except Exception as exc:
        logger.warning("Administrator runtime record unavailable (%s)", type(exc).__name__)


def list_events(limit: int, offset: int = 0, kind: str = "", trace_id: str = "") -> list[dict]:
    with closing(_connection()) as connection:
        rows = connection.execute(
            """SELECT id, created_at, kind, action, outcome, actor_id, target_id,
                      session_id, trace_id, duration_ms, detail
               FROM admin_events WHERE (? = '' OR kind = ?) AND (? = '' OR trace_id = ?)
               ORDER BY id DESC LIMIT ? OFFSET ?""",
            (kind, kind, trace_id, trace_id, limit, offset),
        ).fetchall()
    return [dict(row) for row in rows]


def list_runs(limit: int, offset: int = 0) -> list[dict]:
    with closing(_connection()) as connection:
        rows = connection.execute(
            """SELECT trace_id, MAX(actor_id) AS actor_id, MAX(session_id) AS session_id,
                      MIN(created_at) AS started_at, MAX(created_at) AS last_event_at,
                      COUNT(*) AS event_count,
                      SUM(CASE WHEN outcome = 'error' THEN 1 ELSE 0 END) AS error_count,
                      SUM(CASE WHEN outcome = 'fallback' THEN 1 ELSE 0 END) AS fallback_count
               FROM admin_events WHERE trace_id != '' GROUP BY trace_id
               ORDER BY MAX(id) DESC LIMIT ? OFFSET ?""", (limit, offset),
        ).fetchall()
    return [dict(row) for row in rows]

