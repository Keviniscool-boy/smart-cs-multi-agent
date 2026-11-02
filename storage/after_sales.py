"""Demo after-sales service; no payments are executed."""

import hashlib
import os
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "after_sales.sqlite3"


def _connect():
    db_path = Path(os.getenv("AFTER_SALES_DB_PATH") or DEFAULT_DB_PATH)
    if not db_path.is_absolute():
        db_path = DEFAULT_DB_PATH.parents[1] / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(db_path, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS orders (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('delivered', 'cancelled', 'paid', 'shipped')),
            amount_cents INTEGER NOT NULL CHECK(amount_cents >= 0), delivered_at TEXT, product TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS refund_requests (
            id TEXT PRIMARY KEY, order_id TEXT NOT NULL REFERENCES orders(id),
            user_id TEXT NOT NULL, reason TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('pending_review', 'cancelled', 'approved', 'rejected', 'completed')),
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS refund_tickets (
            id TEXT PRIMARY KEY, refund_id TEXT NOT NULL UNIQUE REFERENCES refund_requests(id),
            user_id TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('pending', 'processing', 'cancelled', 'resolved', 'rejected')), created_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS one_active_refund_per_order
            ON refund_requests(order_id) WHERE status = 'pending_review';
        CREATE UNIQUE INDEX IF NOT EXISTS one_open_or_approved_refund_per_order
            ON refund_requests(order_id) WHERE status IN ('pending_review', 'approved', 'completed');
        CREATE TABLE IF NOT EXISTS workbench_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            refund_id TEXT NOT NULL,
            operator_id TEXT NOT NULL,
            action TEXT NOT NULL CHECK(action IN ('approve', 'reject', 'need_info')),
            note TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS after_sales_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
            action TEXT NOT NULL, actor_id TEXT NOT NULL, refund_id TEXT NOT NULL,
            order_id TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS refund_customer_updates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            refund_id TEXT NOT NULL REFERENCES refund_requests(id),
            user_id TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS handoff_outbox (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            refund_id TEXT NOT NULL REFERENCES refund_requests(id),
            action_id INTEGER REFERENCES workbench_actions(id),
            session_id TEXT NOT NULL, user_id TEXT NOT NULL,
            action TEXT NOT NULL, note TEXT NOT NULL DEFAULT '', operator TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'running', 'done')),
            claimed_at TEXT, completed_at TEXT, response TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,
            UNIQUE(refund_id, action)
        );
        CREATE TABLE IF NOT EXISTS session_turn_locks (
            session_id TEXT PRIMARY KEY, token TEXT NOT NULL, expires_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS refund_idempotency (
            user_id TEXT NOT NULL, request_key TEXT NOT NULL,
            order_id TEXT NOT NULL, reason TEXT NOT NULL,
            refund_id TEXT NOT NULL REFERENCES refund_requests(id),
            PRIMARY KEY(user_id, request_key)
        );
    """)
    # 存量库补 session_id 列：人工恢复流程需要退款单关联会话线程。
    columns = {row["name"] for row in db.execute("PRAGMA table_info(refund_requests)")}
    if "session_id" not in columns:
        try:
            db.execute("ALTER TABLE refund_requests ADD COLUMN session_id TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                db.close()
                raise
    outbox_columns = {row["name"] for row in db.execute("PRAGMA table_info(handoff_outbox)")}
    if "completed_at" not in outbox_columns:
        try:
            db.execute("ALTER TABLE handoff_outbox ADD COLUMN completed_at TEXT")
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                db.close()
                raise
    return db


def _now():
    return datetime.now(timezone.utc)


def claim_session_turn(session_id: str, token: str) -> bool:
    db = _connect()
    try:
        db.execute("BEGIN IMMEDIATE")
        db.execute("DELETE FROM session_turn_locks WHERE session_id = ? AND expires_at <= ?", (session_id, _now().isoformat()))
        result = db.execute(
            "INSERT OR IGNORE INTO session_turn_locks VALUES (?, ?, ?)",
            (session_id, token, (_now() + timedelta(minutes=5)).isoformat()),
        )
        db.commit()
        return result.rowcount == 1
    finally:
        db.close()


def release_session_turn(session_id: str, token: str) -> None:
    with closing(_connect()) as db, db:
        db.execute("DELETE FROM session_turn_locks WHERE session_id = ? AND token = ?", (session_id, token))


def list_orders(user_id: str) -> list[dict]:
    """Provision two sample orders for each account once."""
    if not user_id:
        raise ValueError("user_id is required")
    prefix = hashlib.sha256(user_id.encode()).hexdigest()[:24].upper()
    with closing(_connect()) as db, db:
        for suffix, status, delivered in (("1", "delivered", (_now() - timedelta(days=1)).isoformat()),
                                           ("2", "cancelled", None)):
            db.execute("INSERT OR IGNORE INTO orders VALUES (?, ?, ?, ?, ?, ?)",
                       (f"DEMO-{prefix}-{suffix}", user_id, status, 29900, delivered, "演示商品"))
        rows = db.execute("SELECT id AS order_id, status, amount_cents, delivered_at, product FROM orders WHERE user_id = ? ORDER BY id", (user_id,)).fetchall()
    return [{**dict(row), "mock": True} for row in rows]


def get_order(user_id: str, order_id: str) -> dict:
    with closing(_connect()) as db:
        row = db.execute("SELECT id AS order_id, status, amount_cents, delivered_at, product FROM orders WHERE id = ? AND user_id = ?", (order_id, user_id)).fetchone()
    return {**dict(row), "mock": True} if row else {"error": "order_not_found", "message": "订单不存在或无权访问"}


def apply_refund(user_id: str, order_id: str, reason: str, idempotency_key: str, session_id: str = "") -> dict:
    reason = reason.strip()
    idempotency_key = idempotency_key.strip()
    if not reason or len(reason) > 500:
        return {"error": "invalid_reason", "message": "请填写不超过 500 字的退款原因"}
    if not idempotency_key or len(idempotency_key) > 128:
        return {"error": "invalid_idempotency_key", "message": "请提供不超过 128 字的幂等键"}
    db = _connect()
    try:
        db.execute("BEGIN IMMEDIATE")
        order = db.execute("SELECT * FROM orders WHERE id = ? AND user_id = ?", (order_id, user_id)).fetchone()
        if not order:
            return {"error": "order_not_found", "message": "订单不存在或无权访问"}
        previous = db.execute(
            "SELECT order_id, reason, refund_id FROM refund_idempotency WHERE user_id = ? AND request_key = ?",
            (user_id, idempotency_key),
        ).fetchone()
        if previous:
            if previous["order_id"] != order_id or previous["reason"] != reason:
                return {"error": "idempotency_conflict", "message": "该幂等键已用于其他请求"}
            saved = db.execute(
                "SELECT r.id, r.order_id, r.reason, r.status, r.session_id, r.created_at, t.id AS ticket_id FROM refund_requests r LEFT JOIN refund_tickets t ON t.refund_id = r.id WHERE r.id = ? AND r.user_id = ?",
                (previous["refund_id"], user_id),
            ).fetchone()
            return {**dict(saved), "reused": True, "mock": True}
        existing = db.execute("SELECT r.id, r.order_id, r.reason, r.status, r.session_id, r.created_at, t.id AS ticket_id FROM refund_requests r LEFT JOIN refund_tickets t ON t.refund_id = r.id WHERE r.order_id = ? AND r.user_id = ? AND r.status IN ('pending_review', 'approved', 'completed')", (order_id, user_id)).fetchone()
        if existing:
            if existing["status"] != "pending_review":
                return {"error": "already_approved", "message": "该订单已有通过审核的退款申请"}
            if existing["reason"] != reason:
                return {"error": "refund_exists", "message": "该订单已有退款申请，请查看原申请"}
            db.execute("INSERT INTO refund_idempotency VALUES (?, ?, ?, ?, ?)",
                       (user_id, idempotency_key, order_id, reason, existing["id"]))
            db.commit()
            return {**dict(existing), "reused": True, "mock": True}
        if session_id and db.execute(
            "SELECT 1 FROM refund_requests WHERE session_id = ? AND status = 'pending_review'",
            (session_id,),
        ).fetchone():
            return {"error": "session_busy", "message": "该会话已有等待人工审核的申请"}
        if order["status"] != "delivered" or not order["delivered_at"]:
            return {"error": "not_eligible", "message": "演示规则：只有已签收订单可以申请退款"}
        elapsed = _now() - datetime.fromisoformat(order["delivered_at"])
        if elapsed < timedelta(0):
            return {"error": "not_eligible", "message": "演示订单签收时间无效，请联系人工客服"}
        if elapsed > timedelta(days=7):
            return {"error": "not_eligible", "message": "演示规则：签收超过 7 天不能申请退款"}
        refund_id = "RF-" + uuid.uuid4().hex.upper()
        created = _now().isoformat()
        db.execute("INSERT INTO refund_requests VALUES (?, ?, ?, ?, ?, ?, ?)",
                   (refund_id, order_id, user_id, reason, "pending_review", created, session_id))
        ticket_id = "TK-" + uuid.uuid4().hex.upper()
        db.execute("INSERT INTO refund_tickets VALUES (?, ?, ?, ?, ?)",
                   (ticket_id, refund_id, user_id, "pending", created))
        db.execute("INSERT INTO refund_idempotency VALUES (?, ?, ?, ?, ?)",
                   (user_id, idempotency_key, order_id, reason, refund_id))
        db.execute("INSERT INTO after_sales_audit (created_at, action, actor_id, refund_id, order_id) VALUES (?, 'refund_requested', ?, ?, ?)",
                   (created, user_id, refund_id, order_id))
        db.commit()
        return {"id": refund_id, "order_id": order_id, "reason": reason, "session_id": session_id,
                "ticket_id": ticket_id, "status": "pending_review", "created_at": created, "reused": False, "mock": True}
    finally:
        db.close()


def list_refunds(user_id: str) -> list[dict]:
    with closing(_connect()) as db:
        rows = db.execute("SELECT r.id, r.order_id, r.reason, r.status, r.session_id, r.created_at, t.id AS ticket_id, t.status AS ticket_status, (SELECT note FROM workbench_actions a WHERE a.refund_id = r.id AND a.action = 'need_info' ORDER BY a.id DESC LIMIT 1) AS requested_info, (SELECT status FROM handoff_outbox h WHERE h.refund_id = r.id ORDER BY h.id DESC LIMIT 1) AS resume_status FROM refund_requests r LEFT JOIN refund_tickets t ON t.refund_id = r.id WHERE r.user_id = ? ORDER BY r.created_at DESC", (user_id,)).fetchall()
    return [{**dict(row), "mock": True} for row in rows]


def cancel_refund(user_id: str, refund_id: str) -> dict:
    """Cancel a pending request and its ticket atomically; repeated cancellation is safe."""
    db = _connect()
    try:
        db.execute("BEGIN IMMEDIATE")
        refund = db.execute(
            "SELECT id, order_id, reason, status, session_id, created_at FROM refund_requests WHERE id = ? AND user_id = ?",
            (refund_id, user_id),
        ).fetchone()
        if not refund:
            return {"error": "refund_not_found", "message": "退款申请不存在或无权访问"}
        if refund["status"] == "cancelled":
            return {**dict(refund), "reused": True, "mock": True}
        if refund["status"] != "pending_review":
            return {"error": "invalid_state", "message": "当前状态无法撤回退款申请"}
        updated = db.execute(
            "UPDATE refund_requests SET status = 'cancelled' WHERE id = ? AND user_id = ? AND status = 'pending_review'",
            (refund_id, user_id),
        )
        ticket = db.execute(
            "UPDATE refund_tickets SET status = 'cancelled' WHERE refund_id = ? AND user_id = ? AND status IN ('pending', 'processing')",
            (refund_id, user_id),
        )
        if updated.rowcount != 1 or ticket.rowcount != 1:
            raise RuntimeError("Refund and ticket state are inconsistent")
        if refund["session_id"]:
            db.execute(
                "INSERT OR IGNORE INTO handoff_outbox (refund_id, session_id, user_id, action, created_at) VALUES (?, ?, ?, 'cancel', ?)",
                (refund_id, refund["session_id"], user_id, _now().isoformat()),
            )
        db.execute("INSERT INTO after_sales_audit (created_at, action, actor_id, refund_id, order_id) VALUES (?, 'refund_cancelled', ?, ?, ?)",
                   (_now().isoformat(), user_id, refund_id, refund["order_id"]))
        db.commit()
        return {**dict(refund), "status": "cancelled", "reused": False, "mock": True}
    finally:
        db.close()


def _enrich_usernames(rows: list[dict], id_field: str, target_field: str) -> list[dict]:
    """售后库与账户库是两个 SQLite 文件，用户名在 Python 层回填。"""
    from auth.database import get_usernames_by_ids

    usernames = get_usernames_by_ids([row.get(id_field, "") for row in rows])
    for row in rows:
        row[target_field] = usernames.get(row.get(id_field, ""))
    return rows


def list_refund_reviews() -> list[dict]:
    """客服工作台：全部退款申请（含会话关联），按时间倒序。"""
    with closing(_connect()) as db:
        rows = db.execute(
            """
            SELECT r.id, r.order_id, r.user_id, r.reason, r.status,
                   r.session_id, r.created_at, t.id AS ticket_id, t.status AS ticket_status,
                   o.status AS order_status, o.amount_cents, o.product
            FROM refund_requests r
            LEFT JOIN refund_tickets t ON t.refund_id = r.id
            LEFT JOIN orders o ON o.id = r.order_id
            ORDER BY r.created_at DESC
            """
        ).fetchall()
    return _enrich_usernames([{**dict(row), "mock": True} for row in rows], "user_id", "username")


def admin_refunds(limit: int, offset: int, status: str = "") -> list[dict]:
    with closing(_connect()) as db:
        rows = db.execute(
            """SELECT r.id, r.order_id, r.user_id, r.status, r.session_id, r.created_at,
                      t.id AS ticket_id, t.status AS ticket_status,
                      (SELECT h.status FROM handoff_outbox h WHERE h.refund_id = r.id ORDER BY h.id DESC LIMIT 1) AS resume_status
               FROM refund_requests r LEFT JOIN refund_tickets t ON t.refund_id = r.id
               WHERE (? = '' OR r.status = ?) ORDER BY r.created_at DESC, r.id DESC LIMIT ? OFFSET ?""",
            (status, status, limit, offset),
        ).fetchall()
    return _enrich_usernames([dict(row) for row in rows], "user_id", "username")


def admin_refund_counts() -> dict:
    with closing(_connect()) as db:
        counts = {row["status"]: row["n"] for row in db.execute(
            "SELECT status, COUNT(*) AS n FROM refund_requests GROUP BY status"
        )}
        waiting_info = db.execute("SELECT COUNT(*) FROM refund_tickets WHERE status = 'processing'").fetchone()[0]
        pending_resume = db.execute("SELECT COUNT(*) FROM handoff_outbox WHERE status != 'done'").fetchone()[0]
    return {"by_status": counts, "waiting_for_customer": waiting_info, "pending_resume": pending_resume}


def admin_business_events(limit: int, offset: int = 0) -> list[dict]:
    with closing(_connect()) as db:
        rows = db.execute(
            """SELECT created_at, action, actor_id, refund_id, order_id FROM (
                   SELECT created_at, action, actor_id, refund_id, order_id FROM after_sales_audit
                   UNION ALL
                   SELECT a.created_at, a.action, a.operator_id AS actor_id, a.refund_id, r.order_id
                   FROM workbench_actions a JOIN refund_requests r ON r.id = a.refund_id
               ) ORDER BY created_at DESC LIMIT ? OFFSET ?""", (limit, offset),
        ).fetchall()
    return [{**dict(row), "kind": "business", "outcome": "success"} for row in rows]


def get_refund_review(refund_id: str) -> dict | None:
    """客服工作台：单个退款申请详情（含订单与关联工单）。"""
    with closing(_connect()) as db:
        row = db.execute(
            """
            SELECT r.id, r.order_id, r.user_id, r.reason, r.status,
                   r.session_id, r.created_at, t.id AS ticket_id, t.status AS ticket_status,
                   o.status AS order_status, o.amount_cents, o.product
            FROM refund_requests r
            LEFT JOIN refund_tickets t ON t.refund_id = r.id
            LEFT JOIN orders o ON o.id = r.order_id
            WHERE r.id = ?
            """,
            (refund_id,),
        ).fetchone()
    if row is None:
        return None
    return _enrich_usernames([{**dict(row), "mock": True}], "user_id", "username")[0]


def list_refund_actions(refund_id: str) -> list[dict]:
    """某个退款申请的人工操作审计记录。"""
    with closing(_connect()) as db:
        rows = db.execute(
            """
            SELECT a.id, a.refund_id, a.operator_id,
                   a.action, a.note, a.created_at
            FROM workbench_actions a
            WHERE a.refund_id = ?
            ORDER BY a.id
            """,
            (refund_id,),
        ).fetchall()
    return _enrich_usernames([dict(row) for row in rows], "operator_id", "operator")


def list_customer_updates(refund_id: str) -> list[dict]:
    with closing(_connect()) as db:
        rows = db.execute(
            "SELECT content, created_at FROM refund_customer_updates WHERE refund_id = ? ORDER BY id",
            (refund_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def provide_refund_information(user_id: str, refund_id: str, content: str) -> dict:
    content = content.strip()
    if not content or len(content) > 1000:
        return {"error": "invalid_content", "message": "请填写不超过 1000 字的补充信息"}
    db = _connect()
    try:
        db.execute("BEGIN IMMEDIATE")
        refund = db.execute(
            "SELECT id, status, order_id FROM refund_requests WHERE id = ? AND user_id = ?",
            (refund_id, user_id),
        ).fetchone()
        if not refund:
            return {"error": "refund_not_found", "message": "退款申请不存在或无权访问"}
        if refund["status"] != "pending_review":
            return {"error": "invalid_state", "message": "当前状态不能补充信息"}
        ticket = db.execute(
            "UPDATE refund_tickets SET status = 'pending' WHERE refund_id = ? AND user_id = ? AND status = 'processing'",
            (refund_id, user_id),
        )
        if ticket.rowcount != 1:
            return {"error": "invalid_state", "message": "当前不需要补充信息"}
        db.execute(
            "INSERT INTO refund_customer_updates (refund_id, user_id, content, created_at) VALUES (?, ?, ?, ?)",
            (refund_id, user_id, content, _now().isoformat()),
        )
        db.execute("INSERT INTO after_sales_audit (created_at, action, actor_id, refund_id, order_id) VALUES (?, 'information_submitted', ?, ?, ?)",
                   (_now().isoformat(), user_id, refund_id, refund["order_id"]))
        db.commit()
        return {"id": refund_id, "status": "pending_review", "ticket_status": "pending"}
    finally:
        db.close()


def claim_handoff(refund_id: str) -> dict | None:
    """Claim a pending resume; an expired lease allows recovery after a crash."""
    db = _connect()
    try:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM handoff_outbox WHERE refund_id = ? ORDER BY id DESC LIMIT 1", (refund_id,)).fetchone()
        if not row:
            return None
        event = dict(row)
        if event["status"] == "done":
            return event
        if event["status"] == "running" and event["claimed_at"]:
            if _now() - datetime.fromisoformat(event["claimed_at"]) < timedelta(minutes=5):
                return {**event, "error": "resume_in_progress"}
        token = _now().isoformat()
        db.execute("UPDATE handoff_outbox SET status = 'running', claimed_at = ? WHERE id = ?", (token, event["id"]))
        db.commit()
        return {**event, "status": "running", "claimed_at": token}
    finally:
        db.close()


def get_handoff(refund_id: str) -> dict | None:
    with closing(_connect()) as db:
        row = db.execute(
            "SELECT id, refund_id, session_id, user_id, action, status, response FROM handoff_outbox WHERE refund_id = ? ORDER BY id DESC LIMIT 1",
            (refund_id,),
        ).fetchone()
    return dict(row) if row else None


def has_pending_handoff(session_id: str, user_id: str) -> bool:
    with closing(_connect()) as db:
        return db.execute(
            "SELECT 1 FROM handoff_outbox WHERE session_id = ? AND user_id = ? AND status != 'done' LIMIT 1",
            (session_id, user_id),
        ).fetchone() is not None


def has_pending_refund(session_id: str, user_id: str) -> bool:
    with closing(_connect()) as db:
        return db.execute(
            "SELECT 1 FROM refund_requests WHERE session_id = ? AND user_id = ? AND status = 'pending_review' LIMIT 1",
            (session_id, user_id),
        ).fetchone() is not None


def finish_handoff(event_id: int, token: str, response: str) -> bool:
    with closing(_connect()) as db, db:
        result = db.execute(
            "UPDATE handoff_outbox SET status = 'done', response = ?, completed_at = ? WHERE id = ? AND status = 'running' AND claimed_at = ?",
            (response, _now().isoformat(), event_id, token),
        )
        return result.rowcount == 1


def release_handoff(event_id: int, token: str) -> None:
    with closing(_connect()) as db, db:
        db.execute(
            "UPDATE handoff_outbox SET status = 'pending', claimed_at = NULL WHERE id = ? AND status = 'running' AND claimed_at = ?",
            (event_id, token),
        )


def handoff_status(refund_id: str) -> str | None:
    with closing(_connect()) as db:
        row = db.execute("SELECT status FROM handoff_outbox WHERE refund_id = ? ORDER BY id DESC LIMIT 1", (refund_id,)).fetchone()
    return row["status"] if row else None


def list_handoff_messages(session_id: str, user_id: str) -> list[dict]:
    with closing(_connect()) as db:
        rows = db.execute(
            "SELECT id, response, COALESCE(completed_at, created_at) AS created_at FROM handoff_outbox WHERE session_id = ? AND user_id = ? AND status = 'done' AND response != '' ORDER BY id",
            (session_id, user_id),
        ).fetchall()
        questions = db.execute(
            "SELECT a.id, a.note, a.created_at FROM workbench_actions a JOIN refund_requests r ON r.id = a.refund_id WHERE r.session_id = ? AND r.user_id = ? AND a.action = 'need_info' ORDER BY a.id",
            (session_id, user_id),
        ).fetchall()
    notices = [{"role": "assistant", "content": row["response"], "timestamp": row["created_at"], "handoff_id": row["id"]} for row in rows]
    notices.extend({"role": "assistant", "content": f"客服需要您补充信息：{row['note']}。请在订单售后页面提交。", "timestamp": row["created_at"], "question_id": row["id"]} for row in questions)
    return notices


def record_refund_decision(refund_id: str, action: str, note: str, operator_id: str, operator_name: str = "") -> dict:
    """人工决定：批准 / 拒绝 / 需补充信息，联动工单状态并写审计。

    事务内完成退款单与工单流转；need_info 保持 pending_review
    （工单转 processing 并记录追问内容），approve/reject 终结两者。
    返回 {error: ...} 或更新后的申请信息。
    """
    transitions = {
        "approve": ("approved", "resolved"),
        "reject": ("rejected", "rejected"),
        "need_info": ("pending_review", "processing"),
    }
    if action not in transitions:
        return {"error": "invalid_action", "message": "不支持的操作类型"}
    if action == "need_info" and not note.strip():
        return {"error": "note_required", "message": "要求补充信息时必须填写追问内容"}
    new_refund_status, new_ticket_status = transitions[action]

    db = _connect()
    try:
        db.execute("BEGIN IMMEDIATE")
        refund = db.execute(
            "SELECT id, order_id, user_id, reason, status, session_id, created_at FROM refund_requests WHERE id = ?",
            (refund_id,),
        ).fetchone()
        if not refund:
            return {"error": "refund_not_found", "message": "退款申请不存在"}
        if refund["status"] != "pending_review":
            last = db.execute(
                "SELECT id, action, note FROM workbench_actions WHERE refund_id = ? ORDER BY id DESC LIMIT 1",
                (refund_id,),
            ).fetchone()
            if last and last["action"] == action and last["note"] == note.strip()[:500]:
                return {**dict(refund), "action_id": last["id"], "reused": True, "mock": True}
            return {"error": "invalid_state", "message": f"当前状态（{refund['status']}）无法审批"}

        ticket_status = db.execute(
            "SELECT status FROM refund_tickets WHERE refund_id = ?", (refund_id,),
        ).fetchone()
        if ticket_status and ticket_status["status"] == "processing" and action == "need_info":
            last = db.execute(
                "SELECT id, operator_id, action, note FROM workbench_actions WHERE refund_id = ? ORDER BY id DESC LIMIT 1",
                (refund_id,),
            ).fetchone()
            if last and last["action"] == action and last["operator_id"] == operator_id and last["note"] == note.strip()[:500]:
                return {**dict(refund), "action_id": last["id"], "reused": True, "mock": True}
        if not ticket_status or ticket_status["status"] != "pending":
            return {"error": "invalid_state", "message": "正在等待用户补充信息或工单状态无效"}

        if action == "need_info":
            ticket = db.execute(
                "UPDATE refund_tickets SET status = 'processing' WHERE refund_id = ? AND status = 'pending'",
                (refund_id,),
            )
            if ticket.rowcount != 1:
                return {"error": "invalid_state", "message": "正在等待用户补充信息"}
            db.execute(
                "INSERT INTO workbench_actions (refund_id, operator_id, action, note, created_at) VALUES (?, ?, ?, ?, ?)",
                (refund_id, operator_id, action, note.strip()[:500], _now().isoformat()),
            )
            db.commit()
            return {**dict(refund), "status": "pending_review", "reused": False, "mock": True}

        updated = db.execute(
            "UPDATE refund_requests SET status = ? WHERE id = ? AND status = 'pending_review'",
            (new_refund_status, refund_id),
        )
        ticket = db.execute(
            "UPDATE refund_tickets SET status = ? WHERE refund_id = ? AND status = 'pending'",
            (new_ticket_status, refund_id),
        )
        if updated.rowcount != 1 or ticket.rowcount != 1:
            raise RuntimeError("Refund and ticket state are inconsistent")
        action_row = db.execute(
            "INSERT INTO workbench_actions (refund_id, operator_id, action, note, created_at) VALUES (?, ?, ?, ?, ?)",
            (refund_id, operator_id, action, note.strip()[:500], _now().isoformat()),
        )
        if refund["session_id"]:
            db.execute(
                "INSERT INTO handoff_outbox (refund_id, action_id, session_id, user_id, action, note, operator, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (refund_id, action_row.lastrowid, refund["session_id"], refund["user_id"], action,
                 note.strip()[:500], operator_name or operator_id, _now().isoformat()),
            )
        db.commit()
        return {**dict(refund), "status": new_refund_status, "action_id": action_row.lastrowid, "reused": False, "mock": True}
    finally:
        db.close()
