"""Regression coverage for the demo after-sales workflow."""

import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from auth.security import get_current_user
from storage import after_sales as svc


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    path = tmp_path / "after_sales.sqlite3"
    monkeypatch.setenv("AFTER_SALES_DB_PATH", str(path))
    return path


@pytest.fixture
def order(isolated_db):
    return next(item for item in svc.list_orders("user-a") if item["status"] == "delivered")


@pytest.fixture
def client(isolated_db):
    from api.main import app

    async def current_user(request: Request):
        return {"id": request.headers.get("X-Test-User", "user-a"), "username": "test"}

    app.dependency_overrides[get_current_user] = current_user
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_current_user, None)


def test_demo_orders_are_stable_and_isolated(isolated_db):
    first = svc.list_orders("user-a")
    other = svc.list_orders("user-b")
    assert len(first) == len(other) == 2
    assert {row["order_id"] for row in first}.isdisjoint(row["order_id"] for row in other)
    assert svc.list_orders("user-a") == first
    assert all(row["mock"] for row in first)


def test_missing_and_other_users_orders_have_same_error(order):
    missing = svc.get_order("user-a", "does-not-exist")
    other = svc.get_order("user-b", order["order_id"])
    assert missing == other
    assert missing["error"] == "order_not_found"


def test_create_request_creates_pending_ticket_in_same_database(isolated_db, order):
    result = svc.apply_refund("user-a", order["order_id"], "划痕", "key-1")
    assert result["status"] == "pending_review"
    assert result["reused"] is False
    with sqlite3.connect(isolated_db) as db:
        ticket = db.execute("SELECT refund_id, user_id, status FROM refund_tickets WHERE id = ?", (result["ticket_id"],)).fetchone()
    assert ticket == (result["id"], "user-a", "pending")


def test_same_key_and_payload_reuses_original(order):
    first = svc.apply_refund("user-a", order["order_id"], "划痕", "key-1")
    repeated = svc.apply_refund("user-a", order["order_id"], "划痕", "key-1")
    assert repeated["id"] == first["id"]
    assert repeated["ticket_id"] == first["ticket_id"]
    assert repeated["reused"] is True


@pytest.mark.parametrize("change", ["reason", "order"])
def test_same_key_with_different_content_conflicts(isolated_db, order, change):
    svc.apply_refund("user-a", order["order_id"], "划痕", "key-1")
    cancelled = next(item for item in svc.list_orders("user-a") if item["status"] == "cancelled")
    order_id = cancelled["order_id"] if change == "order" else order["order_id"]
    reason = "不想要了" if change == "reason" else "划痕"
    assert svc.apply_refund("user-a", order_id, reason, "key-1")["error"] == "idempotency_conflict"


def test_new_key_same_request_reuses_and_is_recorded(order):
    first = svc.apply_refund("user-a", order["order_id"], "划痕", "key-1")
    second = svc.apply_refund("user-a", order["order_id"], "划痕", "key-2")
    assert second["id"] == first["id"] and second["reused"]
    svc.cancel_refund("user-a", first["id"])
    replay = svc.apply_refund("user-a", order["order_id"], "划痕", "key-2")
    assert replay["id"] == first["id"] and replay["status"] == "cancelled"


def test_same_order_with_different_reason_conflicts(order):
    svc.apply_refund("user-a", order["order_id"], "划痕", "key-1")
    assert svc.apply_refund("user-a", order["order_id"], "破损", "key-2")["error"] == "refund_exists"


def test_cancelled_order_is_ineligible(isolated_db):
    order = next(item for item in svc.list_orders("user-a") if item["status"] == "cancelled")
    assert svc.apply_refund("user-a", order["order_id"], "退款", "key")["error"] == "not_eligible"


@pytest.mark.parametrize("days,expected", [(8, "reject"), (-1, "manual_review")])
def test_delivery_outside_valid_time(isolated_db, order, days, expected):
    timestamp = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with sqlite3.connect(isolated_db) as db:
        db.execute("UPDATE orders SET delivered_at = ? WHERE id = ?", (timestamp, order["order_id"]))
    result = svc.apply_refund("user-a", order["order_id"], "退款", "key", application_type="no_reason")
    if expected == "reject":
        assert result["error"] == "not_eligible"
    else:
        assert result["policy_action"] == "manual_review"


def test_exact_seven_day_boundary_is_allowed(isolated_db, order, monkeypatch):
    now = datetime(2026, 10, 31, tzinfo=timezone.utc)
    monkeypatch.setattr(svc, "_now", lambda: now)
    with sqlite3.connect(isolated_db) as db:
        db.execute("UPDATE orders SET delivered_at = ? WHERE id = ?", ((now - timedelta(days=7)).isoformat(), order["order_id"]))
    assert svc.apply_refund("user-a", order["order_id"], "退款", "key")["status"] == "pending_review"


def test_cancel_updates_ticket_and_repeat_is_idempotent(isolated_db, order):
    created = svc.apply_refund("user-a", order["order_id"], "划痕", "key-1")
    cancelled = svc.cancel_refund("user-a", created["id"])
    assert cancelled["status"] == "cancelled" and cancelled["reused"] is False
    with sqlite3.connect(isolated_db) as db:
        ticket_status = db.execute("SELECT status FROM refund_tickets WHERE id = ?", (created["ticket_id"],)).fetchone()
    assert ticket_status == ("cancelled",)
    assert svc.cancel_refund("user-a", created["id"])["reused"] is True
    assert svc.cancel_refund("user-b", created["id"])["error"] == "refund_not_found"
    old = svc.apply_refund("user-a", order["order_id"], "划痕", "key-1")
    assert old["id"] == created["id"] and old["status"] == "cancelled"
    new = svc.apply_refund("user-a", order["order_id"], "划痕", "key-2")
    assert new["id"] != created["id"] and new["status"] == "pending_review"


def test_cancel_rolls_back_when_ticket_state_is_inconsistent(isolated_db, order):
    created = svc.apply_refund("user-a", order["order_id"], "划痕", "key-1")
    with sqlite3.connect(isolated_db) as db:
        db.execute("UPDATE refund_tickets SET status = 'resolved' WHERE id = ?", (created["ticket_id"],))
    with pytest.raises(RuntimeError, match="inconsistent"):
        svc.cancel_refund("user-a", created["id"])
    assert svc.list_refunds("user-a")[0]["status"] == "pending_review"


def test_approved_refund_cannot_be_cancelled(isolated_db, order):
    created = svc.apply_refund("user-a", order["order_id"], "划痕", "key-1")
    with sqlite3.connect(isolated_db) as db:
        db.execute("UPDATE refund_requests SET status = 'approved' WHERE id = ?", (created["id"],))
    assert svc.cancel_refund("user-a", created["id"])["error"] == "invalid_state"


@pytest.mark.parametrize("reason,key,error", [
    ("", "key", "invalid_reason"),
    (" ", "key", "invalid_reason"),
    ("x" * 501, "key", "invalid_reason"),
    ("reason", "", "invalid_idempotency_key"),
    ("reason", " " , "invalid_idempotency_key"),
    ("reason", "k" * 129, "invalid_idempotency_key"),
])
def test_invalid_inputs(order, reason, key, error):
    assert svc.apply_refund("user-a", order["order_id"], reason, key)["error"] == error


def test_idempotency_keys_are_scoped_to_user(isolated_db, order):
    other = next(item for item in svc.list_orders("user-b") if item["status"] == "delivered")
    first = svc.apply_refund("user-a", order["order_id"], "划痕", "same-key")
    second = svc.apply_refund("user-b", other["order_id"], "划痕", "same-key")
    assert first["id"] != second["id"]
    assert second["status"] == "pending_review"


def test_concurrent_requests_create_only_one_refund_and_ticket(isolated_db, order):
    def request(index):
        return svc.apply_refund("user-a", order["order_id"], "划痕", f"key-{index}")

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(request, range(8)))
    assert len({item["id"] for item in results}) == 1
    assert sum(item["reused"] is False for item in results) == 1
    with sqlite3.connect(isolated_db) as db:
        assert db.execute("SELECT count(*) FROM refund_requests").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM refund_tickets").fetchone()[0] == 1


def test_api_enforces_ownership_and_idempotency(client):
    order = next(item for item in client.get("/api/orders").json()["orders"] if item["status"] == "delivered")
    order_id = order["order_id"]
    assert client.get(f"/api/orders/{order_id}", headers={"X-Test-User": "user-b"}).status_code == 404
    assert client.post("/api/refunds", json={"order_id": order_id, "reason": "划痕"}).status_code == 400
    headers = {"Idempotency-Key": "key-1"}
    payload = {"order_id": order_id, "reason": "划痕"}
    unauthorized = client.post("/api/refunds", json=payload, headers={**headers, "X-Test-User": "user-b"})
    assert unauthorized.status_code == 404
    first = client.post("/api/refunds", json=payload, headers=headers)
    assert first.status_code == 200 and first.json()["status"] == "pending_review"
    assert client.post("/api/refunds", json=payload, headers=headers).json()["id"] == first.json()["id"]
    conflict = client.post("/api/refunds", json={**payload, "reason": "破损"}, headers=headers)
    assert conflict.status_code == 409
    assert client.post(f"/api/refunds/{first.json()['id']}/cancel", headers={"X-Test-User": "user-b"}).status_code == 404
    assert client.post(f"/api/refunds/{first.json()['id']}/cancel").json()["status"] == "cancelled"


def test_api_requires_authentication(isolated_db):
    from api.main import app

    unauthenticated = TestClient(app)
    assert unauthenticated.get("/api/orders").status_code == 401
    assert unauthenticated.get("/api/refunds").status_code == 401
    assert unauthenticated.post("/api/refunds", json={"order_id": "missing", "reason": "test"}).status_code == 401


def test_mcp_order_tool_uses_authenticated_user(client):
    order = next(item for item in client.get("/api/orders").json()["orders"] if item["status"] == "delivered")
    response = client.post("/api/tools/call", headers={"X-Test-User": "user-b"}, json={
        "name": "order_query", "arguments": {"order_id": order["order_id"], "user_id": "user-a"},
    })
    assert response.status_code == 200
    assert "order_not_found" in response.text


def test_mcp_and_local_fallback_share_refund_idempotency(order):
    from agents.ticket_handler import TicketHandlerAgent
    from mcp_tools.order_server import refund_apply

    args = {"user_id": "user-a", "order_id": order["order_id"], "reason": "划痕", "idempotency_key": "key"}
    first = asyncio.run(refund_apply(**args))
    retry = TicketHandlerAgent._local_tool("refund_apply", args)
    assert retry["id"] == first["id"] and retry["reused"]

