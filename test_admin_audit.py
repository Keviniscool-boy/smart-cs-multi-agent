import asyncio
import sqlite3

import pytest
from fastapi.testclient import TestClient

from auth import database
from auth.security import get_current_user
from storage import after_sales
from tracing.admin_audit import list_events, list_runs, trace_scope
from tracing.otel_config import trace_agent_call


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "auth.sqlite")
    monkeypatch.setenv("AFTER_SALES_DB_PATH", str(tmp_path / "after.sqlite"))
    database.init_database()
    admin = database.create_user("admin", "password")
    database.set_user_role("admin", "admin")
    user = database.create_user("user", "password")
    return admin, user


def test_role_change_audit_and_last_admin(isolated):
    admin, user = isolated
    database.set_user_role("user", "agent", actor_id=admin["id"])
    database.set_user_role("user", "agent", actor_id=admin["id"])
    events = list_events(20, kind="role")
    assert len(events) == 2
    assert events[0]["actor_id"] == admin["id"] and events[0]["target_id"] == user["id"]
    with pytest.raises(ValueError, match="last_admin"):
        database.set_user_role("admin", "user", actor_id=admin["id"])
    assert database.get_user_by_username("admin")["role"] == "admin"
    assert all("password_hash" not in row for row in database.list_users())


def test_role_change_rolls_back_when_audit_fails(isolated):
    with database._connection() as db:
        db.execute("DROP TABLE admin_events")
    with pytest.raises(sqlite3.OperationalError):
        database.set_user_role("user", "agent")
    assert database.get_user_by_username("user")["role"] == "user"


def test_refund_audit_idempotency_and_privacy(isolated):
    order = next(o for o in after_sales.list_orders("customer") if o["status"] == "delivered")
    first = after_sales.apply_refund("customer", order["order_id"], "private reason", "key")
    after_sales.apply_refund("customer", order["order_id"], "private reason", "key")
    assert after_sales.cancel_refund("other", first["id"])["error"]
    after_sales.cancel_refund("customer", first["id"])
    events = after_sales.admin_business_events(20)
    assert [e["action"] for e in events] == ["refund_cancelled", "refund_requested"]
    assert "private reason" not in str(events)


def test_runtime_trace_persists_only_metadata(isolated):
    @trace_agent_call("test_node")
    async def node():
        return {"compliance_passed": False, "compliance_reason": "private content"}
    with trace_scope("trace-1", "actor", "session"):
        asyncio.run(node())
    events = list_events(10, trace_id="trace-1")
    assert events[0]["detail"] == "rejected"
    assert "private content" not in str(events)
    assert list_runs(10)[0]["trace_id"] == "trace-1"


def test_chat_and_stream_create_trace_records(isolated, monkeypatch):
    import api.main as api_main

    @trace_agent_call("compliance_process")
    async def node():
        return {"compliance_passed": True, "sub_results": {"compliance": {"risk_level": "low"}}}

    class Graph:
        async def ainvoke(self, state, config):
            return await node()

        async def astream(self, state, config, stream_mode):
            yield {"compliance_checker": await node()}
            yield {"synthesize": {"final_response": "ok"}}

    async def prepare(request, user):
        return request.session_id, {}, {}

    async def finalize(*args):
        return api_main.ChatResponse(session_id=args[0], response="ok", intent="test", current_agent="fallback", compliance_passed=True, compliance_reason="")

    monkeypatch.setattr(api_main, "graph", Graph())
    monkeypatch.setattr(api_main, "_prepare_chat", prepare)
    monkeypatch.setattr(api_main, "_finalize_chat", finalize)
    monkeypatch.setattr(api_main, "claim_session_turn", lambda *args: True)
    monkeypatch.setattr(api_main, "release_session_turn", lambda *args: None)
    api_main.app.dependency_overrides[get_current_user] = lambda: {"id": "actor", "username": "user", "role": "user"}
    try:
        client = TestClient(api_main.app)
        assert client.post("/api/chat", json={"message": "secret", "session_id": "sync"}).status_code == 200
        response = client.post("/api/chat/stream", json={"message": "secret", "session_id": "stream"})
        assert response.status_code == 200 and '"event": "final"' in response.text
        events = list_events(10, kind="compliance")
        assert {e["session_id"] for e in events} == {"sync", "stream"}
        assert all(e["detail"] == "passed;risk=low" for e in events)
        assert "secret" not in str(events)
    finally:
        api_main.app.dependency_overrides.pop(get_current_user, None)


@pytest.mark.parametrize("role,expected", [("user", 403), ("agent", 403), ("admin", 200)])
def test_admin_routes_require_admin(isolated, role, expected):
    from api.main import app
    app.dependency_overrides[get_current_user] = lambda: {"id": isolated[0]["id"], "username": "admin", "role": role}
    try:
        client = TestClient(app)
        for path in ("users", "overview", "refunds", "runs", "runs/trace-1", "audit"):
            assert client.get(f"/api/admin/{path}").status_code == expected
        assert client.post("/api/admin/users/role", json={"username": "user", "role": "agent"}).status_code == expected
        if role == "admin":
            assert client.get("/api/admin/users?limit=101").status_code == 422
            assert client.post("/api/admin/users/role", json={"username": "admin", "role": "user"}).status_code == 409
    finally:
        app.dependency_overrides.pop(get_current_user, None)
