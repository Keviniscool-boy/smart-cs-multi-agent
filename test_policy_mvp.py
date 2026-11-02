import asyncio
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from storage import after_sales, after_sales_policy
from memory.long_term import EmbeddingUnavailable, LongTermMemory


@pytest.fixture
def order(tmp_path, monkeypatch):
    monkeypatch.setenv("AFTER_SALES_DB_PATH", str(tmp_path / "after.sqlite3"))
    return next(row for row in after_sales.list_orders("customer") if row["status"] == "delivered")


def test_quality_after_seven_days_and_idempotent_replay(order):
    path = after_sales.os.environ["AFTER_SALES_DB_PATH"]
    with sqlite3.connect(path) as db:
        db.execute("UPDATE orders SET delivered_at = ? WHERE id = ?", ((datetime.now(timezone.utc) - timedelta(days=10)).isoformat(), order["order_id"]))
    result = after_sales.apply_refund("customer", order["order_id"], "fault", "key", application_type="quality_issue")
    assert result["status"] == "pending_review" and result["policy_action"] == "manual_review"
    assert "质量问题" in result["policy_ref"]
    replay = after_sales.apply_refund("customer", order["order_id"], "fault", "key", application_type="quality_issue")
    assert replay["id"] == result["id"] and replay["reused"]
    assert after_sales.apply_refund("customer", order["order_id"], "fault", "key", application_type="no_reason")["error"] == "idempotency_conflict"
    assert after_sales.apply_refund("other", order["order_id"], "fault", "other", application_type="quality_issue")["error"] == "order_not_found"


def test_missing_type_and_unknown_product_go_to_review(order):
    result = after_sales.apply_refund("customer", order["order_id"], "reason", "key")
    assert result["policy_action"] == "manual_review"
    assert result["status"] == "pending_review"
    assert after_sales_policy.evaluate("no_reason", None, datetime.now(timezone.utc), datetime.now(timezone.utc)).action == "manual_review"


def test_policy_unavailable_never_rejects(order, monkeypatch, tmp_path):
    monkeypatch.setattr(after_sales_policy, "POLICY_PATH", tmp_path / "missing.md")
    result = after_sales.apply_refund("customer", order["order_id"], "reason", "key", application_type="no_reason")
    assert result["policy_action"] == "manual_review" and not result["policy_ref"]


def test_web_and_local_mcp_paths_preserve_application_type(order, monkeypatch):
    from fastapi.testclient import TestClient
    from api.main import app
    from auth.security import get_current_user
    from agents.ticket_handler import TicketHandlerAgent

    path = after_sales.os.environ["AFTER_SALES_DB_PATH"]
    with sqlite3.connect(path) as db:
        db.execute("UPDATE orders SET delivered_at = ? WHERE id = ?", ((datetime.now(timezone.utc) - timedelta(days=10)).isoformat(), order["order_id"]))
    app.dependency_overrides[get_current_user] = lambda: {"id": "customer", "username": "customer", "role": "user"}
    try:
        client = TestClient(app)
        response = client.post("/api/refunds", headers={"Idempotency-Key": "api-key"}, json={
            "order_id": order["order_id"], "reason": "fault", "application_type": "quality_issue"})
        assert response.status_code == 200
        assert response.json()["policy_action"] == "manual_review"
        local = TicketHandlerAgent._local_tool("refund_apply", {
            "user_id": "customer", "order_id": order["order_id"], "reason": "fault",
            "idempotency_key": "api-key", "application_type": "quality_issue"})
        assert local["id"] == response.json()["id"] and local["reused"]
    finally:
        app.dependency_overrides.pop(get_current_user, None)


def policy(doc_id, status="published", end="null", version="1.0", body="first version", extra=""):
    return f'---\ndoc_id: {doc_id}\nversion: "{version}"\nstatus: {status}\neffective_date: "2020-01-01"\nexpiration_date: {end}\n{extra}---\n# Policy\n## Clause\n{body}\n'


class Embeddings:
    async def aembed_query(self, text):
        return [1.0, 0.0]


def test_index_filters_versions_and_removes_obsolete_files(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    current = kb / "current.md"
    current.write_text(policy("P1"), encoding="utf-8")
    (kb / "draft.md").write_text(policy("P2", status="draft", body="draft secret"), encoding="utf-8")
    (kb / "expired.md").write_text(policy("P3", end='"2020-12-31"', body="expired"), encoding="utf-8")
    memory = LongTermMemory(index_path=str(tmp_path / "index"))
    memory.embeddings = Embeddings()
    assert asyncio.run(memory.load_knowledge_base(str(kb))) == 1
    assert memory.documents[0]["source"] == "P1 v1.0 §Clause"
    current.write_text(policy("P1", version="2.0", body="second version"), encoding="utf-8")
    assert asyncio.run(memory.load_knowledge_base(str(kb))) == 1
    assert "first version" not in str(memory.documents)
    current.unlink()
    asyncio.run(memory.load_knowledge_base(str(kb)))
    assert memory.documents == []


def test_conflicting_published_versions_fail_closed(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "a.md").write_text(policy("P1"), encoding="utf-8")
    (kb / "b.md").write_text(policy("P1", version="2.0"), encoding="utf-8")
    memory = LongTermMemory(index_path=str(tmp_path / "index"))
    memory.embeddings = Embeddings()
    assert asyncio.run(memory.load_knowledge_base(str(kb))) == 0
    assert memory.index_build_failed
    with pytest.raises(EmbeddingUnavailable):
        asyncio.run(memory.search("test"))


def test_policy_reference_comes_from_published_version(tmp_path, monkeypatch):
    path = tmp_path / "policy.md"
    path.write_text(policy("P1", version="2.0", extra="rule_set: after_sales_mvp_1\n"), encoding="utf-8")
    monkeypatch.setattr(after_sales_policy, "POLICY_PATH", path)
    now = datetime.now(timezone.utc)
    assert after_sales_policy.evaluate("no_reason", "normal", now, now).policy_ref == "P1-v2.0#无理由退货"
    path.write_text(policy("P1", status="draft", extra="rule_set: after_sales_mvp_1\n"), encoding="utf-8")
    assert after_sales_policy.evaluate("no_reason", "normal", now, now).action == "manual_review"


def test_chat_returns_retrieval_evidence_without_model_fabrication(monkeypatch):
    import api.main as api_main

    class Working:
        def update(self, *args, **kwargs):
            pass

    class ShortTerm:
        async def add_message(self, *args, **kwargs):
            pass

    monkeypatch.setattr(api_main, "working_memory", Working())
    monkeypatch.setattr(api_main, "short_term_memory", ShortTerm())
    result = {"final_response": "answer", "intent": "knowledge", "current_agent": "knowledge_rag",
              "compliance_passed": True, "compliance_reason": "", "sub_results": {
                  "knowledge_rag_context": [{"source": "P1 v1 §Clause", "content": "evidence"}]}}
    response = asyncio.run(api_main._finalize_chat("session", "user", "question", result))
    assert "P1 v1 §Clause：evidence" in response.response
    result["compliance_passed"] = False
    rejected = asyncio.run(api_main._finalize_chat("session", "user", "question", result))
    assert "evidence" not in rejected.response
