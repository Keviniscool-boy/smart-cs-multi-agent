"""阶段二人工接管回路的图级与 API 级测试。

覆盖：interrupt 暂停、Command 恢复、重启恢复边界、
状态机防重复审批、角色门控、审计记录与幂等保护。
图级测试用可编排假响应的 FakeLLM 驱动完整 LangGraph；
API 级测试用 dependency_overrides 注入带角色的调用方。
"""

import asyncio
import json
import uuid

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

import api.main as api_main
from agents.supervisor import create_supervisor_graph
from auth.security import get_current_user
from memory.working_memory import WorkingMemory
from storage import after_sales as svc


def _compliance_reply(passed: bool) -> str:
    return json.dumps(
        {
            "passed": passed,
            "risk_level": "low" if passed else "medium",
            "violations": [] if passed else ["fake violation"],
            "suggestions": [],
        },
        ensure_ascii=False,
    )


class FakeLLM:
    """严格按调用顺序回放响应：工具调用 → 小结 → 合规 → 合成。

    响应必须用 AIMessage：ticket_handler 会把第一轮响应塞回
    messages 再喂给 LLM，非 BaseMessage 对象会炸序列化。
    """

    def __init__(self, responses: list[AIMessage]):
        self.responses = list(responses)
        self.calls = 0

    def bind_tools(self, tools, **kwargs):
        return self

    async def ainvoke(self, messages, **kwargs):
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    import auth.database as auth_database
    import memory.working_memory as working_memory_module
    from storage.mysql_store import MySQLUnavailableError

    monkeypatch.setenv("AFTER_SALES_DB_PATH", str(tmp_path / "after_sales.sqlite3"))
    monkeypatch.setenv("REDIS_URL", "")
    monkeypatch.setattr(auth_database, "DATABASE_PATH", tmp_path / "auth.sqlite3")
    auth_database.init_database()

    class OfflineMySQL:
        def __getattr__(self, name):
            def unavailable(*args, **kwargs):
                raise MySQLUnavailableError("isolated test")
            return unavailable

    monkeypatch.setattr(working_memory_module, "mysql_store", OfflineMySQL())
    return tmp_path / "after_sales.sqlite3"


def _order_for(user: str = "user-a") -> dict:
    return next(
        item for item in svc.list_orders(user) if item["status"] == "delivered"
    )


def _refund_llm(order_id: str, final_reply: str = "客服已批准您的退款申请，感谢耐心等待。") -> FakeLLM:
    return FakeLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "refund_apply",
                        "args": {"order_id": order_id, "reason": "划痕"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="已为您提交退款申请，等待人工审核。"),
            AIMessage(content=_compliance_reply(True)),
            AIMessage(content=final_reply),
        ]
    )


def _initial_state(session_id: str) -> dict:
    return {
        "messages": [HumanMessage(content="申请退款")],
        "user_id": "user-a",
        "session_id": session_id,
        "intent": "",
        "current_agent": "",
        "final_response": "",
        "compliance_passed": False,
        "compliance_reason": "",
        "compliance_retries": 0,
        "sub_results": {},
        "pending_handoff": {},
        "handoff_decision": {},
    }


def _pause_graph(order_id: str, session_id: str | None = None, checkpointer=None):
    """驱动图走到 human_review 并暂停，返回 (graph, config, session_id)。"""
    session_id = session_id or f"s-{uuid.uuid4().hex[:10]}"
    from auth.database import save_session_owner
    save_session_owner(session_id, "user-a")
    graph = create_supervisor_graph(
        _refund_llm(order_id), None, WorkingMemory(redis_url=None), None,
        checkpointer=checkpointer,
    )
    config = {"configurable": {"thread_id": session_id}}
    result = asyncio.run(graph.ainvoke(_initial_state(session_id), config=config))
    assert result.get("__interrupt__"), "创建待审核退款单后应暂停等待人工"
    return graph, config, session_id


def test_refund_request_pauses_graph_for_human_review(isolated_db):
    """Agent 创建待审核退款单 → 合规通过 → 图暂停在 human_review。"""
    order = _order_for()
    graph, config, session_id = _pause_graph(order["order_id"])

    state = asyncio.run(graph.aget_state(config))
    assert state.next == ("human_review",)
    assert state.values["pending_handoff"]["order_id"] == order["order_id"]
    assert state.values["pending_handoff"]["refund_id"]

    refund = svc.list_refunds("user-a")[0]
    assert refund["status"] == "pending_review"
    assert refund["session_id"] == session_id, "退款单应记录发起会话，供恢复定位线程"


def test_refund_reaches_interrupt_without_another_model_call(isolated_db):
    order = _order_for()
    llm = _refund_llm(order["order_id"])
    original = llm.ainvoke

    async def fail_after_first(messages, **kwargs):
        if llm.calls:
            raise RuntimeError("model unavailable after refund creation")
        return await original(messages, **kwargs)

    llm.ainvoke = fail_after_first
    graph = create_supervisor_graph(llm, None, WorkingMemory(redis_url=None), None)
    config = {"configurable": {"thread_id": "no-second-model"}}
    result = asyncio.run(graph.ainvoke(_initial_state("no-second-model"), config=config))
    assert result.get("__interrupt__")
    assert llm.calls == 1


def test_approve_resumes_graph_and_finalizes_state(isolated_db):
    """批准：状态机落库 → Command 恢复 → 合成答复反映决定。"""
    order = _order_for()
    graph, config, session_id = _pause_graph(order["order_id"])
    refund = svc.list_refunds("user-a")[0]

    assert svc.record_refund_decision(refund["id"], "approve", "包装完好", "ops-1")["status"] == "approved"
    result = asyncio.run(
        graph.ainvoke(
            Command(resume={"action": "approve", "note": "包装完好", "operator": "ops"}),
            config=config,
        )
    )
    assert "批准" in result["final_response"]
    assert result["handoff_decision"]["action"] == "approve"
    assert svc.list_refunds("user-a")[0]["status"] == "approved"
    actions = svc.list_refund_actions(refund["id"])
    assert len(actions) == 1 and actions[0]["action"] == "approve" and actions[0]["operator_id"] == "ops-1"


def test_reject_decision_produces_rejection(isolated_db):
    """拒绝：退款单与工单同转 rejected，恢复后答复告知拒绝。"""
    order = _order_for()
    graph = create_supervisor_graph(
        _refund_llm(order["order_id"], final_reply="很抱歉，您的退款申请未通过审核。"),
        None, WorkingMemory(redis_url=None), None,
    )
    config = {"configurable": {"thread_id": "s-reject"}}
    asyncio.run(graph.ainvoke(_initial_state("s-reject"), config=config))
    refund = svc.list_refunds("user-a")[0]

    svc.record_refund_decision(refund["id"], "reject", "超过演示规则", "ops-1")
    result = asyncio.run(
        graph.ainvoke(Command(resume={"action": "reject", "note": "超过演示规则", "operator": "ops"}), config=config)
    )
    assert "拒绝" in result["final_response"] or "未通过" in result["final_response"]
    assert svc.list_refunds("user-a")[0]["status"] == "rejected"


def test_need_info_keeps_refund_pending_and_ticket_processing(isolated_db):
    """需补充信息：申请保持 pending_review，工单转 processing。"""
    order = _order_for()
    graph = create_supervisor_graph(
        _refund_llm(order["order_id"], final_reply="请补充商品照片和包装状态。"),
        None, WorkingMemory(redis_url=None), None,
    )
    config = {"configurable": {"thread_id": "s-info"}}
    asyncio.run(graph.ainvoke(_initial_state("s-info"), config=config))
    refund = svc.list_refunds("user-a")[0]

    result = svc.record_refund_decision(refund["id"], "need_info", "请提供商品照片", "ops-1")
    assert result["status"] == "pending_review"
    resumed = asyncio.run(
        graph.ainvoke(Command(resume={"action": "need_info", "note": "请提供商品照片", "operator": "ops"}), config=config)
    )
    assert "补充" in resumed["final_response"]
    with svc._connect() as db:
        ticket_status = db.execute(
            "SELECT status FROM refund_tickets WHERE refund_id = ?", (refund["id"],)
        ).fetchone()[0]
    assert ticket_status == "processing"
    assert svc.list_refunds("user-a")[0]["status"] == "pending_review"


def test_duplicate_decision_is_rejected_by_state_machine(isolated_db):
    """防重复审批：第二次决定被状态机以 invalid_state 拒绝。"""
    order = _order_for()
    graph, config, session_id = _pause_graph(order["order_id"])
    refund = svc.list_refunds("user-a")[0]

    assert svc.record_refund_decision(refund["id"], "approve", "", "ops-1")["status"] == "approved"
    second = svc.record_refund_decision(refund["id"], "reject", "", "ops-1")
    assert second["error"] == "invalid_state"
    # 注意：图的暂停点仍在（未调用 resume）；防重复恢复由 API 层保证——
    # 第二次决定在 _resume_handoff 之前就被状态机以 409 拒绝。


def test_restart_recovery_resumes_from_persistent_checkpoint(isolated_db, tmp_path):
    """重启恢复边界：服务重启后同一 checkpoint 文件上的暂停点仍可恢复。"""
    order = _order_for()
    checkpoint = tmp_path / "checkpoints.sqlite3"

    async def scenario():
        import aiosqlite

        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        conn1 = await aiosqlite.connect(str(checkpoint))
        graph1 = create_supervisor_graph(
            _refund_llm(order["order_id"]), None, WorkingMemory(redis_url=None), None,
            checkpointer=AsyncSqliteSaver(conn1),
        )
        config = {"configurable": {"thread_id": "restart-1"}}
        result = await graph1.ainvoke(_initial_state("restart-1"), config=config)
        assert result.get("__interrupt__")
        await conn1.close()

        # 模拟重启：全新的图实例 + 同一份 checkpoint 文件。
        # 新实例的 LLM 调用序列从零开始，恢复后只剩合成一次调用，
        # 所以给它一个只含合成响应的独立队列。
        conn2 = await aiosqlite.connect(str(checkpoint))
        graph2 = create_supervisor_graph(
            FakeLLM([AIMessage(content="客服已批准您的退款申请，感谢耐心等待。")]),
            None, WorkingMemory(redis_url=None), None,
            checkpointer=AsyncSqliteSaver(conn2),
        )
        state = await graph2.aget_state(config)
        assert any(task.interrupts for task in state.tasks), "重启后暂停点应仍在检查点中"

        refund = svc.list_refunds("user-a")[0]
        svc.record_refund_decision(refund["id"], "approve", "ok", "ops-1")
        resumed = await graph2.ainvoke(
            Command(resume={"action": "approve", "note": "ok", "operator": "ops"}), config=config
        )
        assert "批准" in resumed["final_response"]
        await conn2.close()

    asyncio.run(scenario())


# ---------- API 级 ----------


@pytest.fixture
def api(isolated_db, monkeypatch):
    from api.main import app

    from auth.database import init_database

    # 使用隔离的账户库和会话记录，避免写入运行中的服务。
    init_database()

    class FakeHistory:
        def __init__(self):
            self.messages = {}

        async def add_message(self, session_id, role, content, user_id=""):
            self.messages.setdefault(session_id, []).append({"role": role, "content": content})

        async def get_history(self, session_id, last_n=None):
            return self.messages.get(session_id, [])

    monkeypatch.setattr(api_main, "short_term_memory", FakeHistory())
    monkeypatch.setattr(api_main, "working_memory", WorkingMemory(redis_url=""))

    def fake_user(request: Request):
        return {
            "id": request.headers.get("X-Test-User", "user-a"),
            "username": request.headers.get("X-Test-Name", "tester"),
            "role": request.headers.get("X-Test-Role", "user"),
        }

    app.dependency_overrides[get_current_user] = fake_user
    yield app
    app.dependency_overrides.pop(get_current_user, None)


def test_workbench_requires_agent_or_admin_role(api):
    """角色门控：普通用户访问工作台/授权接口一律 403。"""
    client = TestClient(api)
    assert client.get("/api/workbench/refunds", headers={"X-Test-Role": "user"}).status_code == 403
    assert client.get("/api/workbench/refunds", headers={"X-Test-Role": "agent"}).status_code == 200
    assert client.get("/api/workbench/refunds", headers={"X-Test-Role": "admin"}).status_code == 200
    assert client.post(
        "/api/admin/users/role", headers={"X-Test-Role": "agent"},
        json={"username": "nobody", "role": "admin"},
    ).status_code == 403
    assert client.post(
        "/api/admin/users/role", headers={"X-Test-Role": "admin"},
        json={"username": "no-such-user-xyz", "role": "agent"},
    ).status_code == 404
    assert client.post(
        "/api/admin/users/role", headers={"X-Test-Role": "admin"},
        json={"username": "nobody", "role": "superuser"},
    ).status_code == 400


def test_workbench_detail_exposes_handoff_payload(api):
    """详情页包含暂停线程的交接单与会话历史；未暂停时标记 thread_paused=False。"""
    order = _order_for()
    graph, config, session_id = _pause_graph(order["order_id"])
    api_main.graph = graph
    refund = svc.list_refunds("user-a")[0]
    try:
        client = TestClient(api)
        body = client.get(
            f"/api/workbench/refunds/{refund['id']}", headers={"X-Test-Role": "agent"}
        ).json()
        assert body["thread_paused"] is True
        assert body["handoff"]["type"] == "refund_review"
        assert body["handoff"]["refund_id"] == refund["id"]
        assert body["refund"]["reason"] == "划痕"
    finally:
        api_main.graph = None


def test_decision_endpoint_updates_state_and_resumes_thread(api, monkeypatch):
    """决定接口：状态机落库 → 恢复线程 → 答复写入会话历史；重复决定 409。"""
    order = _order_for()
    graph, config, session_id = _pause_graph(order["order_id"])
    monkeypatch.setattr(api_main, "graph", graph)
    refund = svc.list_refunds("user-a")[0]
    try:
        client = TestClient(api)
        headers = {"X-Test-Role": "agent", "X-Test-User": "user-a", "X-Test-Name": "ops"}
        response = client.post(
            f"/api/workbench/refunds/{refund['id']}/decision",
            headers=headers, json={"action": "approve", "note": "包装完好"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["resumed"] is True
        assert "批准" in body["agent_response"]
        assert body["refund"]["status"] == "approved"
        assert body["actions"][0]["action"] == "approve"

        # 恢复产生的最终答复进入客户会话历史。
        history = asyncio.run(api_main._session_history(session_id, "user-a"))
        assert any("批准" in message["content"] for message in history if message["role"] == "assistant")

        # 重复决定：状态机拒绝，不再触发恢复。
        duplicate = client.post(
            f"/api/workbench/refunds/{refund['id']}/decision",
            headers=headers, json={"action": "reject", "note": ""},
        )
        assert duplicate.status_code == 409
        assert duplicate.json()["detail"]["error"] == "invalid_state"
    finally:
        api_main.graph = None


def test_decision_on_rest_created_refund_updates_without_resume(api):
    """面板直建的退款单没有图线程：决定只更新状态，resumed=False。"""
    order = _order_for()
    svc.apply_refund("user-a", order["order_id"], "划痕", "rest-key")
    refund = svc.list_refunds("user-a")[0]
    client = TestClient(api)
    response = client.post(
        f"/api/workbench/refunds/{refund['id']}/decision",
        headers={"X-Test-Role": "agent"}, json={"action": "approve", "note": "ok"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["resumed"] is False
    assert body["refund"]["status"] == "approved"


def test_need_info_decision_requires_note(api):
    """要求补充信息时必须填写追问内容，否则 400。"""
    order = _order_for()
    svc.apply_refund("user-a", order["order_id"], "划痕", "info-key")
    refund = svc.list_refunds("user-a")[0]
    client = TestClient(api)
    response = client.post(
        f"/api/workbench/refunds/{refund['id']}/decision",
        headers={"X-Test-Role": "agent"}, json={"action": "need_info", "note": "  "},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "note_required"


def test_need_info_then_user_can_cancel(isolated_db):
    order = _order_for()
    refund = svc.apply_refund("user-a", order["order_id"], "划痕", "cancel-after-info")
    assert svc.record_refund_decision(refund["id"], "need_info", "请补充信息", "operator")["status"] == "pending_review"
    assert svc.cancel_refund("user-a", refund["id"])["status"] == "cancelled"


def test_first_registered_user_is_not_automatically_admin(isolated_db):
    from auth.database import create_user, init_database

    init_database()
    assert create_user("first-user", "secure-password")["role"] == "user"


def test_admin_bootstrap_requires_explicit_local_command(isolated_db, monkeypatch):
    import subprocess
    import sys
    import auth.database as database

    database.init_database()
    database.create_user("local-operator", "secure-password")
    monkeypatch.setenv("AUTH_DATABASE_PATH", str(database.DATABASE_PATH))
    result = subprocess.run(
        [sys.executable, "-m", "auth.bootstrap_admin", "local-operator"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert database.get_user_by_username("local-operator")["role"] == "admin"


def test_untrusted_tool_cannot_attach_foreign_session(api):
    client = TestClient(api)
    response = client.post("/api/tools/call", headers={"X-Test-User": "user-b"}, json={
        "name": "refund_apply",
        "arguments": {"order_id": "anything", "reason": "reason", "idempotency_key": "key", "session_id": "another-users-session"},
    })
    assert response.status_code == 400


def test_rest_request_need_info_and_review_can_complete(api):
    client = TestClient(api)
    order = _order_for()
    refund = svc.apply_refund("user-a", order["order_id"], "reason", "rest-info")
    headers = {"X-Test-Role": "agent"}
    decision_url = f"/api/workbench/refunds/{refund['id']}/decision"
    first = client.post(decision_url, headers=headers, json={"action": "need_info", "note": "Please describe packaging"})
    assert first.status_code == 200
    assert svc.list_refunds("user-a")[0]["ticket_status"] == "processing"
    other = client.post(f"/api/refunds/{refund['id']}/information", headers={"X-Test-User": "user-b"}, json={"content": "sealed"})
    assert other.status_code == 404
    answer = client.post(f"/api/refunds/{refund['id']}/information", json={"content": "sealed"})
    assert answer.status_code == 200
    detail = client.get(f"/api/workbench/refunds/{refund['id']}", headers=headers).json()
    assert detail["customer_updates"][0]["content"] == "sealed"
    assert client.post(decision_url, headers=headers, json={"action": "approve", "note": "ok"}).json()["refund"]["status"] == "approved"


def test_resume_failure_is_retryable_and_notification_is_not_duplicated(api, monkeypatch):
    order = _order_for()
    real_graph, config, session_id = _pause_graph(order["order_id"])
    refund = svc.list_refunds("user-a")[0]

    class FailOnce:
        def __init__(self):
            self.failed = False

        async def aget_state(self, config):
            return await real_graph.aget_state(config)

        async def ainvoke(self, *args, **kwargs):
            if not self.failed:
                self.failed = True
                raise RuntimeError("temporary failure")
            return await real_graph.ainvoke(*args, **kwargs)

    monkeypatch.setattr(api_main, "graph", FailOnce())
    client = TestClient(api)
    headers = {"X-Test-Role": "agent"}
    decision = client.post(f"/api/workbench/refunds/{refund['id']}/decision", headers=headers, json={"action": "approve", "note": "ok"})
    assert decision.status_code == 200 and decision.json()["resume_pending"] is True
    assert svc.get_handoff(refund["id"])["status"] == "pending"
    url = f"/api/workbench/refunds/{refund['id']}/resume"
    assert client.post(url, headers=headers).json()["resumed"] is True
    assert client.post(url, headers=headers).json()["resumed"] is True
    messages = asyncio.run(api_main._session_history(session_id, "user-a"))
    assert sum("批准" in message["content"] for message in messages) == 1
    assert len(svc.list_refund_actions(refund["id"])) == 1


def test_cancel_resumes_the_matching_paused_session(api, monkeypatch):
    order = _order_for()
    graph, config, session_id = _pause_graph(order["order_id"])
    monkeypatch.setattr(api_main, "graph", graph)
    refund = svc.list_refunds("user-a")[0]
    client = TestClient(api)
    response = client.post(f"/api/refunds/{refund['id']}/cancel")
    assert response.status_code == 200 and response.json()["status"] == "cancelled"
    assert response.json()["resume_pending"] is False
    assert not asyncio.run(graph.aget_state(config)).next
    assert any("撤回" in message["content"] for message in asyncio.run(api_main._session_history(session_id, "user-a")))


def test_chat_does_not_replace_pending_human_checkpoint(api, monkeypatch):
    order = _order_for()
    graph, config, session_id = _pause_graph(order["order_id"])
    monkeypatch.setattr(api_main, "graph", graph)
    response = TestClient(api).post("/api/chat", json={"session_id": session_id, "message": "hello"})
    assert response.status_code == 409
    assert asyncio.run(graph.aget_state(config)).next == ("human_review",)


def test_reusing_rest_refund_does_not_pause_an_unlinked_thread(isolated_db):
    order = _order_for()
    svc.apply_refund("user-a", order["order_id"], "划痕", "rest-first")
    graph = create_supervisor_graph(_refund_llm(order["order_id"]), None, WorkingMemory(redis_url=None), None)
    config = {"configurable": {"thread_id": "unlinked-session"}}
    result = asyncio.run(graph.ainvoke(_initial_state("unlinked-session"), config=config))
    assert not result.get("__interrupt__")
    assert svc.list_refunds("user-a")[0]["session_id"] == ""


def test_human_question_user_reply_then_approve_resumes_original_graph(api, monkeypatch):
    order = _order_for()
    graph, config, session_id = _pause_graph(order["order_id"])
    monkeypatch.setattr(api_main, "graph", graph)
    refund = svc.list_refunds("user-a")[0]
    client = TestClient(api)
    staff = {"X-Test-Role": "agent"}
    decision_url = f"/api/workbench/refunds/{refund['id']}/decision"
    ask = client.post(decision_url, headers=staff, json={"action": "need_info", "note": "Packaging?"})
    assert ask.status_code == 200 and ask.json()["resumed"] is False
    duplicate = client.post(decision_url, headers=staff, json={"action": "need_info", "note": "Packaging?"})
    assert duplicate.status_code == 200 and duplicate.json()["refund"]["reused"] is True
    assert len(svc.list_refund_actions(refund["id"])) == 1
    assert asyncio.run(graph.aget_state(config)).next == ("human_review",)
    assert any("Packaging?" in item["content"] for item in asyncio.run(api_main._session_history(session_id, "user-a")))
    reply = client.post(f"/api/refunds/{refund['id']}/information", json={"content": "Unopened"})
    assert reply.status_code == 200
    approved = client.post(decision_url, headers=staff, json={"action": "approve", "note": "ok"})
    assert approved.status_code == 200 and approved.json()["resumed"] is True
    assert not asyncio.run(graph.aget_state(config)).next


def test_foreign_legacy_session_cannot_be_read_or_approved(api):
    from auth.database import save_session_owner

    save_session_owner("other-session", "user-b")
    order = _order_for()
    refund = svc.apply_refund("user-a", order["order_id"], "reason", "key", "other-session")
    client = TestClient(api)
    headers = {"X-Test-Role": "agent"}
    assert client.get(f"/api/workbench/refunds/{refund['id']}", headers=headers).status_code == 409
    decision = client.post(f"/api/workbench/refunds/{refund['id']}/decision", headers=headers, json={"action": "approve"})
    assert decision.status_code == 409
    assert svc.list_refunds("user-a")[0]["status"] == "pending_review"


def test_concurrent_turn_and_resume_claims_are_exclusive(isolated_db):
    assert svc.claim_session_turn("session", "one")
    assert not svc.claim_session_turn("session", "two")
    svc.release_session_turn("session", "wrong-token")
    assert not svc.claim_session_turn("session", "two")
    svc.release_session_turn("session", "one")
    assert svc.claim_session_turn("session", "two")

    order = _order_for()
    refund = svc.apply_refund("user-a", order["order_id"], "reason", "key", "session")
    svc.record_refund_decision(refund["id"], "approve", "", "operator")
    first = svc.claim_handoff(refund["id"])
    assert svc.claim_handoff(refund["id"])["error"] == "resume_in_progress"
    svc.release_handoff(first["id"], first["claimed_at"])
    assert svc.claim_handoff(refund["id"])["status"] == "running"


def test_checkpoint_completed_before_notice_can_be_retried(api, monkeypatch):
    order = _order_for()
    graph, config, session_id = _pause_graph(order["order_id"])
    monkeypatch.setattr(api_main, "graph", graph)
    original_finish = api_main.finish_handoff
    attempts = []

    def fail_once(*args):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("storage interruption")
        return original_finish(*args)

    monkeypatch.setattr(api_main, "finish_handoff", fail_once)
    refund = svc.list_refunds("user-a")[0]
    staff = {"X-Test-Role": "agent"}
    client = TestClient(api)
    first = client.post(f"/api/workbench/refunds/{refund['id']}/decision", headers=staff, json={"action": "approve", "note": "ok"})
    assert first.json()["resume_pending"] is True
    assert not asyncio.run(graph.aget_state(config)).next
    blocked = client.post("/api/chat", json={"session_id": session_id, "message": "hello"})
    assert blocked.status_code == 409
    retry = client.post(f"/api/workbench/refunds/{refund['id']}/resume", headers=staff)
    assert retry.json()["resumed"] is True
    assert len(asyncio.run(api_main._session_history(session_id, "user-a"))) == 1


def test_mismatched_refund_cannot_resume_another_paused_thread(api, monkeypatch, isolated_db):
    import sqlite3

    order = _order_for()
    graph, config, session_id = _pause_graph(order["order_id"])
    monkeypatch.setattr(api_main, "graph", graph)
    original = svc.list_refunds("user-a")[0]
    with sqlite3.connect(isolated_db) as db:
        db.execute("INSERT INTO orders (id, user_id, status, amount_cents, delivered_at, product) VALUES (?, ?, 'delivered', 100, ?, ?)",
                   ("another-order", "user-a", order["delivered_at"], "sample"))
    other = svc.apply_refund("user-a", "another-order", "reason", "other-key")
    with sqlite3.connect(isolated_db) as db:
        db.execute("UPDATE refund_requests SET session_id = ? WHERE id = ?", (session_id, other["id"]))
    response = TestClient(api).post(f"/api/workbench/refunds/{other['id']}/decision", headers={"X-Test-Role": "agent"}, json={"action": "approve"})
    assert response.status_code == 409
    assert svc.get_refund_review(other["id"])["status"] == "pending_review"
    assert asyncio.run(graph.aget_state(config)).values["pending_handoff"]["refund_id"] == original["id"]


def test_first_registration_role_and_live_role_update(isolated_db, monkeypatch):
    from auth.database import create_user, set_user_role
    from api.main import app

    monkeypatch.setenv("JWT_SECRET_KEY", "isolated-test-secret-that-is-long-enough")
    user = create_user("staff-test", "test-password")
    assert user["role"] == "user"
    client = TestClient(app)
    assert client.post("/api/auth/login", json={"username": "staff-test", "password": "test-password"}).status_code == 200
    assert client.get("/api/workbench/refunds").status_code == 403
    set_user_role("staff-test", "agent")
    assert client.get("/api/workbench/refunds").status_code == 200


def test_persistent_checkpointer_failure_is_not_silently_downgraded(tmp_path, monkeypatch):
    import agents.supervisor as supervisor

    monkeypatch.setattr(supervisor, "CHECKPOINT_PATH", tmp_path)
    monkeypatch.setattr(supervisor, "_CHECKPOINT_CONN", None)
    with pytest.raises(RuntimeError, match="Persistent checkpoint"):
        asyncio.run(supervisor.create_sqlite_checkpointer())


def test_corrupt_checkpoint_database_refuses_startup(tmp_path, monkeypatch):
    import agents.supervisor as supervisor

    checkpoint = tmp_path / "corrupt.sqlite3"
    checkpoint.write_bytes(b"not-a-sqlite-database")
    monkeypatch.setattr(supervisor, "CHECKPOINT_PATH", checkpoint)
    monkeypatch.setattr(supervisor, "_CHECKPOINT_CONN", None)
    with pytest.raises(RuntimeError, match="Persistent checkpoint"):
        asyncio.run(supervisor.create_sqlite_checkpointer())


def test_concurrent_session_claims_allow_only_one_winner(isolated_db):
    from concurrent.futures import ThreadPoolExecutor

    _order_for()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda i: svc.claim_session_turn("shared-session", f"token-{i}"), range(8)))
    assert sum(results) == 1


def test_concurrent_opposing_decisions_record_only_one(isolated_db):
    from concurrent.futures import ThreadPoolExecutor

    order = _order_for()
    refund = svc.apply_refund("user-a", order["order_id"], "reason", "key")
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda action: svc.record_refund_decision(refund["id"], action, "", "operator"), ("approve", "reject")))
    assert sum("error" not in result for result in results) == 1
    assert sum(result.get("error") == "invalid_state" for result in results) == 1
    assert len(svc.list_refund_actions(refund["id"])) == 1


def test_expired_leases_cannot_release_or_complete_new_claim(isolated_db):
    import sqlite3

    order = _order_for()
    assert svc.claim_session_turn("lease-session", "old-token")
    with sqlite3.connect(isolated_db) as db:
        db.execute("UPDATE session_turn_locks SET expires_at = '2000-01-01T00:00:00+00:00' WHERE session_id = ?", ("lease-session",))
    assert svc.claim_session_turn("lease-session", "new-token")
    svc.release_session_turn("lease-session", "old-token")
    assert not svc.claim_session_turn("lease-session", "third-token")

    refund = svc.apply_refund("user-a", order["order_id"], "reason", "key", "lease-session")
    svc.record_refund_decision(refund["id"], "approve", "", "operator")
    first = svc.claim_handoff(refund["id"])
    with sqlite3.connect(isolated_db) as db:
        db.execute("UPDATE handoff_outbox SET claimed_at = '2000-01-01T00:00:00+00:00' WHERE id = ?", (first["id"],))
    second = svc.claim_handoff(refund["id"])
    assert second["claimed_at"] != first["claimed_at"]
    assert not svc.finish_handoff(first["id"], first["claimed_at"], "old")
    assert svc.finish_handoff(second["id"], second["claimed_at"], "new")


def test_approved_order_cannot_start_another_refund(isolated_db):
    order = _order_for()
    first = svc.apply_refund("user-a", order["order_id"], "reason", "key-1")
    assert svc.record_refund_decision(first["id"], "approve", "", "operator")["status"] == "approved"
    assert svc.apply_refund("user-a", order["order_id"], "reason", "key-2")["error"] == "already_approved"
    assert svc.apply_refund("user-a", order["order_id"], "reason", "key-1")["id"] == first["id"]


def test_sse_chat_pause_then_staff_approval_is_visible_in_history(api, monkeypatch):
    order = _order_for()
    graph = create_supervisor_graph(_refund_llm(order["order_id"]), None, WorkingMemory(redis_url=""), None)
    monkeypatch.setattr(api_main, "graph", graph)
    client = TestClient(api)
    session_id = "sse-session"
    response = client.post("/api/chat/stream", json={"session_id": session_id, "message": "申请退款"})
    assert response.status_code == 200
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    final = next(event for event in events if event.get("event") == "final")
    assert final["awaiting_human"] is True
    refund = svc.list_refunds("user-a")[0]
    review = client.post(f"/api/workbench/refunds/{refund['id']}/decision", headers={"X-Test-Role": "agent"}, json={"action": "approve", "note": "ok"})
    assert review.status_code == 200 and review.json()["resumed"] is True
    history = client.get(f"/api/history/{session_id}").json()["messages"]
    assert any("批准" in message["content"] for message in history if message["role"] == "assistant")


def test_staff_can_rebuild_pause_after_ticket_node_crashes(api, monkeypatch):
    from agents.ticket_handler import TicketHandlerAgent
    from auth.database import save_session_owner

    order = _order_for()
    session_id = "crashed-ticket-session"
    save_session_owner(session_id, "user-a")
    original = TicketHandlerAgent.process

    async def fail_after_creation(self, state):
        await original(self, state)
        raise RuntimeError("simulated crash before graph checkpoint")

    monkeypatch.setattr(TicketHandlerAgent, "process", fail_after_creation)
    graph = create_supervisor_graph(_refund_llm(order["order_id"]), None, WorkingMemory(redis_url=None), None)
    config = {"configurable": {"thread_id": session_id}}
    with pytest.raises(RuntimeError, match="simulated crash"):
        asyncio.run(graph.ainvoke(_initial_state(session_id), config=config))
    refund = svc.list_refunds("user-a")[0]
    assert asyncio.run(graph.aget_state(config)).next == ("ticket_handler",)
    monkeypatch.setattr(api_main, "graph", graph)
    client = TestClient(api)
    staff = {"X-Test-Role": "agent"}
    blocked = client.post("/api/chat", json={"session_id": session_id, "message": "hello"})
    assert blocked.status_code == 409
    prepared = client.post(f"/api/workbench/refunds/{refund['id']}/prepare", headers=staff)
    assert prepared.status_code == 200 and prepared.json()["prepared"] is True
    state = asyncio.run(graph.aget_state(config))
    assert state.next == ("human_review",)
    decision = client.post(f"/api/workbench/refunds/{refund['id']}/decision", headers=staff, json={"action": "approve", "note": "ok"})
    assert decision.status_code == 200 and decision.json()["resumed"] is True


def test_pending_outbox_survives_restart_and_resumes_once(api, monkeypatch, tmp_path):
    import aiosqlite
    from auth.database import save_session_owner
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    order = _order_for()
    checkpoint = tmp_path / "restart-outbox.sqlite3"
    session_id = "persisted-session"
    save_session_owner(session_id, "user-a")

    async def scenario():
        conn1 = await aiosqlite.connect(str(checkpoint))
        graph1 = create_supervisor_graph(_refund_llm(order["order_id"]), None, WorkingMemory(redis_url=None), None, checkpointer=AsyncSqliteSaver(conn1))
        config = {"configurable": {"thread_id": session_id}}
        assert (await graph1.ainvoke(_initial_state(session_id), config=config)).get("__interrupt__")
        refund = svc.list_refunds("user-a")[0]
        svc.record_refund_decision(refund["id"], "approve", "ok", "operator")
        monkeypatch.setattr(api_main, "graph", None)
        assert (await api_main._resume_handoff(refund["id"]))["pending"] is True
        await conn1.close()

        conn2 = await aiosqlite.connect(str(checkpoint))
        try:
            graph2 = create_supervisor_graph(FakeLLM([AIMessage(content="unused")]), None, WorkingMemory(redis_url=None), None, checkpointer=AsyncSqliteSaver(conn2))
            monkeypatch.setattr(api_main, "graph", graph2)
            assert (await api_main._resume_handoff(refund["id"]))["resumed"] is True
            assert (await api_main._resume_handoff(refund["id"]))["resumed"] is True
            assert len(await api_main._session_history(session_id, "user-a")) == 1
            assert len(svc.list_refund_actions(refund["id"])) == 1
        finally:
            await conn2.close()

    asyncio.run(scenario())


def test_login_and_register_responses_include_role(isolated_db, monkeypatch):
    """登录/注册响应必须带 role——否则前端登录后看不到客服工作台入口。"""
    from auth.database import init_database, set_user_role

    monkeypatch.setenv("AUTH_DATABASE_PATH", str(isolated_db.parent / "app.db"))
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-for-role-payload-check")
    import importlib

    from auth import database as auth_db

    importlib.reload(auth_db)
    auth_db.init_database()
    auth_db.create_user("role_check_user", "Secret#123")
    set_user_role("role_check_user", "agent")

    monkeypatch.setenv("AUTH_DATABASE_PATH", str(isolated_db.parent / "app.db"))
    importlib.reload(auth_db)
    api_main_reloaded = importlib.reload(api_main)
    client = TestClient(api_main_reloaded.app)

    login = client.post("/api/auth/login", json={"username": "role_check_user", "password": "Secret#123"})
    assert login.status_code == 200
    assert login.json()["user"]["role"] == "agent", "登录响应必须包含实时角色"

    reg = client.post("/api/auth/register", json={"username": "role_check_new", "password": "Secret#123"})
    assert reg.status_code in (200, 201)
    assert reg.json()["user"]["role"] == "user"
