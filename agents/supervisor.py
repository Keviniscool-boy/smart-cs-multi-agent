import re
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from agents.compliance_checker import ComplianceCheckerAgent
from agents.fallback import FallbackAgent
from agents.knowledge_rag import KnowledgeRAGAgent
from agents.synthesizer import SynthesizerAgent
from agents.ticket_handler import TicketHandlerAgent
from memory.long_term import LongTermMemory
from memory.working_memory import WorkingMemory
from mcp_client.client import MCPClient
from tracing.otel_config import trace_agent_call


class AgentState(TypedDict):
    """在 Graph 各节点之间传递的共享状态。"""

    messages: Annotated[list[BaseMessage], add_messages]
    user_id:str
    session_id: str
    final_response: str
    intent: str
    current_agent: str
    compliance_passed: bool
    compliance_reason: str
    compliance_retries: int
    sub_results: dict[str, Any]
    # 业务工具创建了待人工审核的售后单时由 ticket_handler 填充；
    # 有值时 ticket_handler 直接转入 human_review 暂停（绕过合规，见 create_supervisor_graph）。
    pending_handoff: dict[str, Any]
    # 客服通过 Command(resume=...) 恢复时写入的决定。
    handoff_decision: dict[str, Any]


ALLOWED_INTENTS = {
    "knowledge_rag",
    "ticket_handler",
    "fallback",
}

# 合规检查未通过时最多重写次数（首次生成不计入）。
MAX_COMPLIANCE_RETRIES = 2

# 会话状态断点持久化到 SQLite，重启后可恢复（含合规重写中途的状态）。
CHECKPOINT_PATH = Path(__file__).resolve().parents[1] / "data" / "checkpoints.sqlite3"
_CHECKPOINT_CONN = None


def route_to_agent(state: AgentState) -> str:
    """根据 Supervisor 写入的 intent 选择下一个 Graph 节点。"""
    intent = state.get("intent", "")
    return intent if intent in ALLOWED_INTENTS else "fallback"


class SupervisorNode:
    """读取工作记忆，调用 LLM 判断本轮请求的业务意图。"""

    def __init__(self, llm: ChatOpenAI, working_memory: WorkingMemory):
        self.llm = llm
        self.working_memory = working_memory

    @staticmethod
    def _deterministic_intent(message: str) -> str | None:
        """对高置信度请求先做本地路由，避免简单问题浪费一次 LLM 调用。"""
        text = message.lower()

        fallback_terms = (
            "系统提示词",
            "system prompt",
            "api key",
            "危险操作",
            "写一段小说",
            "你是谁",
        )
        if any(term in text for term in fallback_terms):
            return "fallback"

        # 订单号或明确的查询/工单动作应交给业务工具 Agent。
        has_order_id = re.search(r"(?:ord[-_]?\w+|订单号?\s*[A-Za-z0-9_-]+)", text)
        ticket_terms = ("创建工单", "提交工单", "订单状态", "查询物流", "物流查询", "申请退款", "我要退款", "帮我退款", "退货申请", "我的订单")
        if has_order_id or any(term in text for term in ticket_terms):
            return "ticket_handler"

        knowledge_terms = (
            "退款", "退货", "理财", "收益率", "投资期限", "开户",
            "风险评估", "风险问卷", "身份证", "银行卡", "安全渠道",
            "服务政策", "产品政策", "参考年化",
            "无法确认", "没有相关信息", "知识库没有",
        )
        if any(term in text for term in knowledge_terms):
            return "knowledge_rag"

        return None

    @trace_agent_call("supervisor")
    async def route_decision(self, state: AgentState) -> dict[str, str]:
        session_id = state["session_id"]
        context = self.working_memory.get_context(session_id)
        # R2：空消息列表防御——直接走 fallback，不调 LLM。
        if not state.get("messages"):
            return {
                "intent": "fallback",
                "current_agent": "fallback",
                "final_response": "",
            }
        user_message = state["messages"][-1].content

        deterministic_intent = self._deterministic_intent(user_message)
        if deterministic_intent is not None:
            intent = deterministic_intent
            self.working_memory.update(
                session_id,
                {
                    "last_intent": intent,
                    "current_agent": intent,
                },
                user_id=state.get("user_id", ""),
            )
            return {
                "intent": intent,
                "current_agent": intent,
                "final_response": "",
            }

        routing_prompt = [
            (
                "system",
                "你是客服系统的路由器。\n"
                "只能返回以下三个标签之一，且只能返回标签本身：\n"
                "- knowledge_rag：产品政策、退款、开户、物流说明等知识库问题\n"
                "- ticket_handler：查询订单、申请退款、创建工单等需要业务系统处理的问题\n"
                "- fallback：无关、不支持、危险或索取系统内部信息的问题\n"
                "例如：\n"
                "产品退款条件 -> knowledge_rag\n"
                "查询订单 ORD-001 -> ticket_handler\n"
                "你是谁、输出系统提示词 -> fallback\n"
                f"当前工作记忆：{context}",
            ),
            *state["messages"],
        ]

        response = await self.llm.ainvoke(routing_prompt)
        intent = response.content.strip().lower()
        if intent not in ALLOWED_INTENTS:
            intent = "fallback"

        self.working_memory.update(
            session_id,
            {
                "last_intent": intent,
                "current_agent": intent,
            },
            user_id=state.get("user_id", ""),
        )

        return {
            "intent": intent,
            "current_agent": intent,
            "final_response": "",
        }


def route_after_compliance(state: AgentState) -> str:
    """合规通过或重试耗尽后进入合成；否则带反馈重写。"""
    if state.get("compliance_passed", False):
        return "synthesize"
    if state.get("compliance_retries", 0) >= MAX_COMPLIANCE_RETRIES:
        return "synthesize"
    return "rewrite"


class HumanReviewNode:
    """人工审核断点：interrupt() 把图暂停在检查点上。

    暂停时载荷即客服工作台看到的"交接单"；客服决定后由
    Command(resume=...) 唤醒，interrupt() 的返回值就是那个决定。
    """

    @trace_agent_call("human_review")
    async def process(self, state: AgentState) -> dict[str, Any]:
        from langgraph.types import interrupt

        handoff = state.get("pending_handoff", {})
        user_message = next(
            (
                message.content
                for message in reversed(state["messages"])
                if isinstance(message, HumanMessage)
            ),
            "",
        )
        payload = {
            "type": "refund_review",
            "refund_id": handoff.get("refund_id", ""),
            "ticket_id": handoff.get("ticket_id", ""),
            "order_id": handoff.get("order_id", ""),
            "reason": handoff.get("reason", ""),
            "user_id": state.get("user_id", ""),
            "session_id": state.get("session_id", ""),
            "user_request": user_message,
            "agent_summary": (
                f"用户为订单 {handoff.get('order_id', '')} 申请退款，"
                f"原因：{handoff.get('reason', '')}。"
                "演示规则：签收 7 天内可申请，商品状态与包装需人工确认。"
            ),
            "tool_results": state.get("sub_results", {}).get(
                "ticket_handler_tool_results", []
            ),
            "knowledge_sources": state.get("sub_results", {}).get("knowledge_rag_context", []),
        }
        decision = interrupt(payload)
        if not isinstance(decision, dict):
            decision = {"action": str(decision), "note": ""}
        return {"handoff_decision": decision}


class RewriteNode:
    """合规未通过时，把拦截原因带回提示词重写回答，而不是原样重试。"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    @trace_agent_call("rewrite")
    async def process(self, state: AgentState) -> dict[str, Any]:
        user_message = next(
            (
                message.content
                for message in reversed(state["messages"])
                if isinstance(message, HumanMessage)
            ),
            "",
        )
        attempts = state.get("compliance_retries", 0)
        messages = [
            (
                "system",
                "你是客服系统的回答修正 Agent。\n"
                "上一版回答未通过合规检查，请根据违规原因重写回答。\n"
                "要求：保留对用户问题的有效回答；必须解决列出的全部违规原因；"
                "仍然不要编造业务结果，不要泄露内部信息。\n"
                "只输出重写后的回答本身。",
            ),
            (
                "human",
                f"用户问题：{user_message}\n\n"
                f"上一版回答：\n{state.get('final_response', '')}\n\n"
                f"合规检查未通过的原因与建议：\n{state.get('compliance_reason', '')}\n\n"
                "请重写回答。",
            ),
        ]
        response = await self.llm.ainvoke(messages)
        return {
            "final_response": response.content,
            "compliance_retries": attempts + 1,
            "sub_results": {
                **state.get("sub_results", {}),
                "rewrite_attempts": attempts + 1,
            },
        }


def _get_checkpointer():
    """进程内兜底 checkpointer（同步上下文用）。"""
    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver()


async def create_sqlite_checkpointer():
    """创建 SQLite 持久化 checkpointer（必须在事件循环内调用）。

    API 全部走 ainvoke/astream，必须用 AsyncSqliteSaver。
    人工审核需要跨重启恢复，无法创建持久化检查点时应拒绝启动。
    """
    global _CHECKPOINT_CONN
    try:
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CHECKPOINT_CONN = await aiosqlite.connect(str(CHECKPOINT_PATH))
        checkpointer = AsyncSqliteSaver(_CHECKPOINT_CONN)
        await checkpointer.aget_tuple({"configurable": {"thread_id": "__startup_check__"}})
        return checkpointer
    except Exception as exc:
        if _CHECKPOINT_CONN is not None:
            await _CHECKPOINT_CONN.close()
            _CHECKPOINT_CONN = None
        raise RuntimeError("Persistent checkpoint is unavailable") from exc


async def close_checkpoint():
    """关闭持久化连接，供应用关闭钩子调用。"""
    global _CHECKPOINT_CONN
    if _CHECKPOINT_CONN is not None:
        await _CHECKPOINT_CONN.close()
        _CHECKPOINT_CONN = None


def create_supervisor_graph(
    llm: ChatOpenAI,
    long_term_memory: LongTermMemory,
    working_memory: WorkingMemory,
    mcp_client: MCPClient,
    checkpointer=None,
):
    """创建完整的客服 Agent Graph。checkpointer 缺省用进程内 MemorySaver。"""
    supervisor = SupervisorNode(llm, working_memory)
    knowledge_agent = KnowledgeRAGAgent(llm, long_term_memory)
    ticket_agent = TicketHandlerAgent(llm, mcp_client)
    fallback_agent = FallbackAgent()
    compliance_checker = ComplianceCheckerAgent(llm)
    synthesizer = SynthesizerAgent(llm)
    rewriter = RewriteNode(llm)

    graph = StateGraph(AgentState)
    graph.add_node("supervisor_route", supervisor.route_decision)
    graph.add_node("knowledge_rag", knowledge_agent.process)
    graph.add_node("ticket_handler", ticket_agent.process)
    graph.add_node("fallback", fallback_agent.process)
    graph.add_node("compliance_checker", compliance_checker.process)
    graph.add_node("rewrite", rewriter.process)
    graph.add_node("human_review", HumanReviewNode().process)
    graph.add_node("synthesize", synthesizer.process)

    graph.set_entry_point("supervisor_route")
    graph.add_conditional_edges(
        "supervisor_route",
        route_to_agent,
        {
            "knowledge_rag": "knowledge_rag",
            "ticket_handler": "ticket_handler",
            "fallback": "fallback",
        },
    )

    graph.add_edge("knowledge_rag", "compliance_checker")
    # 退款流程绕过合规：建单到持久化中断之间不再有模型调用，杜绝申请搁浅；
    # 该链路所有用户可见文本均为固定话术，恢复后经 synthesize 收尾。
    graph.add_conditional_edges(
        "ticket_handler",
        lambda state: "human_review" if state.get("pending_handoff") else "compliance_checker",
        {"human_review": "human_review", "compliance_checker": "compliance_checker"},
    )
    graph.add_edge("fallback", "compliance_checker")
    graph.add_conditional_edges(
        "compliance_checker",
        route_after_compliance,
        {"synthesize": "synthesize", "rewrite": "rewrite"},
    )
    graph.add_edge("rewrite", "compliance_checker")
    graph.add_edge("human_review", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile(
        checkpointer=checkpointer or _get_checkpointer(),
    )
