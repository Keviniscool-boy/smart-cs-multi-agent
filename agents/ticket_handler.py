import asyncio
import hashlib
import json
import uuid

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from mcp_client.client import MCPClient
from mcp_tools.order_server import server as local_mcp_server
from tools.order import order_query
from tools.ticket import ticket_create
from storage.after_sales import apply_refund, list_orders
from tracing.otel_config import trace_agent_call


class TicketHandlerAgent:
    def __init__(
        self,
        llm: ChatOpenAI,
        mcp_client: MCPClient,
    ):
        self.llm = llm
        self.mcp_client = mcp_client

    async def _call_mcp_tool(
        self,
        tool_name: str,
        arguments: dict,
    ) -> dict:
        """通过 MCP Client 调用工具，并提取工具返回的数据。"""
        result = await self.mcp_client.call_tool(
            tool_name,
            arguments,
        )

        if result.is_error:
            raise RuntimeError(f"MCP tool failed: {tool_name}")

        for item in result.content:
            if getattr(item, "type", None) == "text":
                return json.loads(item.text)

        if result.structuredContent:
            return result.structuredContent

        return {}

    @staticmethod
    def _local_tool(tool_name: str, arguments: dict) -> dict:
        """运行时 MCP 失败后的本地备用工具。"""
        if tool_name == "order_query":
            return order_query.invoke(
                {key: value for key, value in arguments.items() if key != "request_id"}
            )
        if tool_name == "order_list":
            return {"orders": list_orders(arguments["user_id"])}
        if tool_name == "refund_apply":
            return apply_refund(
                arguments["user_id"],
                arguments["order_id"],
                arguments["reason"],
                arguments["idempotency_key"],
                arguments.get("session_id", ""),
                arguments.get("application_type") or None,
            )
        if tool_name == "ticket_create":
            # LLM 可能多传无关字段，白名单过滤防止 TypeError。
            allowed = {"title", "description", "priority", "category", "idempotency_key"}
            return ticket_create(**{key: arguments[key] for key in allowed if key in arguments})
        raise RuntimeError(f"没有本地备用工具: {tool_name}")

    async def _call_with_retry(
        self,
        tool_name: str,
        arguments: dict,
    ) -> tuple[dict, int]:
        """只对网络类异常做有限重试，业务错误不重试。"""
        retry_delays = (1, 2)
        for retry_count in range(len(retry_delays) + 1):
            try:
                return (
                    await self._call_mcp_tool(tool_name, arguments),
                    retry_count,
                )
            except (TimeoutError, ConnectionError):
                if retry_count == len(retry_delays):
                    raise
                await asyncio.sleep(retry_delays[retry_count])

    @staticmethod
    def build_idempotency_key(user_id: str, arguments: dict) -> str:
        """按业务语义（用户 + 工单内容指纹）生成幂等键。

        不使用本次 tool_call id，这样重试、降级和用户重复提交
        同一工单都命中同一条幂等记录。
        """
        fingerprint = json.dumps(
            {
                "title": arguments.get("title", ""),
                "description": arguments.get("description", ""),
                "category": arguments.get("category", "general"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
        return f"{user_id or 'anonymous'}:{digest}"

    @trace_agent_call("ticket_handler")
    async def process(self, state: dict):
        messages = [
            SystemMessage(
                content=(
                    "你是订单客服 Agent。"
                    "用户询问订单状态或物流时，必须调用 order_query 工具。"
                    "如果用户没有提供订单号，请先询问订单号。"
                    "查询自己的订单列表时使用 order_list。"
                    "用户明确申请退款时使用 refund_apply；缺少订单号或用户提供的退款原因时先询问，绝不编造原因。"
                    "仅当用户明确说明无理由或商品质量问题时，分别传入 application_type 的 no_reason 或 quality_issue；不确定时留空并转人工。"
                    "退款申请只是等待人工审核，不代表退款成功或资金已退回。"
                    "不要编造订单信息。"
                )
            ),
            *state["messages"],
        ]

        # 从 MCP Server 获取工具描述，并转换成 OpenAI function schema。
        try:
            mcp_tools = await self.mcp_client.list_tools()
            available_tools = mcp_tools.tools
            use_local = False
        except (TimeoutError, ConnectionError, RuntimeError, OSError, AttributeError):
            # AttributeError：测试/降级场景 mcp_client 为 None，直接走本地工具。
            available_tools = await local_mcp_server.list_tools()
            use_local = True
        # 从 LLM 可见参数中剔除服务端注入字段（user_id/session_id 等），防伪造身份与幂等键。
        llm_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": {
                        **tool.input_schema,
                        "properties": {k: v for k, v in tool.input_schema.get("properties", {}).items() if k not in {"user_id", "request_id", "idempotency_key", "session_id"}},
                        "required": [k for k in tool.input_schema.get("required", []) if k not in {"user_id", "request_id", "idempotency_key", "session_id"}],
                    },
                },
            }
            for tool in available_tools if tool.name in {"order_query", "order_list", "refund_apply", "ticket_create"}
        ]
        llm_with_tools = self.llm.bind_tools(llm_tools)

        # 第一次调用由 LLM 决定是否需要使用 MCP 工具。
        response = await llm_with_tools.ainvoke(messages)

        if not response.tool_calls:
            return {
                "messages": [response],
                "final_response": response.content,
                "sub_results": {
                    **state.get("sub_results", {}),
                    "ticket_handler": response.content,
                },
            }

        messages.append(response)
        tool_messages = []
        tool_results = []

        for tool_call in response.tool_calls:
            print(
                f"[TicketHandlerAgent] 调用 MCP 工具: {tool_call['name']}"
            )
            tool_name = tool_call["name"]
            if tool_name not in {"order_query", "order_list", "refund_apply", "ticket_create"}:
                raise ValueError("Unsupported tool")
            tool_arguments = dict(tool_call["args"])
            tool_arguments.pop("user_id", None)
            tool_arguments.pop("idempotency_key", None)
            if tool_name in {"order_query", "order_list", "refund_apply"}:
                tool_arguments["user_id"] = state["user_id"]
            # 阶段二：会话号服务端注入，退款单据此关联可恢复的图线程。
            if tool_name == "refund_apply":
                tool_arguments["session_id"] = state.get("session_id", "")
            request_id = tool_call["id"] or str(uuid.uuid4())
            tool_arguments["request_id"] = request_id
            if tool_name == "refund_apply":
                tool_arguments["idempotency_key"] = f"agent:{request_id}"
            if tool_name == "ticket_create":
                tool_arguments["idempotency_key"] = self.build_idempotency_key(
                    state.get("user_id", ""),
                    tool_arguments,
                )

            fallback_used = use_local
            error_type = "mcp_unavailable" if use_local else None
            retry_count = 0
            try:
                if use_local:
                    result = await asyncio.to_thread(self._local_tool, tool_name, tool_arguments)
                else:
                    result, retry_count = await self._call_with_retry(
                        tool_name,
                        tool_arguments,
                    )
            except TimeoutError as error:
                error_type = "timeout"
                fallback_used = True
                result = await asyncio.to_thread(self._local_tool, tool_name, tool_arguments)
            except (ConnectionError, RuntimeError) as error:
                error_type = "mcp_error"
                fallback_used = True
                result = await asyncio.to_thread(self._local_tool, tool_name, tool_arguments)

            tool_results.append(
                {
                    "tool": tool_name,
                    "request_id": request_id,
                    "error_type": error_type,
                    "fallback_used": fallback_used,
                    "retry_count": retry_count,
                    "result": result,
                }
            )
            tool_message = ToolMessage(
                content=json.dumps(result, ensure_ascii=False),
                tool_call_id=tool_call["id"],
            )
            messages.append(tool_message)
            tool_messages.append(tool_message)

        # 阶段二：本轮创建了待人工审核的退款单 → 触发 human_review 暂停。
        # 无命中时必须显式清空，防止同线程上一轮的暂停信号残留误触发。
        pending_handoff = {}
        for entry in tool_results:
            result = entry.get("result") or {}
            if (entry.get("tool") == "refund_apply" and result.get("status") == "pending_review"
                    and result.get("session_id") == state.get("session_id")):
                pending_handoff = {
                    "refund_id": result.get("id", ""),
                    "ticket_id": result.get("ticket_id", ""),
                    "order_id": result.get("order_id", ""),
                    "reason": result.get("reason", ""),
                }
                break

        # 退款单已落库，直接进入持久化中断，不再做第二次模型调用——
        # 模型调用一旦失败，申请就会无人认领。
        final_response = (
            AIMessage(content="您的退款申请已提交人工审核，客服处理后将继续为您跟进。")
            if pending_handoff else await llm_with_tools.ainvoke(messages)
        )

        return {
            "messages": [response, *tool_messages, final_response],
            "final_response": final_response.content,
            "pending_handoff": pending_handoff,
            # 等待话术未经模型审查，如实标记合规状态，避免下游当作已审内容。
            **({"compliance_passed": False, "compliance_reason": "固定系统等待话术，未执行模型审查"} if pending_handoff else {}),
            "sub_results": {
                **state.get("sub_results", {}),
                "ticket_handler": final_response.content,
                "ticket_handler_tool_results": tool_results,
            },
        }
