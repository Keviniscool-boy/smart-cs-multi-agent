import asyncio
import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer

# MCP Inspector 按文件路径加载 Server 时，把项目根目录加入导入路径。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from storage.after_sales import apply_refund, get_order, list_orders

from tools.ticket_store import create_or_get_ticket

server=MCPServer(
    name="smart-cs-order-server",
    version="1.0.0",
)

@server.tool(
    name="order_query",
    description="查询当前用户的演示订单",
)
async def order_query(

    order_id:str ="",
    user_id:str = "",
    request_id: str = "",
) -> dict:
    return await asyncio.to_thread(get_order, user_id, order_id)


@server.tool(name="order_list", description="列出当前用户的演示订单")
async def order_list(user_id: str = "", request_id: str = "") -> dict:
    return {"orders": await asyncio.to_thread(list_orders, user_id)}


@server.tool(name="refund_apply", description="为自己的演示订单提交退款申请；需要订单号和用户提供的退款原因，不执行支付")
async def refund_apply(order_id: str, reason: str, user_id: str = "", request_id: str = "", idempotency_key: str = "", session_id: str = "") -> dict:
    return await asyncio.to_thread(apply_refund, user_id, order_id, reason, idempotency_key, session_id)

@server.tool(
    name="knowledge_search",
    description="搜索企业知识库，返回相关文档片段（mock 模板数据，演示用）",
)
async def knowledge_search(
    query:str,
    top_k: int =3,

)->list[dict]:
    return[
        {
            "content":f"关于'{query}'的知识库文档片段",
            "source":f"FAQ.md",
            "score":0.95,
        }
    ]


@server.tool(
    name="ticket_create",
    description="创建工单，返回工单ID",
)
async def ticket_create(
    title:str,
    description:str,
    priority:str="medium",
    category:str="general",
    idempotency_key: str = "",
    request_id: str = "",
)->dict:
    return create_or_get_ticket(
        title=title,
        description=description,
        priority=priority,
        category=category,
        idempotency_key=idempotency_key or request_id,
    )

@server.tool(
    name="risk_check",
    description="检查交易操作的风险等级（演示实现：仅按金额阈值）",
)
async def risk_check(
    user_id:str,
    action:str,
    amount:float=0.0,
)->dict:
    risk_level="low"
    if amount>50000:
        risk_level="high"
    elif amount>10000:
        risk_level="medium"
    return {
        "user_id":user_id,
        "action":action,
        "risk_level":risk_level,
        "requires_manual_review":risk_level == "high",
    }

if __name__ == "__main__":
    import asyncio
    asyncio.run(server.run_stdio_async())
