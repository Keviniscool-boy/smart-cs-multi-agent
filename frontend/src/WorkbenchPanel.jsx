import { useCallback, useEffect, useState } from "react";
import { api } from "./api.js";

const refundStatus = {
  pending_review: "待人工审核",
  cancelled: "已撤回",
  approved: "已批准",
  rejected: "已拒绝",
  completed: "已完成",
};
const actionLabel = { approve: "批准", reject: "拒绝", need_info: "要求补充" };

export default function WorkbenchPanel() {
  const [refunds, setRefunds] = useState([]);
  const [detail, setDetail] = useState(null);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const loadList = useCallback(async () => {
    const data = await api("/api/workbench/refunds");
    setRefunds(data.refunds || []);
  }, []);

  useEffect(() => {
    loadList().catch((err) => setError(err.message));
  }, [loadList]);

  async function openDetail(refundId) {
    setError("");
    setNotice("");
    try {
      const data = await api(`/api/workbench/refunds/${encodeURIComponent(refundId)}`);
      setDetail(data);
      setNote("");
    } catch (err) {
      setError(err.message);
    }
  }

  async function decide(action) {
    if (busy || !detail) return;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const result = await api(`/api/workbench/refunds/${encodeURIComponent(detail.refund.id)}/decision`, {
        method: "POST",
        body: JSON.stringify({ action, note: note.trim() }),
      });
      setNote("");
      await loadList();
      await openDetail(detail.refund.id);
      setNotice(action === "need_info" ? "已要求客户补充信息。" : result.resume_pending ? "审核已保存，会话恢复等待重试。" : result.resumed ? "流程已恢复，客户可查看答复。" : "状态已更新。");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const pending = detail?.refund.status === "pending_review" && detail.refund.ticket_status === "pending" && (!detail.session_id || detail.thread_paused);

  async function retryResume() {
    if (busy || !detail) return;
    setBusy(true);
    setError("");
    try {
      const result = await api(`/api/workbench/refunds/${encodeURIComponent(detail.refund.id)}/resume`, { method: "POST" });
      await openDetail(detail.refund.id);
      setNotice(result.pending ? "恢复尚未完成，请稍后重试。" : "会话已恢复。");
    } catch (err) { setError(err.message); } finally { setBusy(false); }
  }

  async function retryPreparation() {
    if (busy || !detail) return;
    setBusy(true);
    setError("");
    try {
      await api(`/api/workbench/refunds/${encodeURIComponent(detail.refund.id)}/prepare`, { method: "POST" });
      await openDetail(detail.refund.id);
      setNotice("已建立人工审核暂停点。");
    } catch (err) { setError(err.message); } finally { setBusy(false); }
  }

  return (
    <section className="after-sales" aria-label="客服工作台">
      <header><h2>客服工作台</h2><p className="muted">处理转人工的退款申请；批准后 Agent 会恢复流程并答复客户。</p></header>
      {error && <p className="error" role="alert">{error}</p>}
      {notice && <p className="after-sales-notice" role="status">{notice}</p>}
      <h3>退款申请</h3>
      <div className="after-sales-list">
        {refunds.map((refund) => (
          <article className={`after-sales-row ${detail?.refund.id === refund.id ? "active" : ""}`} key={refund.id} data-status={refund.status}>
            <button className="text-button" onClick={() => openDetail(refund.id)}>
              <strong>{refundStatus[refund.status] || refund.status}</strong>
              <p>{refund.id}</p>
              <span>{refund.username || refund.user_id} · 订单 {refund.order_id} · {refund.reason}</span>
            </button>
          </article>
        ))}
        {refunds.length === 0 && <p className="muted">暂无退款申请</p>}
      </div>

      {detail && (
        <>
          <h3>申请详情</h3>
          <section className="panel-card">
            <p><strong>{detail.refund.id}</strong> · {refundStatus[detail.refund.status] || detail.refund.status}</p>
            <p>客户：{detail.refund.username || detail.refund.user_id}</p>
            <p>订单：{detail.refund.order_id}（{detail.refund.product}，¥{(detail.refund.amount_cents / 100).toFixed(2)}）</p>
            <p>退款原因：{detail.refund.reason}</p>
            <p>申请类型：{({ no_reason: "无理由退货", quality_issue: "质量问题" })[detail.refund.application_type] || "待确认"}</p>
            <p>规则判定：{detail.refund.policy_reason || "历史申请，未记录"} {detail.refund.policy_ref && `（${detail.refund.policy_ref}）`}</p>
            {detail.thread_paused && <p className="muted">Agent 流程已暂停在人工审核节点，等待你的决定。</p>}
            {detail.handoff && (
              <>
                <p className="muted">Agent 交接摘要：{detail.handoff.agent_summary}</p>
                <p className="muted">用户原话：{detail.handoff.user_request}</p>
              </>
            )}
          </section>

          {detail.history?.length > 0 && (
            <>
              <h3>会话历史</h3>
              <section className="panel-card">
                {detail.history.slice(-6).map((message, index) => (
                  <p key={index}><strong>{message.role === "user" ? "客户" : "Agent"}：</strong>{message.content}</p>
                ))}
              </section>
            </>
          )}

          {detail.handoff?.tool_results?.length > 0 && <>
            <h3>工具结果</h3>
            <section className="panel-card">
              {detail.handoff.tool_results.map((item, index) => <div key={index}>
                <strong>{item.tool}</strong>
                <pre className="workbench-evidence">{JSON.stringify(item.result, null, 2)}</pre>
              </div>)}
            </section>
          </>}
          {detail.handoff?.knowledge_sources?.length > 0 && <>
            <h3>知识来源</h3>
            <section className="panel-card">
              {detail.handoff.knowledge_sources.map((item, index) => <p key={index}><strong>{item.source}</strong>：{item.content}</p>)}
            </section>
          </>}

          {detail.customer_updates?.length > 0 && <><h3>客户补充</h3><section className="panel-card">{detail.customer_updates.map((update, index) => <p key={index}>{update.content}</p>)}</section></>}
          {detail.refund.ticket_status === "processing" && <p className="muted">等待客户补充信息。</p>}
          {detail.session_id && detail.refund.status === "pending_review" && !detail.thread_paused && <button className="text-button" disabled={busy} onClick={retryPreparation}>重试建立人工审核</button>}
          {detail.handoff_event && detail.handoff_event.status !== "done" && <button className="text-button" disabled={busy} onClick={retryResume}>重试会话恢复</button>}

          {pending && (
            <>
              <h3>人工决定</h3>
              <form className="after-sales-form" onSubmit={(event) => { event.preventDefault(); decide("need_info"); }}>
                <label>备注 / 追问内容<textarea rows={2} maxLength={500} value={note} onChange={(event) => setNote(event.target.value)} /></label>
                <div className="after-sales-actions">
                  <button type="button" className="primary" disabled={busy} onClick={() => decide("approve")}>批准</button>
                  <button type="button" className="text-button" disabled={busy} onClick={() => decide("reject")}>拒绝</button>
                  <button type="submit" className="text-button" disabled={busy || !note.trim()}>要求补充信息</button>
                </div>
              </form>
            </>
          )}

          {detail.actions?.length > 0 && (
            <>
              <h3>操作记录</h3>
              <section className="panel-card">
                {detail.actions.map((action) => (
                  <p key={action.id}>
                    <strong>{actionLabel[action.action] || action.action}</strong> · {action.operator || action.operator_id} ·{" "}
                    {new Date(action.created_at).toLocaleString("zh-CN")}
                    {action.note && <> · {action.note}</>}
                  </p>
                ))}
              </section>
            </>
          )}
        </>
      )}
    </section>
  );
}
