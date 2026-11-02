import { useCallback, useEffect, useState } from "react";
import { api } from "./api.js";

const tabs = { overview: "概览", users: "账号", refunds: "退款工单", runs: "运行记录", audit: "操作审计" };

export default function AdminPanel() {
  const [tab, setTab] = useState("overview");
  const [offset, setOffset] = useState(0);
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [selectedRun, setSelectedRun] = useState(null);
  const [message, setMessage] = useState("");
  const load = useCallback(async () => {
    setError(""); setData(null);
    try { setData(await api(`/api/admin/${tab}?limit=20&offset=${offset}`)); }
    catch (err) { setError(err.message); }
  }, [tab, offset]);
  useEffect(() => { load(); }, [load]);
  async function changeRole(username, role) {
    setMessage("");
    try {
      await api("/api/admin/users/role", { method: "POST", body: JSON.stringify({ username, role }) });
      setMessage("角色已更新"); await load();
    } catch (err) { setError(err.message); }
  }
  async function openRun(id) {
    try { setSelectedRun((await api(`/api/admin/runs/${encodeURIComponent(id)}?limit=100`)).events); }
    catch (err) { setError(err.message); }
  }
  const rows = data?.[tab] || data?.events || [];
  return <section className="chat-panel admin-panel">
    <div className="chat-heading"><h2>管理员工作区</h2></div>
    <nav className="ops-tabs" aria-label="管理员功能">
      {Object.entries(tabs).map(([key, label]) => <button key={key} className={tab === key ? "active" : ""} onClick={() => { setTab(key); setOffset(0); setSelectedRun(null); setMessage(""); }}>{label}</button>)}
      <button onClick={load}>刷新</button>
    </nav>
    <div className="ops-body">
      {error && <p className="error" role="alert">{error}</p>}{message && <p>{message}</p>}
      {!data && !error && <p>正在加载...</p>}
      {data && tab === "overview" && <dl className="kv-list">
        {Object.entries(data.by_status || {}).map(([key, value]) => <div className="kv-row" key={key}><dt>{key}</dt><dd>{value}</dd></div>)}
        <div className="kv-row"><dt>等待补充信息</dt><dd>{data.waiting_for_customer}</dd></div>
        <div className="kv-row"><dt>等待恢复</dt><dd>{data.pending_resume}</dd></div>
      </dl>}
      {data && tab !== "overview" && <div style={{ overflowX: "auto" }}>
        {rows.length === 0 ? <p>暂无记录</p> : <table style={{ width: "100%", textAlign: "left", borderSpacing: "0 12px" }}><thead><tr>
          {(tab === "users" ? ["用户名", "用户 ID", "角色"] : tab === "refunds" ? ["退款单", "订单", "用户", "状态", "恢复状态", "会话 ID"] : tab === "runs" ? ["运行 ID", "用户", "开始时间", "事件数", "会话 ID"] : ["时间", "操作", "执行者", "目标 / 会话", "结果"]).map(x => <th key={x}>{x}</th>)}
        </tr></thead><tbody>{rows.map((row, index) => <tr key={row.id || row.trace_id || index}>
          {tab === "users" ? <><td>{row.username}</td><td>{row.id}</td><td><select value={row.role} onChange={e => changeRole(row.username, e.target.value)} aria-label={`${row.username} 的角色`}><option value="user">用户</option><option value="agent">客服</option><option value="admin">管理员</option></select></td></> :
          tab === "refunds" ? <><td>{row.id}</td><td>{row.order_id}</td><td>{row.username || row.user_id}</td><td>{row.status}</td><td>{row.resume_status || "—"}</td><td>{row.session_id || "—"}</td></> :
          tab === "runs" ? <><td><button onClick={() => openRun(row.trace_id)}>{row.trace_id}</button></td><td>{row.actor_id}</td><td>{row.started_at}</td><td>{row.event_count}</td><td>{row.session_id}</td></> :
          <><td>{row.created_at}</td><td>{row.action}</td><td>{row.actor_id}</td><td>{row.target_id || row.refund_id || row.session_id}</td><td>{row.detail || row.outcome}</td></>}
        </tr>)}</tbody></table>}
        <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 20))}>上一页</button>{" "}<button disabled={rows.length < 20} onClick={() => setOffset(offset + 20)}>下一页</button>
      </div>}
      {selectedRun && <section><h3>运行详情</h3>{selectedRun.length === 0 ? <p>暂无事件</p> : selectedRun.map(event => <p key={event.id}>{event.created_at} · {event.action} · {event.outcome} · {event.detail}</p>)}</section>}
    </div>
  </section>;
}
