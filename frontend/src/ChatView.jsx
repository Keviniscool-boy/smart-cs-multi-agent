import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { API_URL, api, logout, INTENT_CLASS, INTENT_LABELS } from "./api.js";
import AfterSales from "./AfterSales.jsx";
import WorkbenchPanel from "./WorkbenchPanel.jsx";
import AdminPanel from "./AdminPanel.jsx";

function formatMeta(value, fallback = "—") {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value);
}

function MetricTable({ title, rows }) {
  return (
    <section className="panel-card">
      <h3>{title}</h3>
      <dl className="kv-list">
        {rows.map(([key, value]) => (
          <div className="kv-row" key={key}>
            <dt>{key}</dt>
            <dd>{formatMeta(value)}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function OpsPanel() {
  const [tab, setTab] = useState("tools");
  const [tools, setTools] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    setError("");
    if (tab === "tools" && tools === null) {
      api("/api/tools")
        .then((data) => !cancelled && setTools(data.tools || []))
        .catch((err) => !cancelled && setError(err.message));
    }
    if (tab === "metrics" && metrics === null) {
      api("/api/metrics")
        .then((data) => !cancelled && setMetrics(data.agent_metrics || {}))
        .catch((err) => !cancelled && setError(err.message));
    }
    return () => {
      cancelled = true;
    };
  }, [tab, tools, metrics]);

  return (
    <aside className="ops-panel">
      <div className="ops-tabs" role="tablist">
        <button
          className={tab === "tools" ? "active" : ""}
          onClick={() => setTab("tools")}
          role="tab"
          aria-selected={tab === "tools"}
        >
          MCP 工具
        </button>
        <button
          className={tab === "metrics" ? "active" : ""}
          onClick={() => setTab("metrics")}
          role="tab"
          aria-selected={tab === "metrics"}
        >
          运行指标
        </button>
      </div>
      <div className="ops-body">
        {error && <p className="error">{error}</p>}
        {tab === "tools" &&
          (tools === null ? (
            <p className="loading-state">正在加载工具列表...</p>
          ) : tools.length === 0 ? (
            <p className="loading-state">暂无可用工具</p>
          ) : (
            tools.map((tool) => (
              <section className="panel-card" key={tool.name}>
                <h3>
                  <code>{tool.name}</code>
                </h3>
                <p className="muted">{tool.description || "暂无描述"}</p>
              </section>
            ))
          ))}
        {tab === "metrics" &&
          (metrics === null ? (
            <p className="loading-state">正在加载指标...</p>
          ) : (
            Object.entries(metrics).map(([agent, summary]) =>
              typeof summary === "object" && summary !== null ? (
                <MetricTable
                  key={agent}
                  title={agent}
                  rows={Object.entries(summary)}
                />
              ) : (
                <MetricTable key={agent} title={agent} rows={[["值", summary]]} />
              ),
            )
          ))}
      </div>
    </aside>
  );
}

function TraceChips({ meta }) {
  if (!meta) return null;
  const intentLabel = INTENT_LABELS[meta.current_agent] || meta.current_agent;
  const intentClass = INTENT_CLASS[meta.current_agent] || "info";
  return (
    <div className="trace-chips">
      <span className={`chip ${intentClass}`} title="本轮命中的 Agent">
        {intentLabel}
      </span>
      {meta.awaiting_human ? (
        <span className="chip info" title="等待客服审核">等待人工</span>
      ) : meta.compliance_passed ? (
        <span className="chip pass" title="已通过合规审查">
          ✓ 合规
        </span>
      ) : (
        <span className="chip fail" title={meta.compliance_reason || "未通过合规审查"}>
          ⚠ 未过合规
        </span>
      )}
    </div>
  );
}

export default function ChatView({ user, onLogout }) {
  const [sessionId, setSessionId] = useState(() => crypto.randomUUID());
  const [sessions, setSessions] = useState([]);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [error, setError] = useState("");
  const [showOps, setShowOps] = useState(false);
  const [showAfterSales, setShowAfterSales] = useState(false);
  const [showWorkbench, setShowWorkbench] = useState(false);
  const [showAdmin, setShowAdmin] = useState(false);
  const isAgent = user.role === "agent" || user.role === "admin";
  const [showNav, setShowNav] = useState(false);
  const [streamStatus, setStreamStatus] = useState("");
  const messagesEndRef = useRef(null);

  const refreshSessions = useCallback(async () => {
    const data = await api("/api/sessions");
    setSessions(data.sessions || []);
  }, []);

  const loadHistory = useCallback(async (id) => {
    setShowAfterSales(false);
    setShowWorkbench(false);
    setShowNav(false);
    setLoadingHistory(true);
    setError("");
    try {
      const data = await api(`/api/history/${id}`);
      setSessionId(id);
      setMessages(data.messages || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingHistory(false);
    }
  }, []);

  useEffect(() => {
    refreshSessions().catch((err) => setError(err.message));
    const expireHandler = () => onLogout();
    window.addEventListener("auth-expired", expireHandler);
    return () => window.removeEventListener("auth-expired", expireHandler);
  }, [onLogout, refreshSessions]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy, streamStatus]);

  async function send(event) {
    event.preventDefault();
    const message = input.trim();
    if (!message || busy) return;
    setInput("");
    setError("");
    setBusy(true);
    setMessages((old) => [...old, { role: "user", content: message }]);
    setStreamStatus("正在连接...");
    try {
      // 优先走 SSE 流式端点，实时展示 Agent 处理阶段。
      const response = await fetch(`${API_URL}/api/chat/stream`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, session_id: sessionId }),
      });
      if (response.status === 401) {
        // B11：SSE 通道与 api() 保持一致的过期处理——广播登出。
        window.dispatchEvent(new Event("auth-expired"));
        throw new Error("登录已过期，请重新登录");
      }
      if (!response.ok || !response.body) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.detail || "请求失败，请稍后重试");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let finished = false;

      while (!finished) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop();
        for (const part of parts) {
          if (!part.startsWith("data: ")) continue;
          const data = JSON.parse(part.slice(6));
          if (data.event === "status") {
            setStreamStatus(data.message);
          } else if (data.event === "final") {
            finished = true;
            setMessages((old) => [
              ...old,
              {
                role: "assistant",
                content: data.response,
                meta: {
                  current_agent: data.current_agent,
                  intent: data.intent,
                  compliance_passed: data.compliance_passed,
                  compliance_reason: data.compliance_reason,
                  awaiting_human: data.awaiting_human,
                },
              },
            ]);
          } else if (data.event === "error") {
            throw new Error(data.message || "Agent processing failed");
          }
        }
      }
      await refreshSessions();
    } catch (err) {
      // 流式失败时移除本轮用户消息，保持界面一致。
      setMessages((old) => old.slice(0, -1));
      setError(err.message);
    } finally {
      setBusy(false);
      setStreamStatus("");
    }
  }

  function newConversation() {
    setShowAfterSales(false);
    setShowWorkbench(false);
    setShowNav(false);
    setSessionId(crypto.randomUUID());
    setMessages([]);
    setError("");
  }

  const currentTitle = useMemo(() => {
    const session = sessions.find((item) => item.session_id === sessionId);
    return session?.title || "新会话";
  }, [sessionId, sessions]);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="topbar-brand">
          <strong>Smart CS</strong>
          <span className="status-dot" />
          在线
        </div>
        <div className="user-menu">
          {user.role === "admin" && <button className="text-button inverse" onClick={() => { setShowAfterSales(false); setShowWorkbench(false); setShowAdmin(v => !v); }}>{showAdmin ? "返回会话" : "管理员"}</button>}
          {isAgent && (
            <button
              className="text-button inverse"
              onClick={() => { setShowAdmin(false); setShowAfterSales(false); setShowWorkbench((v) => !v); }}
            >
              {showWorkbench ? "返回会话" : "客服工作台"}
            </button>
          )}
          <button
            className="text-button inverse"
            onClick={() => { setShowAdmin(false); setShowWorkbench(false); setShowAfterSales((v) => !v); }}
          >
            {showAfterSales ? "返回会话" : "订单售后"}
          </button>
          <button className="text-button inverse" onClick={() => setShowNav((v) => !v)}>
            会话
          </button>
          <button className="text-button inverse" onClick={() => setShowOps((v) => !v)}>
            {showOps ? "隐藏面板" : "运维面板"}
          </button>
          <span className="user-badge">{user.username[0].toUpperCase()}</span>
          <span>{user.username}</span>
          <button
            className="text-button inverse"
            onClick={async () => {
              await logout();
              onLogout();
            }}
          >
            退出登录
          </button>
        </div>
      </header>
      <section className={`workspace ${showNav ? "open-nav" : ""}`}>
        <aside className="sidebar">
          <div className="side-heading">
            <span>我的会话</span>
            <button className="icon-button" title="新建会话" onClick={newConversation}>
              +
            </button>
          </div>
          <button
            className={`session-item ${
              !sessions.some((item) => item.session_id === sessionId) ? "active" : ""
            }`}
            onClick={newConversation}
          >
            <span className="session-title">{currentTitle}</span>
            <span>当前会话</span>
          </button>
          <div className="session-list">
            {sessions.map((session) => (
              <button
                className={`session-item ${
                  session.session_id === sessionId ? "active" : ""
                }`}
                key={session.session_id}
                onClick={() => loadHistory(session.session_id)}
              >
                <span className="session-title">{session.title || "历史会话"}</span>
                <span>{new Date(session.created_at).toLocaleDateString("zh-CN")}</span>
              </button>
            ))}
          </div>
          <div className="side-footer">
            当前会话
            <br />
            <code>{sessionId.slice(0, 18)}...</code>
          </div>
        </aside>

        {showAdmin && user.role === "admin" ? <AdminPanel /> : showWorkbench && isAgent ? <WorkbenchPanel /> : showAfterSales ? <AfterSales /> : <section className="chat-panel">
          <div className="chat-heading">
            <div>
              <p className="eyebrow">SUPERVISOR AGENT</p>
              <h2>智能客服</h2>
              <p className="chat-subtitle">{currentTitle}</p>
            </div>
          </div>
          <div className="messages">
            {loadingHistory && <div className="loading-state">正在加载会话...</div>}
            {!loadingHistory && messages.length === 0 && (
              <div className="empty-state">
                <div className="empty-icon">✦</div>
                <h3>有什么可以帮你？</h3>
                <p>可以咨询产品政策、查询订单或提交工单。</p>
                <div className="suggestions">
                  <button onClick={() => setInput("商品签收后几天内可以申请退货？")}>
                    退货政策
                  </button>
                  <button onClick={() => setInput("查询我的订单")}>
                    我的订单
                  </button>
                  <button onClick={() => setInput("开户需要准备什么？")}>开户流程</button>
                </div>
              </div>
            )}
            {!loadingHistory &&
              messages.map((message, index) => (
                <div className={`message-row ${message.role}`} key={`${index}-${message.content.slice(0, 24)}`}>
                  <div className="avatar">
                    {message.role === "user" ? user.username[0].toUpperCase() : "AI"}
                  </div>
                  <div className="bubble-wrap">
                    <div className="bubble">{message.content}</div>
                    {message.role === "assistant" && (
                      <TraceChips meta={message.meta} />
                    )}
                  </div>
                </div>
              ))}
            {busy && (
              <div className="message-row assistant">
                <div className="avatar">AI</div>
                <div className="bubble-wrap">
                  <div className="bubble status-bubble">
                    <span className="status-spinner" />
                    {streamStatus || "正在处理"}
                    <span className="typing-dots">
                      <span>.</span>
                      <span>.</span>
                      <span>.</span>
                    </span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          <form className="composer" onSubmit={send}>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="输入你的问题，按 Enter 发送，Shift + Enter 换行"
              rows="1"
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send(e);
                }
              }}
            />
            <button className="send-button" disabled={busy || !input.trim()} title="发送">
              ↑
            </button>
          </form>
          {error && <p className="error composer-error">{error}</p>}
          <p className="chat-footnote">
            回答由 Agent 生成，请以系统记录和公开政策为准
          </p>
        </section>}

        {showOps && <OpsPanel />}
      </section>
    </main>
  );
}
