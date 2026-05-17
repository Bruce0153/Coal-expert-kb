const STORAGE_KEYS = {
  settings: "coal-kb-ui-settings",
};

// Pixel art SVG icons (16×16 viewport, pixelated)
const PIXEL_ICONS = {
  ai: `<svg class="pixel-icon" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">
    <rect x="5" y="1" width="6" height="2" fill="#b87a1f"/>
    <rect x="3" y="3" width="4" height="2" fill="#b87a1f"/>
    <rect x="9" y="3" width="4" height="2" fill="#b87a1f"/>
    <rect x="1" y="5" width="2" height="6" fill="#b87a1f"/>
    <rect x="13" y="5" width="2" height="6" fill="#b87a1f"/>
    <rect x="3" y="5" width="10" height="6" fill="#fef9f0"/>
    <rect x="5" y="6" width="2" height="2" fill="#145e69"/>
    <rect x="9" y="6" width="2" height="2" fill="#145e69"/>
    <rect x="5" y="9" width="6" height="1" fill="#145e69"/>
    <rect x="3" y="11" width="4" height="2" fill="#b87a1f"/>
    <rect x="9" y="11" width="4" height="2" fill="#b87a1f"/>
    <rect x="5" y="13" width="6" height="2" fill="#b87a1f"/>
  </svg>`,
  user: `<svg class="pixel-icon" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">
    <rect x="6" y="1" width="4" height="3" fill="#fff"/>
    <rect x="5" y="4" width="6" height="1" fill="#fff"/>
    <rect x="4" y="5" width="2" height="1" fill="#fff"/>
    <rect x="10" y="5" width="2" height="1" fill="#fff"/>
    <rect x="2" y="6" width="2" height="4" fill="#fff"/>
    <rect x="12" y="6" width="2" height="4" fill="#fff"/>
    <rect x="4" y="8" width="8" height="1" fill="#fff"/>
    <rect x="5" y="9" width="6" height="2" fill="#fff"/>
    <rect x="4" y="11" width="8" height="1" fill="#fff"/>
    <rect x="5" y="12" width="2" height="3" fill="#fff"/>
    <rect x="9" y="12" width="2" height="3" fill="#fff"/>
  </svg>`,
};

// ---- Global State ----
const state = {
  defaults: null,
  settings: null,
  conversations: [],
  activeConversationId: null,
  messages: [],
  selectedAssistantMessageId: null,
  sending: false,
  selectedFiles: [],   // files pending upload
  manageTab: "tab-upload",
};

// ---- DOM refs ----
function $(id) { return document.getElementById(id); }

const els = {
  body: document.body,
  backendPill: $("backend-pill"),
  threadStatus: $("thread-status"),
  conversationEmpty: $("conversation-empty"),
  conversationList: $("conversation-list"),
  welcomePanel: $("welcome-panel"),
  threadEmpty: $("thread-empty"),
  thread: $("message-thread"),
  typingIndicator: $("typing-indicator"),
  form: $("chat-form"),
  messageInput: $("message-input"),
  composerNote: $("composer-note"),
  sendButton: $("send-button"),
  settingsSummary: $("settings-summary"),
  inspectorEmpty: $("inspector-empty"),
  inspectorContent: $("inspector-content"),
  inspectorCaption: $("inspector-caption"),
  inspectorBadges: $("inspector-badges"),
  claimList: $("claim-list"),
  citationReferenceList: $("citation-reference-list"),
  evidenceList: $("evidence-list"),
  sourceCardList: $("source-card-list"),
  newChatTop: $("new-chat-top"),
  newChatSidebar: $("new-chat-sidebar"),
  refreshConversations: $("refresh-conversations"),
  settingsToggle: $("settings-toggle"),
  settingsModal: $("settings-modal"),
  settingsBackdrop: $("settings-backdrop"),
  settingsClose: $("settings-close"),
  settingsForm: $("settings-form"),
  settingsReset: $("settings-reset"),
  settingsNotes: $("settings-notes"),
  settingApiBaseUrl: $("setting-api-base-url"),
  settingProviderBaseUrl: $("setting-provider-base-url"),
  settingApiKey: $("setting-api-key"),
  settingLlmProvider: $("setting-llm-provider"),
  settingLlmModel: $("setting-llm-model"),
  settingEmbeddingModel: $("setting-embedding-model"),
  settingBackend: $("setting-backend"),
  settingMode: $("setting-mode"),
  settingK: $("setting-k"),
  settingRerank: $("setting-rerank"),
  settingLlm: $("setting-llm"),
  settingDebug: $("setting-debug"),
  // Management
  manageToggle: $("manage-toggle"),
  manageModal: $("manage-modal"),
  manageBackdrop: $("manage-backdrop"),
  manageClose: $("manage-close"),
  uploadDropzone: $("upload-dropzone"),
  fileInput: $("file-input"),
  uploadBtn: $("upload-btn"),
  uploadFileList: $("upload-file-list"),
  uploadActions: $("upload-actions"),
  uploadClear: $("upload-clear"),
  uploadStart: $("upload-start"),
  uploadStatus: $("upload-status"),
  refreshDocs: $("refresh-docs"),
  docCount: $("doc-count"),
  docList: $("doc-list"),
  kbStats: $("kb-stats"),
  ingestRebuild: $("ingest-rebuild"),
  ingestForce: $("ingest-force"),
  ingestStart: $("ingest-start"),
  ingestStatus: $("ingest-status"),
};

// ---- Utilities ----
function escapeHtml(value) {
  return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

function formatDate(value) {
  if (!value) return "";
  return new Intl.DateTimeFormat("zh-CN", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(new Date(value));
}

function formatFileSize(bytes) {
  if (!bytes || bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let i = Math.floor(Math.log(bytes) / Math.log(1024));
  if (i > 3) i = 3;
  return (bytes / Math.pow(1024, i)).toFixed(i === 0 ? 0 : 1) + " " + units[i];
}

function normalizedApiBaseUrl() {
  return (state.settings?.apiBaseUrl || "").trim().replace(/\/+$/, "");
}

function apiUrl(path) {
  return `${normalizedApiBaseUrl()}${path}`;
}

async function apiFetch(path, options = {}) {
  const resp = await fetch(apiUrl(path), options);
  if (!resp.ok) {
    const ep = await resp.json().catch(() => ({}));
    throw new Error(ep.detail || `请求失败，状态码 ${resp.status}。`);
  }
  return resp.json();
}

// ---- Markdown ----
const LATEX_PLACEHOLDERS = [];

function _protectLatex(text) {
  LATEX_PLACEHOLDERS.length = 0;
  // Protect display math $$...$$
  let t = text.replace(/\$\$([\s\S]*?)\$\$/g, (_, math) => {
    LATEX_PLACEHOLDERS.push({ type: "display", math: math.trim() });
    return `\x00LATEX${LATEX_PLACEHOLDERS.length - 1}\x00`;
  });
  // Protect inline math $...$ (single $ only, no $ inside)
  t = t.replace(/\$([^$\n]+?)\$/g, (_, math) => {
    LATEX_PLACEHOLDERS.push({ type: "inline", math: math.trim() });
    return `\x00LATEX${LATEX_PLACEHOLDERS.length - 1}\x00`;
  });
  return t;
}

function _restoreLatex(html) {
  return html.replace(/\x00LATEX(\d+)\x00/g, (_, i) => {
    const entry = LATEX_PLACEHOLDERS[Number(i)];
    if (!entry) return "";
    if (entry.type === "display") {
      return `<span class="katex-block">$$${escapeHtml(entry.math)}$$</span>`;
    }
    return `<span class="katex-inline">$${escapeHtml(entry.math)}$</span>`;
  });
}

function renderMarkdown(md) {
  const text = _protectLatex(String(md || ""));
  const lines = text.split(/\r?\n/);
  let html = "";
  const stack = [];

  function closeBlocks(until) {
    while (stack.length > 0) {
      const top = stack[stack.length - 1];
      if (until !== undefined && top === until) break;
      html += { ul: "</ul>", ol: "</ol>", blockquote: "</blockquote>", pre: "</pre>", table: "</tbody></table>" }[top] || "";
      stack.pop();
      if (top === until) break;
    }
  }

  function closeAll() { closeBlocks(); }

  function inlineTokens(src) {
    let s = escapeHtml(src);
    s = s.replace(/\*\*\*(.+?)\*\*\*/g, "<strong><em>$1</em></strong>");
    s = s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/\*(.+?)\*/g, "<em>$1</em>");
    s = s.replace(/`(.+?)`/g, "<code>$1</code>");
    s = s.replace(/\[(E\d+)\]/g, "<code>[$1]</code>");
    return s;
  }

  function parseTableRow(line) {
    const cells = [];
    let cell = "";
    let i = 0;
    while (i < line.length) {
      if (line[i] === "|") {
        cells.push(cell.trim());
        cell = "";
      } else if (line[i] === "\\" && i + 1 < line.length) {
        cell += line[i + 1];
        i++;
      } else {
        cell += line[i];
      }
      i++;
    }
    cells.push(cell.trim());
    // Remove leading empty and trailing empty from split
    return cells.filter((_, idx, arr) => idx > 0 && idx < arr.length - 1);
  }

  function isSepRow(line) { return /^\|[\s\-:|]+\|$/.test(line); }

  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i];
    const line = raw.trim();
    if (!line) { closeAll(); continue; }

    if (line.startsWith("```")) {
      stack.includes("pre") ? closeBlocks("pre") : (closeAll(), html += "<pre>", stack.push("pre"));
      continue;
    }
    if (stack.includes("pre")) { html += escapeHtml(raw) + "\n"; continue; }

    // Table detection
    if (line.startsWith("|") && line.endsWith("|")) {
      if (stack.includes("table")) {
        // Already in a table — body row
        const cells = parseTableRow(line);
        html += "<tr>" + cells.map(c => `<td>${inlineTokens(c)}</td>`).join("") + "</tr>";
        continue;
      }
      // Check if next line is separator
      if (i + 1 < lines.length && isSepRow(lines[i + 1].trim())) {
        closeAll();
        html += '<div class="table-wrap"><table><thead>';
        const hcells = parseTableRow(line);
        html += "<tr>" + hcells.map(c => `<th>${inlineTokens(c)}</th>`).join("") + "</tr></thead><tbody>";
        stack.push("table");
        i++; // skip separator row
        continue;
      }
      // Single-row table (no sep)
      closeAll();
      const scells = parseTableRow(line);
      html += '<div class="table-wrap"><table><tbody><tr>' + scells.map(c => `<td>${inlineTokens(c)}</td>`).join("") + "</tr></tbody></table></div>";
      continue;
    }

    if (line.startsWith("### ")) { closeAll(); html += `<h3>${inlineTokens(line.slice(4))}</h3>`; continue; }
    if (line.startsWith("## ")) { closeAll(); html += `<h2>${inlineTokens(line.slice(3))}</h2>`; continue; }
    if (/^[-*_]{3,}$/.test(line)) { closeAll(); html += "<hr>"; continue; }

    if (line.startsWith("> ")) {
      if (!stack.includes("blockquote")) { closeAll(); html += "<blockquote>"; stack.push("blockquote"); }
      html += `<p>${inlineTokens(line.slice(2))}</p>`; continue;
    }

    const olMatch = line.match(/^(\d+)\.\s+(.+)/);
    if (olMatch) {
      if (!stack.includes("ol")) { closeAll(); html += "<ol>"; stack.push("ol"); }
      html += `<li>${inlineTokens(olMatch[2])}</li>`; continue;
    }

    if (/^[-*]\s+/.test(line)) {
      if (!stack.includes("ul")) { closeAll(); html += "<ul>"; stack.push("ul"); }
      html += `<li>${inlineTokens(line.replace(/^[-*]\s+/, ""))}</li>`; continue;
    }

    if (!stack.includes("ul") && !stack.includes("ol") && !stack.includes("blockquote")) {
      html += `<p>${inlineTokens(line)}</p>`;
    } else {
      closeAll(); html += `<p>${inlineTokens(line)}</p>`;
    }
  }
  closeAll();
  return _restoreLatex(html);
}

// KaTeX render after DOM insert
function renderMathInElement(el) {
  if (typeof renderMathInElement_ !== "undefined") {
    renderMathInElement_(el);
    return;
  }
  // Manual fallback if auto-render not loaded
  el.querySelectorAll(".katex-inline, .katex-block").forEach(span => {
    try {
      const isBlock = span.classList.contains("katex-block");
      const math = span.textContent.replace(/^\$+|\$+$/g, "");
      katex.render(math, span, { displayMode: isBlock, throwOnError: false });
    } catch (_) {}
  });
}

// ---- Settings ----
function mergeSettings(defaults, saved) {
  const d = defaults || {};
  return {
    apiBaseUrl: saved?.apiBaseUrl ?? d.api_base_url ?? "",
    providerBaseUrl: saved?.providerBaseUrl ?? d.provider_base_url ?? "",
    apiKey: saved?.apiKey ?? "",
    llmProvider: saved?.llmProvider ?? d.llm_provider ?? "none",
    llmModel: saved?.llmModel ?? d.llm_model ?? "",
    embeddingModel: saved?.embeddingModel ?? d.embedding_model ?? "",
    backend: saved?.backend ?? d.backend ?? "elastic",
    mode: saved?.mode ?? d.mode ?? "balanced",
    k: Number(saved?.k ?? d.k ?? 6),
    rerank: saved?.rerank ?? d.rerank ?? true,
    llm: saved?.llm ?? d.llm ?? true,
    debug: saved?.debug ?? d.debug ?? false,
  };
}

function loadSavedSettings() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEYS.settings) || "null"); }
  catch { return null; }
}

function saveSettings() {
  localStorage.setItem(STORAGE_KEYS.settings, JSON.stringify(state.settings));
}

function setStatus(message, kind = "idle") {
  els.threadStatus.textContent = message;
  if (kind === "error") { els.backendPill.textContent = "连接异常"; els.backendPill.className = "status-pill error"; }
  else if (kind === "loading") { els.backendPill.textContent = "处理中…"; els.backendPill.className = "status-pill"; }
  else { els.backendPill.textContent = "就绪"; els.backendPill.className = "status-pill"; }
}

function setError(message) { state.error = message; setStatus(message, "error"); }

function populateSelect(node, options, selectedValue) {
  node.innerHTML = options.map(o => `<option value="${escapeHtml(o)}" ${o === selectedValue ? "selected" : ""}>${escapeHtml(o)}</option>`).join("");
}

function syncSettingsForm() {
  const { settings, defaults } = state;
  if (!settings || !defaults) return;
  populateSelect(els.settingLlmProvider, defaults.llm_provider_options, settings.llmProvider);
  populateSelect(els.settingBackend, defaults.backend_options, settings.backend);
  populateSelect(els.settingMode, defaults.mode_options, settings.mode);
  els.settingApiBaseUrl.value = settings.apiBaseUrl;
  els.settingProviderBaseUrl.value = settings.providerBaseUrl;
  els.settingApiKey.value = settings.apiKey;
  els.settingLlmModel.value = settings.llmModel;
  els.settingEmbeddingModel.value = settings.embeddingModel;
  els.settingK.value = settings.k;
  els.settingRerank.checked = settings.rerank;
  els.settingLlm.checked = settings.llm;
  els.settingDebug.checked = settings.debug;
  els.settingsNotes.innerHTML = (defaults.notes || []).map(n => `<div>· ${escapeHtml(n)}</div>`).join("");
}

function renderSettingsSummary() {
  if (!state.settings) return;
  const chips = [
    `检索: ${state.settings.backend}`, `模式: ${state.settings.mode}`, `Top-K: ${state.settings.k}`,
    state.settings.rerank ? "重排序: 开" : "重排序: 关",
    state.settings.llm ? `LLM: ${state.settings.llmModel || "已启用"}` : "仅证据",
  ];
  els.settingsSummary.innerHTML = chips.map(c => `<span class="settings-chip">${escapeHtml(c)}</span>`).join("");
  els.composerNote.textContent = state.settings.debug
    ? "调试模式已开启。" : "回答基于检索文献证据生成，每条结论均可溯源至具体文献。";
}

// ---- Conversations ----
function messageEvidence(message) {
  return {
    citations: message.metadata?.citations || [],
    renderedCitations: message.metadata?.rendered_citations || [],
    sourceCards: message.metadata?.source_cards || [],
    claimItems: message.metadata?.claim_items || [],
    diagnostics: message.metadata?.diagnostics || {},
    retrievalTraceSummary: message.metadata?.retrieval_trace_summary || {},
    evidenceSufficiency: message.metadata?.evidence_sufficiency || "insufficient",
    confidenceScore: message.metadata?.confidence_score ?? 0,
  };
}

function selectedAssistantMessage() {
  if (!state.selectedAssistantMessageId) return null;
  return state.messages.find(m => m.message_id === state.selectedAssistantMessageId) || null;
}

function sufficiencyLabel(v) {
  const m = { insufficient: "证据不足", partial: "部分支撑", sufficient: "证据充分", grounded: "证据充分" };
  return m[v] || v;
}

function getConversationPreview(conversationId) {
  const msgs = state.activeConversationId === conversationId ? state.messages : [];
  const last = [...msgs].reverse().find(m => m.role === "assistant");
  return last ? last.content.replace(/\s+/g, " ").slice(0, 88) : "暂无 AI 回复。";
}

function renderConversationList() {
  els.conversationEmpty.classList.toggle("hidden", state.conversations.length > 0);
  els.conversationList.innerHTML = state.conversations.map(c => {
    const active = c.conversation_id === state.activeConversationId ? "active" : "";
    return `<article class="conversation-item ${active}" data-conversation-id="${escapeHtml(c.conversation_id)}">
      <div class="conversation-title-row">
        <div class="conversation-title">${escapeHtml(c.title)}</div>
        <button class="conversation-delete" type="button" data-delete-id="${escapeHtml(c.conversation_id)}">删除</button>
      </div>
      <p class="conversation-preview">${escapeHtml(getConversationPreview(c.conversation_id))}</p>
      <div class="conversation-meta">${escapeHtml(formatDate(c.updated_at))} · ${escapeHtml(String(c.message_count))} 条消息</div>
    </article>`;
  }).join("");
}

function renderThread() {
  const has = state.messages.length > 0;
  els.welcomePanel.classList.toggle("hidden", has);
  els.threadEmpty.classList.toggle("hidden", has);
  els.thread.innerHTML = state.messages.map(msg => {
    const isAssist = msg.role === "assistant";
    const sel = msg.message_id === state.selectedAssistantMessageId ? "selected" : "";
    const ev = isAssist ? messageEvidence(msg) : null;
    const chips = (ev?.renderedCitations || []).slice(0, 3);
    const badgeHTML = isAssist
      ? `<div class="assistant-meta">
          <span class="mini-badge ${escapeHtml(ev.evidenceSufficiency)}">${escapeHtml(sufficiencyLabel(ev.evidenceSufficiency))}</span>
          <span class="mini-badge">置信度 ${String(Number(ev.confidenceScore || 0).toFixed(2))}</span>
          <span class="conversation-meta">${escapeHtml(formatDate(msg.created_at))}</span>
        </div>`
      : `<div class="assistant-meta"><span class="conversation-meta">${escapeHtml(formatDate(msg.created_at))}</span></div>`;
    const citeRow = isAssist && chips.length ? `<div class="pill-list">${chips.map(c => `<span class="citation-pill">${escapeHtml(c)}</span>`).join("")}</div>` : "";
    return `<article class="message ${escapeHtml(msg.role)} ${sel}" data-message-id="${escapeHtml(msg.message_id)}">
      <div class="avatar">${isAssist ? PIXEL_ICONS.ai : PIXEL_ICONS.user}</div>
      <div class="bubble-wrap">
        <div class="bubble">${isAssist ? renderMarkdown(msg.content) : `<p>${escapeHtml(msg.content)}</p>`}</div>
        <div class="assistant-footer">${badgeHTML}${citeRow}</div>
      </div>
    </article>`;
  }).join("");
  els.typingIndicator.classList.toggle("hidden", !state.sending);
  renderMathInElement(els.thread);
}

function renderInspector() {
  const msg = selectedAssistantMessage();
  if (!msg) {
    els.inspectorEmpty.classList.remove("hidden");
    els.inspectorContent.classList.add("hidden");
    els.inspectorCaption.textContent = "点击某条 AI 回复以查看其支撑证据和检索详情。";
    els.inspectorBadges.innerHTML = "";
    return;
  }
  const ev = messageEvidence(msg);
  const summary = ev.retrievalTraceSummary || {};
  els.inspectorEmpty.classList.add("hidden");
  els.inspectorContent.classList.remove("hidden");
  els.inspectorCaption.textContent = summary.retrieval_query ? `检索语句: ${summary.retrieval_query}` : "查看所选 AI 回复的证据详情。";
  els.inspectorBadges.innerHTML = `
    <span class="mini-badge ${escapeHtml(ev.evidenceSufficiency)}">${escapeHtml(sufficiencyLabel(ev.evidenceSufficiency))}</span>
    <span class="mini-badge">置信度 ${String(Number(ev.confidenceScore || 0).toFixed(2))}</span>
    <span class="mini-badge">${String((ev.citations || []).length)} 条证据</span>`;

  const claims = (ev.claimItems || []).slice(0, 10);
  els.claimList.innerHTML = claims.map(c => {
    const labels = (c.citations || []).map(l => `[${l}]`).join(" ");
    const sup = c.support === "direct" ? "直接支撑" : c.support === "contextual" ? "上下文引用" : (c.support || "支撑");
    return `<article class="claim-card">
      <div class="claim-support">${escapeHtml(c.claim_id || "声明")} · ${escapeHtml(sup)}</div>
      <p class="evidence-snippet">${renderMarkdown(c.text || "")}</p>
      ${labels ? `<div class="pill-list"><span class="pill">${escapeHtml(labels)}</span></div>` : ""}
    </article>`;
  }).join("");

  els.citationReferenceList.innerHTML = (ev.renderedCitations || []).map(c => `<span class="pill">${escapeHtml(c)}</span>`).join("");

  els.evidenceList.innerHTML = (ev.citations || []).map(item => {
    const page = item.page != null ? `第 ${item.page} 页` : "页码未知";
    const heading = item.heading_path || "章节未知";
    return `<article class="evidence-card">
      <div class="evidence-header">
        <div class="evidence-label">[${escapeHtml(item.label)}]</div>
        <div class="evidence-source"><strong>${escapeHtml(item.source_file || item.title || "未知来源")}</strong><br/>${escapeHtml(page)} · ${escapeHtml(heading)}</div>
      </div>
      <p class="evidence-snippet">${escapeHtml(item.snippet || "无摘要。")}</p>
    </article>`;
  }).join("");

  els.sourceCardList.innerHTML = (ev.sourceCards || []).map(card => {
    const pages = (card.pages || []).join(", ") || "未知";
    const headings = (card.headings || []).join(", ") || "未知";
    const labels = (card.evidence_labels || []).map(l => `[${l}]`).join(" ");
    return `<article class="source-card">
      <div class="source-card-header">
        <div><h3>${escapeHtml(card.title || card.source_file || "未知来源")}</h3>
          <div class="source-card-meta">${escapeHtml(card.source_file || "未知文件")} · 页码 ${escapeHtml(pages)}</div>
          <div class="source-card-meta">${escapeHtml(headings)}</div></div>
        <div class="mini-badge">${String(card.evidence_count || 0)} 条片段</div>
      </div>
      <p class="source-card-preview">${escapeHtml(card.snippet_preview || "无预览。")}</p>
      ${labels ? `<div class="pill-list"><span class="pill">${escapeHtml(labels)}</span></div>` : ""}
    </article>`;
  }).join("");
  renderMathInElement(els.inspectorContent);
}

function renderAll() { renderSettingsSummary(); renderConversationList(); renderThread(); renderInspector(); }

// ---- API ----
async function loadSettingsDefaults() {
  state.defaults = await apiFetch("/api/settings/defaults");
  state.settings = mergeSettings(state.defaults, loadSavedSettings());
  syncSettingsForm(); renderSettingsSummary();
}

async function refreshConversations() {
  state.conversations = await apiFetch("/api/conversations");
  if (state.activeConversationId && !state.conversations.some(c => c.conversation_id === state.activeConversationId)) {
    state.activeConversationId = null; state.messages = []; state.selectedAssistantMessageId = null;
  }
}

function chooseAssistantSelection() {
  const last = [...state.messages].reverse().find(m => m.role === "assistant");
  state.selectedAssistantMessageId = last?.message_id || null;
}

async function loadMessages(cid) {
  if (!cid) { state.messages = []; state.selectedAssistantMessageId = null; renderAll(); return; }
  setStatus("正在加载对话…", "loading");
  state.messages = await apiFetch(`/api/conversations/${cid}/messages`);
  chooseAssistantSelection(); setStatus("对话已加载。", "idle"); renderAll();
}

async function openConversation(cid) {
  if (!cid) { state.activeConversationId = null; state.messages = []; state.selectedAssistantMessageId = null; setStatus("已新建对话。", "idle"); renderAll(); return; }
  state.activeConversationId = cid; await loadMessages(cid);
}

async function deleteConversation(cid) {
  const resp = await fetch(apiUrl(`/api/conversations/${cid}`), { method: "DELETE" });
  if (!resp.ok) { const p = await resp.json().catch(() => ({})); throw new Error(p.detail || "删除失败。"); }
  if (state.activeConversationId === cid) { state.activeConversationId = null; state.messages = []; state.selectedAssistantMessageId = null; }
  await refreshConversations(); renderAll(); setStatus("对话已删除。", "idle");
}

function buildChatPayload(message) {
  const s = state.settings || {};
  return {
    conversation_id: state.activeConversationId, message,
    llm: s.llm ?? true, debug: s.debug ?? false,
    backend: s.backend || "elastic", mode: s.mode || "broad", k: Number(s.k ?? 10), rerank: s.rerank ?? true,
    llm_provider: s.llmProvider || "none", api_key: s.apiKey || null,
    provider_base_url: s.providerBaseUrl || null, llm_model: s.llmModel || null, embedding_model: s.embeddingModel || null,
  };
}

async function sendMessage(event) {
  event.preventDefault();
  const message = els.messageInput.value.trim();
  if (!message || state.sending) return;
  state.sending = true; els.sendButton.disabled = true;
  setStatus("正在检索文献并生成回答…", "loading"); renderAll();
  try {
    const payload = await apiFetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(buildChatPayload(message)) });
    state.activeConversationId = payload.conversation_id;
    state.messages = [...state.messages,
      { message_id: `local-user-${Date.now()}`, conversation_id: payload.conversation_id, role: "user", content: message, metadata: {}, created_at: new Date().toISOString() },
      { message_id: payload.message_id, conversation_id: payload.conversation_id, role: "assistant", content: payload.answer,
        metadata: { citations: payload.citations || [], rendered_citations: payload.rendered_citations || [], source_cards: payload.source_cards || [], claim_items: payload.claim_items || [], diagnostics: payload.diagnostics || {}, retrieval_trace_summary: payload.retrieval_trace_summary || {}, evidence_sufficiency: payload.evidence_sufficiency || "insufficient", confidence_score: payload.confidence_score ?? 0 },
        created_at: new Date().toISOString() },
    ];
    state.selectedAssistantMessageId = payload.message_id;
    els.messageInput.value = "";
    await refreshConversations();
    setStatus(`回答完成，耗时 ${Number(payload.timings_ms?.total || 0).toFixed(0)} ms。`, "idle");
  } catch (e) { setError(e.message || "发送失败。"); }
  finally { state.sending = false; els.sendButton.disabled = false; renderAll(); }
}

function handleThreadClick(event) {
  const node = event.target.closest("[data-message-id]");
  if (!node) return;
  const msg = state.messages.find(m => m.message_id === node.getAttribute("data-message-id"));
  if (msg?.role !== "assistant") return;
  state.selectedAssistantMessageId = msg.message_id; renderAll();
}

function handleConversationClick(event) {
  const del = event.target.closest("[data-delete-id]");
  if (del) { deleteConversation(del.getAttribute("data-delete-id")).catch(e => setError(e.message)); return; }
  const node = event.target.closest("[data-conversation-id]");
  if (!node) return;
  openConversation(node.getAttribute("data-conversation-id")).catch(e => setError(e.message));
}

function applySample(text) { els.messageInput.value = text; els.messageInput.focus(); }

// ===== Management Modal =====
function openManage() {
  els.manageModal.classList.remove("hidden");
  els.manageModal.setAttribute("aria-hidden", "false");
  switchManageTab("tab-upload");
}

function closeManage() {
  els.manageModal.classList.add("hidden");
  els.manageModal.setAttribute("aria-hidden", "true");
}

function switchManageTab(tabId) {
  state.manageTab = tabId;
  document.querySelectorAll(".manage-tab").forEach(t => t.classList.toggle("active", t.dataset.tab === tabId));
  document.querySelectorAll(".manage-tab-panel").forEach(p => p.classList.toggle("hidden", p.id !== tabId));

  if (tabId === "tab-stats") loadKBStats();
  if (tabId === "tab-docs") loadDocumentList();
}

function handleManageTabClick(e) {
  const tab = e.target.closest(".manage-tab");
  if (!tab) return;
  switchManageTab(tab.dataset.tab);
}

// ---- File Upload ----
function updateUploadUI() {
  if (state.selectedFiles.length === 0) {
    els.uploadFileList.classList.add("hidden");
    els.uploadActions.classList.add("hidden");
    els.uploadFileList.innerHTML = "";
    return;
  }
  els.uploadFileList.classList.remove("hidden");
  els.uploadActions.classList.remove("hidden");
  els.uploadFileList.innerHTML = state.selectedFiles.map((f, i) =>
    `<div class="upload-file-item">
      <span class="file-name">${escapeHtml(f.name)}</span>
      <span class="file-size">${formatFileSize(f.size)}</span>
      <button class="file-remove" type="button" data-file-index="${i}">移除</button>
    </div>`
  ).join("");
  // Bind remove buttons
  els.uploadFileList.querySelectorAll(".file-remove").forEach(btn => {
    btn.addEventListener("click", () => {
      const idx = Number(btn.dataset.fileIndex);
      state.selectedFiles.splice(idx, 1);
      updateUploadUI();
    });
  });
}

function addFiles(files) {
  for (const f of files) {
    if (!state.selectedFiles.some(existing => existing.name === f.name && existing.size === f.size)) {
      state.selectedFiles.push(f);
    }
  }
  updateUploadUI();
}

async function startUpload() {
  if (state.selectedFiles.length === 0) return;
  const formData = new FormData();
  state.selectedFiles.forEach(f => formData.append("files", f));

  els.uploadStatus.className = "upload-status loading";
  els.uploadStatus.innerHTML = "正在上传…";
  els.uploadStart.disabled = true;

  try {
    const resp = await fetch(apiUrl("/api/admin/documents/upload"), { method: "POST", body: formData });
    const data = await resp.json();
    els.uploadStatus.className = "upload-status " + (data.errors && data.errors.length ? "error" : "success");
    els.uploadStatus.innerHTML = escapeHtml(data.message)
      + (data.saved && data.saved.length ? `<br/>已保存: ${data.saved.map(escapeHtml).join(", ")}` : "")
      + (data.errors && data.errors.length ? `<br/>失败: ${data.errors.map(escapeHtml).join(", ")}` : "");
    if (data.saved && data.saved.length) { state.selectedFiles = []; updateUploadUI(); }
  } catch (e) {
    els.uploadStatus.className = "upload-status error";
    els.uploadStatus.textContent = "上传失败: " + (e.message || "未知错误");
  } finally {
    els.uploadStart.disabled = false;
  }
}

// ---- Document List ----
async function loadDocumentList() {
  els.docList.innerHTML = `<div class="empty-card">加载中…</div>`;
  try {
    const docs = await apiFetch("/api/admin/documents");
    els.docCount.textContent = `共 ${docs.length} 个文档`;
    if (docs.length === 0) { els.docList.innerHTML = `<div class="empty-card">暂无已索引文档。</div>`; return; }
    els.docList.innerHTML = docs.map(d =>
      `<div class="doc-item">
        <div class="doc-info">
          <div class="doc-name">${escapeHtml(d.title || d.source_file)}</div>
          <div class="doc-meta">${escapeHtml(d.source_file)} · ${formatFileSize(d.size)} · ${escapeHtml(d.doc_type || "未知类型")} · ${escapeHtml(d.status)}</div>
        </div>
        <button class="doc-delete" type="button" data-doc-id="${escapeHtml(d.document_id)}">删除</button>
      </div>`
    ).join("");
    // Bind delete buttons
    els.docList.querySelectorAll(".doc-delete").forEach(btn => {
      btn.addEventListener("click", async () => {
        const docId = btn.dataset.docId;
        if (!confirm(`确定要删除文档 "${docId}" 及其所有分块数据吗？`)) return;
        try {
          await apiFetch(`/api/admin/documents/${docId}`, { method: "DELETE" });
          loadDocumentList();
          loadKBStats();
        } catch (e) { alert("删除失败: " + (e.message || "未知错误")); }
      });
    });
  } catch (e) { els.docList.innerHTML = `<div class="empty-card">加载失败: ${escapeHtml(e.message || "")}</div>`; }
}

// ---- KB Stats ----
async function loadKBStats() {
  els.kbStats.innerHTML = `<div class="empty-card">加载中…</div>`;
  try {
    const stats = await apiFetch("/api/admin/stats");
    els.kbStats.innerHTML = `
      <div class="stats-grid">
        <div class="stat-card"><div class="stat-value">${stats.total_documents}</div><div class="stat-label">文档总数</div></div>
        <div class="stat-card"><div class="stat-value">${stats.active_documents}</div><div class="stat-label">活跃文档</div></div>
        <div class="stat-card"><div class="stat-value">${stats.total_chunks}</div><div class="stat-label">分块总数</div></div>
      </div>
      <div class="stat-extra">
        <strong>检索后端：</strong>${escapeHtml(stats.backend || "未知")}<br/>
        <strong>嵌入模型：</strong>${escapeHtml(stats.embedding_model || "未知")}<br/>
        <strong>最近摄入：</strong>${escapeHtml(stats.last_ingestion || "暂无记录")}
      </div>`;
  } catch (e) { els.kbStats.innerHTML = `<div class="empty-card">加载失败: ${escapeHtml(e.message || "")}</div>`; }
}

// ---- Ingestion ----
async function startIngestion() {
  const rebuild = els.ingestRebuild.checked;
  const force = els.ingestForce.checked;
  els.ingestStatus.className = "ingest-status loading";
  els.ingestStatus.textContent = "摄入任务已提交，正在处理中（可能需要几分钟）…";
  els.ingestStart.disabled = true;

  try {
    const data = await apiFetch(`/api/admin/ingest?rebuild=${rebuild}&force=${force}`, { method: "POST" });
    els.ingestStatus.className = "ingest-status " + (data.status === "completed" ? "success" : "error");
    els.ingestStatus.innerHTML = `<strong>${escapeHtml(data.message)}</strong>`
      + (data.stats ? `<br/>文档数: ${data.stats.documents || "?"} · 分块数: ${data.stats.chunks || "?"}` : "");
    loadKBStats();
    loadDocumentList();
  } catch (e) {
    els.ingestStatus.className = "ingest-status error";
    els.ingestStatus.textContent = "摄入失败: " + (e.message || "未知错误");
  } finally {
    els.ingestStart.disabled = false;
  }
}

// ===== Settings Modal =====
function openSettings() { syncSettingsForm(); els.settingsModal.classList.remove("hidden"); els.settingsModal.setAttribute("aria-hidden", "false"); }
function closeSettings() { els.settingsModal.classList.add("hidden"); els.settingsModal.setAttribute("aria-hidden", "true"); }
function resetSettings() { if (!state.defaults) return; state.settings = mergeSettings(state.defaults, null); saveSettings(); syncSettingsForm(); renderSettingsSummary(); }
function saveSettingsFromForm(event) {
  event.preventDefault();
  state.settings = {
    apiBaseUrl: els.settingApiBaseUrl.value.trim(), providerBaseUrl: els.settingProviderBaseUrl.value.trim(), apiKey: els.settingApiKey.value.trim(),
    llmProvider: els.settingLlmProvider.value, llmModel: els.settingLlmModel.value.trim(), embeddingModel: els.settingEmbeddingModel.value.trim(),
    backend: els.settingBackend.value, mode: els.settingMode.value, k: Math.max(1, Number(els.settingK.value) || 6),
    rerank: els.settingRerank.checked, llm: els.settingLlm.checked, debug: els.settingDebug.checked,
  };
  saveSettings(); renderSettingsSummary(); closeSettings();
  setStatus("设置已保存。", "idle");
}

// ===== Bootstrap =====
async function bootstrap() {
  setStatus("正在连接后端服务…", "loading");
  try {
    await apiFetch("/health");
    await loadSettingsDefaults();
    await refreshConversations();
    if (state.conversations.length) { await openConversation(state.conversations[0].conversation_id); }
    else { renderAll(); setStatus("暂无历史对话，请新建对话开始提问。", "idle"); }
  } catch (e) { setError(e.message || "初始化失败。"); }
}

// ---- Event Bindings ----
els.form.addEventListener("submit", sendMessage);
els.thread.addEventListener("click", handleThreadClick);
els.conversationList.addEventListener("click", handleConversationClick);
els.newChatTop.addEventListener("click", () => openConversation(null));
els.newChatSidebar.addEventListener("click", () => openConversation(null));
els.refreshConversations.addEventListener("click", () => { refreshConversations().then(renderAll).catch(e => setError(e.message)); });
els.sampleQuestion?.addEventListener("click", () => applySample("蒸汽气化在1200K附近如何影响NH3和HCN的生成？"));
document.querySelectorAll(".sample-chip").forEach(chip => chip.addEventListener("click", () => applySample(chip.dataset.sample || "")));
// Settings
els.settingsToggle.addEventListener("click", openSettings);
els.settingsClose.addEventListener("click", closeSettings);
els.settingsBackdrop.addEventListener("click", closeSettings);
els.settingsReset.addEventListener("click", resetSettings);
els.settingsForm.addEventListener("submit", saveSettingsFromForm);
// Management
els.manageToggle.addEventListener("click", openManage);
els.manageClose.addEventListener("click", closeManage);
els.manageBackdrop.addEventListener("click", closeManage);
document.querySelectorAll(".manage-tab").forEach(t => t.addEventListener("click", handleManageTabClick));
els.manageModal.querySelector(".manage-tabs").addEventListener("click", handleManageTabClick);
// Upload
els.uploadDropzone.addEventListener("click", () => els.fileInput.click());
els.uploadBtn.addEventListener("click", (e) => { e.stopPropagation(); els.fileInput.click(); });
els.fileInput.addEventListener("change", () => { if (els.fileInput.files.length) addFiles(els.fileInput.files); els.fileInput.value = ""; });
els.uploadDropzone.addEventListener("dragover", (e) => { e.preventDefault(); els.uploadDropzone.classList.add("drag-over"); });
els.uploadDropzone.addEventListener("dragleave", () => els.uploadDropzone.classList.remove("drag-over"));
els.uploadDropzone.addEventListener("drop", (e) => { e.preventDefault(); els.uploadDropzone.classList.remove("drag-over"); if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files); });
els.uploadClear.addEventListener("click", () => { state.selectedFiles = []; updateUploadUI(); });
els.uploadStart.addEventListener("click", startUpload);
// Docs
els.refreshDocs.addEventListener("click", loadDocumentList);
// Ingest
els.ingestStart.addEventListener("click", startIngestion);

bootstrap();
