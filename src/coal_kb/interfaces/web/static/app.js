"use strict";

const STORAGE_KEYS = {
  settings: "coal-kb-ui-settings",
  secrets: "coal-kb-ui-session-secrets",
};

const state = {
  defaults: null,
  settings: null,
  conversations: [],
  conversationsLoading: false,
  activeConversationId: null,
  messages: [],
  selectedAssistantMessageId: null,
  selectedFiles: [],
  sending: false,
};

const $ = (id) => document.getElementById(id);
const els = {
  backendPill: $("backend-pill"), threadStatus: $("thread-status"), conversationStatus: $("conversation-status"),
  conversationEmpty: $("conversation-empty"), conversationList: $("conversation-list"), welcomePanel: $("welcome-panel"),
  threadEmpty: $("thread-empty"), thread: $("message-thread"), typingIndicator: $("typing-indicator"),
  form: $("chat-form"), messageInput: $("message-input"), composerNote: $("composer-note"), sendButton: $("send-button"),
  settingsSummary: $("settings-summary"), inspectorEmpty: $("inspector-empty"), inspectorContent: $("inspector-content"),
  inspectorCaption: $("inspector-caption"), inspectorBadges: $("inspector-badges"), claimList: $("claim-list"),
  citationReferenceList: $("citation-reference-list"), evidenceList: $("evidence-list"), sourceCardList: $("source-card-list"),
  newChatTop: $("new-chat-top"), newChatSidebar: $("new-chat-sidebar"), refreshConversations: $("refresh-conversations"),
  settingsToggle: $("settings-toggle"), settingsModal: $("settings-modal"), settingsBackdrop: $("settings-backdrop"),
  settingsClose: $("settings-close"), settingsForm: $("settings-form"), settingsReset: $("settings-reset"),
  settingsSave: $("settings-save"), settingsNotes: $("settings-notes"), providerEffectiveSummary: $("provider-effective-summary"),
  settingApiBaseUrl: $("setting-api-base-url"), settingBackend: $("setting-backend"), settingMode: $("setting-mode"),
  settingResearchRoute: $("setting-research-route"), settingK: $("setting-k"), settingRerank: $("setting-rerank"),
  settingLlm: $("setting-llm"), settingDebug: $("setting-debug"), manageToggle: $("manage-toggle"),
  manageModal: $("manage-modal"), manageBackdrop: $("manage-backdrop"), manageClose: $("manage-close"),
  uploadDropzone: $("upload-dropzone"), fileInput: $("file-input"), uploadBtn: $("upload-btn"),
  uploadSummary: $("upload-summary"), uploadFileList: $("upload-file-list"), uploadActions: $("upload-actions"),
  uploadClear: $("upload-clear"), uploadStart: $("upload-start"), uploadAutoIngest: $("upload-auto-ingest"),
  uploadStatus: $("upload-status"), uploadProgress: $("upload-progress"), uploadProgressBar: $("upload-progress-bar"),
  uploadProgressLabel: $("upload-progress-label"), refreshDocs: $("refresh-docs"), docCount: $("doc-count"),
  docList: $("doc-list"), kbStats: $("kb-stats"), ingestRebuild: $("ingest-rebuild"), ingestForce: $("ingest-force"),
  ingestStart: $("ingest-start"), ingestStatus: $("ingest-status"), toastRegion: $("toast-region"),
};

const CAPABILITIES = {
  tokenizer: {
    label: "Tokenizer", defaultsKey: "tokenizer", mode: $("setting-tokenizer-mode"),
    provider: $("setting-tokenizer-provider"), baseUrl: $("setting-tokenizer-base-url"),
    apiKey: $("setting-tokenizer-api-key"), model: $("setting-tokenizer-model"), status: $("provider-status-tokenizer"),
  },
  embeddings: {
    label: "Embeddings", defaultsKey: "embeddings", mode: $("setting-embedding-mode"),
    provider: $("setting-embedding-provider"), baseUrl: $("setting-embedding-base-url"),
    apiKey: $("setting-embedding-api-key"), model: $("setting-embedding-model"), status: $("provider-status-embeddings"),
  },
  rerank: {
    label: "Rerank", defaultsKey: "rerank_config", mode: $("setting-rerank-mode"),
    provider: $("setting-rerank-provider"), baseUrl: $("setting-rerank-base-url"),
    apiKey: $("setting-rerank-api-key"), model: $("setting-rerank-model"), status: $("provider-status-rerank"),
  },
  llm: {
    label: "LLM", defaultsKey: "llm_config", mode: $("setting-llm-mode"),
    provider: $("setting-llm-provider"), baseUrl: $("setting-llm-base-url"),
    apiKey: $("setting-llm-api-key"), model: $("setting-llm-model"), status: $("provider-status-llm"),
  },
};

const ROUTE_LABELS = {
  standard: "Standard",
  graph: "Graph",
  multimodal: "Multimodal",
  agent: "Agent",
};

const FILE_STATE_LABELS = {
  ready: "待上传",
  uploading: "上传中",
  queued: "等待处理",
  saving: "保存中",
  indexing: "写入索引",
  completed: "可检索",
  saved: "已保存",
  failed: "失败",
};

// 通用交互工具
function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function formatDate(value) {
  if (!value) return "";
  return new Intl.DateTimeFormat("zh-CN", {
    month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit",
  }).format(new Date(value));
}

function formatFileSize(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(3, Math.floor(Math.log(bytes) / Math.log(1024)));
  return `${(bytes / (1024 ** index)).toFixed(index ? 1 : 0)} ${units[index]}`;
}

function normalizedBase(value = state.settings?.apiBaseUrl || "") {
  return String(value).trim().replace(/\/+$/, "");
}

function apiUrl(path, base) { return `${normalizedBase(base)}${path}`; }

async function apiFetch(path, options = {}, base) {
  const response = await fetch(apiUrl(path, base), options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `请求失败，状态码 ${response.status}。`);
  return payload;
}

function setStatus(message, kind = "idle") {
  els.threadStatus.textContent = message;
  const labels = { error: "连接异常", loading: "处理中…", idle: "就绪" };
  els.backendPill.textContent = labels[kind] || labels.idle;
  els.backendPill.className = `status-pill${kind === "error" ? " error" : ""}`;
}

function showToast(message, kind = "info") {
  const node = document.createElement("div");
  node.className = `toast ${kind}`;
  node.textContent = message;
  els.toastRegion.appendChild(node);
  window.setTimeout(() => node.remove(), 4200);
}

function setButtonBusy(button, busy, busyText) {
  if (!button) return;
  if (!button.dataset.originalText) button.dataset.originalText = button.textContent;
  button.disabled = busy;
  button.setAttribute("aria-busy", String(busy));
  button.textContent = busy ? busyText : button.dataset.originalText;
}

function renderMarkdown(value) {
  let html = escapeHtml(value || "");
  html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
  html = html.replace(/^## (.+)$/gm, "<h2>$1</h2>");
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/`(.+?)`/g, "<code>$1</code>");
  html = html.replace(/\n\n/g, "</p><p>").replace(/\n/g, "<br/>");
  return `<p>${html}</p>`;
}

function loadJson(storage, key) {
  try { return JSON.parse(storage.getItem(key) || "null"); } catch (_) { return null; }
}

// Provider 与运行配置
function saveBrowserSettings() {
  const safe = structuredClone(state.settings);
  const secrets = {};
  Object.keys(CAPABILITIES).forEach((capability) => {
    const key = `${capability}ApiKey`;
    secrets[key] = safe[key] || "";
    delete safe[key];
  });
  localStorage.setItem(STORAGE_KEYS.settings, JSON.stringify(safe));
  sessionStorage.setItem(STORAGE_KEYS.secrets, JSON.stringify(secrets));
}

function activeCapability(defaults, capability) {
  const block = defaults[CAPABILITIES[capability].defaultsKey];
  const mode = block.mode;
  return { mode, ...block[mode] };
}

function mergeSettings(defaults) {
  const saved = loadJson(localStorage, STORAGE_KEYS.settings) || {};
  const secrets = loadJson(sessionStorage, STORAGE_KEYS.secrets) || {};
  const merged = {
    apiBaseUrl: saved.apiBaseUrl ?? defaults.api_base_url ?? "",
    backend: saved.backend ?? defaults.backend,
    retrievalMode: saved.retrievalMode ?? defaults.mode,
    researchRoute: saved.researchRoute ?? defaults.research_route ?? "standard",
    k: Number(saved.k ?? defaults.k),
    rerank: saved.rerank ?? defaults.rerank,
    llm: saved.llm ?? defaults.llm,
    debug: saved.debug ?? defaults.debug,
  };
  Object.keys(CAPABILITIES).forEach((capability) => {
    const current = activeCapability(defaults, capability);
    merged[`${capability}Mode`] = saved[`${capability}Mode`] ?? current.mode;
    merged[`${capability}Provider`] = saved[`${capability}Provider`] ?? current.provider ?? "";
    merged[`${capability}BaseUrl`] = saved[`${capability}BaseUrl`] ?? current.base_url ?? "";
    merged[`${capability}Model`] = saved[`${capability}Model`] ?? current.model ?? "";
    merged[`${capability}ApiKey`] = secrets[`${capability}ApiKey`] ?? "";
  });
  return merged;
}

function populateSelect(node, options, selected) {
  const values = Array.from(new Set([...(options || []), selected].filter(Boolean)));
  node.innerHTML = values.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join("");
  node.value = selected || values[0] || "";
}

function providerOptions(capability, mode) {
  return state.defaults?.provider_options?.[capability]?.[mode] || [];
}

function capabilityEnabled(capability, settings = state.settings) {
  if (capability === "llm") return Boolean(settings.llm);
  if (capability === "rerank") return Boolean(settings.rerank);
  return true;
}

function serverHasApiKey(capability) {
  return Boolean(state.defaults?.[CAPABILITIES[capability].defaultsKey]?.remote?.api_key_configured);
}

function capabilityState(capability, settings = state.settings) {
  const enabled = capabilityEnabled(capability, settings);
  if (!enabled) return { kind: "disabled", text: "未启用", missing: [] };
  const mode = settings[`${capability}Mode`];
  const provider = String(settings[`${capability}Provider`] || "").trim();
  const model = String(settings[`${capability}Model`] || "").trim();
  const apiKey = String(settings[`${capability}ApiKey`] || "").trim();
  const missing = [];
  if (!provider) missing.push("Provider");
  if (!model) missing.push("模型");
  if (mode === "remote" && !apiKey && !serverHasApiKey(capability)) missing.push("API Key");
  if (missing.length) return { kind: "warning", text: `缺少 ${missing.join(" / ")}`, missing };
  return { kind: "ready", text: `${mode === "remote" ? "远程" : "本地"}配置完整`, missing: [] };
}

function renderCapabilityStatus(capability) {
  const control = CAPABILITIES[capability];
  const status = capabilityState(capability);
  control.status.className = `provider-status ${status.kind}`;
  control.status.textContent = status.text;
  control.status.closest(".provider-card").dataset.state = status.kind;
  control.apiKey.disabled = state.settings[`${capability}Mode`] !== "remote";
  control.apiKey.placeholder = control.apiKey.disabled
    ? "本地模式不使用远程密钥"
    : serverHasApiKey(capability) && !control.apiKey.value
      ? "服务端已配置，留空保持不变"
      : "仅保存于当前会话";
}

function renderProviderSummary() {
  const active = Object.keys(CAPABILITIES).filter((capability) => capabilityEnabled(capability));
  const ready = active.filter((capability) => capabilityState(capability).kind === "ready");
  const warnings = active.length - ready.length;
  els.providerEffectiveSummary.className = `configuration-banner${warnings ? " warning" : ""}`;
  els.providerEffectiveSummary.innerHTML = warnings
    ? `<span><strong>${ready.length}/${active.length}</strong> 项已配置完整，请补齐黄色 Provider 卡片。</span><span>保存前检查</span>`
    : `<span><strong>${active.length}/${active.length}</strong> 项配置完整，将用于后续问答和增量入库。</span><span>未执行网络探测</span>`;
}

function syncCapability(capability) {
  const control = CAPABILITIES[capability];
  const mode = state.settings[`${capability}Mode`];
  control.mode.value = mode;
  populateSelect(control.provider, providerOptions(capability, mode), state.settings[`${capability}Provider`]);
  control.baseUrl.value = state.settings[`${capability}BaseUrl`] || "";
  control.model.value = state.settings[`${capability}Model`] || "";
  control.apiKey.value = state.settings[`${capability}ApiKey`] || "";
  renderCapabilityStatus(capability);
}

function syncSettingsForm() {
  populateSelect(els.settingBackend, state.defaults.backend_options, state.settings.backend);
  populateSelect(els.settingMode, state.defaults.mode_options, state.settings.retrievalMode);
  populateSelect(els.settingResearchRoute, state.defaults.research_route_options, state.settings.researchRoute);
  els.settingApiBaseUrl.value = state.settings.apiBaseUrl;
  els.settingK.value = state.settings.k;
  els.settingRerank.checked = state.settings.rerank;
  els.settingLlm.checked = state.settings.llm;
  els.settingDebug.checked = state.settings.debug;
  Object.keys(CAPABILITIES).forEach(syncCapability);
  renderProviderSummary();
  els.settingsNotes.innerHTML = (state.defaults.notes || []).map((note) => `<div>· ${escapeHtml(note)}</div>`).join("");
}

function collectSettings() {
  const next = {
    apiBaseUrl: els.settingApiBaseUrl.value.trim(),
    backend: els.settingBackend.value,
    retrievalMode: els.settingMode.value,
    researchRoute: els.settingResearchRoute.value,
    k: Math.max(1, Number(els.settingK.value) || 5),
    rerank: els.settingRerank.checked,
    llm: els.settingLlm.checked,
    debug: els.settingDebug.checked,
  };
  Object.entries(CAPABILITIES).forEach(([capability, control]) => {
    next[`${capability}Mode`] = control.mode.value;
    next[`${capability}Provider`] = control.provider.value;
    next[`${capability}BaseUrl`] = control.baseUrl.value.trim();
    next[`${capability}Model`] = control.model.value.trim();
    next[`${capability}ApiKey`] = control.apiKey.value.trim();
  });
  return next;
}

function runtimePayload(settings = state.settings) {
  return {
    llm: settings.llm,
    debug: settings.debug,
    backend: settings.backend,
    mode: settings.retrievalMode,
    research_route: settings.researchRoute,
    k: Number(settings.k),
    rerank: settings.rerank,
    tokenizer_mode: settings.tokenizerMode,
    tokenizer_provider: settings.tokenizerProvider,
    tokenizer_base_url: settings.tokenizerBaseUrl || null,
    tokenizer_api_key: settings.tokenizerApiKey || null,
    tokenizer_model: settings.tokenizerModel,
    embedding_mode: settings.embeddingsMode,
    embedding_provider: settings.embeddingsProvider,
    embedding_base_url: settings.embeddingsBaseUrl || null,
    embedding_api_key: settings.embeddingsApiKey || null,
    embedding_model: settings.embeddingsModel,
    rerank_mode: settings.rerankMode,
    rerank_provider: settings.rerankProvider,
    rerank_base_url: settings.rerankBaseUrl || null,
    rerank_api_key: settings.rerankApiKey || null,
    rerank_model: settings.rerankModel,
    llm_mode: settings.llmMode,
    llm_provider: settings.llm ? settings.llmProvider : "none",
    llm_base_url: settings.llmBaseUrl || null,
    llm_api_key: settings.llmApiKey || null,
    llm_model: settings.llmModel,
  };
}

function renderSettingsSummary() {
  const settings = state.settings;
  if (!settings) return;
  const chips = [
    `后端: ${settings.backend}`,
    `路线: ${ROUTE_LABELS[settings.researchRoute] || settings.researchRoute}`,
    `Top-K: ${settings.k}`,
    `Embedding: ${settings.embeddingsProvider}/${settings.embeddingsModel}`,
    settings.rerank ? `Rerank: ${settings.rerankProvider}` : "Rerank: 关",
    settings.llm ? `LLM: ${settings.llmProvider}/${settings.llmModel}` : "仅证据",
  ];
  els.settingsSummary.innerHTML = chips.map((chip) => `<span class="settings-chip">${escapeHtml(chip)}</span>`).join("");
  els.composerNote.textContent = settings.debug
    ? `诊断模式已开启 · ${ROUTE_LABELS[settings.researchRoute] || settings.researchRoute} 路线`
    : `当前使用 ${ROUTE_LABELS[settings.researchRoute] || settings.researchRoute} 路线，回答基于可溯源证据。`;
}

function validateSettings(next) {
  const missing = [];
  Object.keys(CAPABILITIES).forEach((capability) => {
    const previous = state.settings;
    state.settings = next;
    const status = capabilityState(capability, next);
    state.settings = previous;
    if (status.kind === "warning") missing.push(`${CAPABILITIES[capability].label}: ${status.text}`);
  });
  return missing;
}

async function saveSettings(event) {
  event.preventDefault();
  const next = collectSettings();
  const missing = validateSettings(next);
  if (missing.length) {
    showToast(`请先补齐配置：${missing.join("；")}`, "error");
    state.settings = next;
    Object.keys(CAPABILITIES).forEach(renderCapabilityStatus);
    renderProviderSummary();
    return;
  }
  setButtonBusy(els.settingsSave, true, "正在应用…");
  try {
    const response = await apiFetch(
      "/api/settings/runtime",
      { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(runtimePayload(next)) },
      next.apiBaseUrl,
    );
    state.defaults = response;
    state.settings = next;
    saveBrowserSettings();
    renderSettingsSummary();
    syncSettingsForm();
    closeSettings();
    setStatus("模型、Provider 与研究路线已应用。", "idle");
    showToast("运行配置已生效。", "success");
  } finally {
    setButtonBusy(els.settingsSave, false, "");
  }
}

async function resetSettings() {
  setButtonBusy(els.settingsReset, true, "恢复中…");
  try {
    const response = await apiFetch("/api/settings/runtime", { method: "DELETE" });
    localStorage.removeItem(STORAGE_KEYS.settings);
    sessionStorage.removeItem(STORAGE_KEYS.secrets);
    state.defaults = response;
    state.settings = mergeSettings(response);
    syncSettingsForm();
    renderSettingsSummary();
    showToast("已恢复服务端默认配置。", "success");
  } finally {
    setButtonBusy(els.settingsReset, false, "");
  }
}

function openSettings() {
  syncSettingsForm();
  els.settingsModal.classList.remove("hidden");
  els.settingsModal.setAttribute("aria-hidden", "false");
}

function closeSettings() {
  els.settingsModal.classList.add("hidden");
  els.settingsModal.setAttribute("aria-hidden", "true");
}

function openManage() {
  els.manageModal.classList.remove("hidden");
  els.manageModal.setAttribute("aria-hidden", "false");
}

function closeManage() {
  els.manageModal.classList.add("hidden");
  els.manageModal.setAttribute("aria-hidden", "true");
}

// 对话与消息
async function refreshConversations() {
  state.conversationsLoading = true;
  els.conversationStatus.textContent = "正在刷新会话…";
  renderConversationList();
  try {
    state.conversations = await apiFetch("/api/conversations");
    els.conversationStatus.textContent = state.conversations.length ? `已加载 ${state.conversations.length} 个会话。` : "暂无历史会话。";
  } catch (error) {
    els.conversationStatus.textContent = error.message;
    els.conversationStatus.className = "inline-status error";
    throw error;
  } finally {
    state.conversationsLoading = false;
    renderConversationList();
  }
}

function renderConversationList() {
  if (state.conversationsLoading) {
    els.conversationEmpty.classList.add("hidden");
    els.conversationList.innerHTML = `<div class="empty-card">正在加载会话列表…</div>`;
    return;
  }
  els.conversationStatus.className = "inline-status";
  els.conversationEmpty.classList.toggle("hidden", state.conversations.length > 0);
  els.conversationList.innerHTML = state.conversations.map((item) => `
    <article class="conversation-item ${item.conversation_id === state.activeConversationId ? "active" : ""}" data-conversation-id="${escapeHtml(item.conversation_id)}">
      <div class="conversation-title-row">
        <div class="conversation-title">${escapeHtml(item.title)}</div>
        <button class="conversation-delete" data-delete-id="${escapeHtml(item.conversation_id)}" type="button" aria-label="删除会话">删除</button>
      </div>
      <div class="conversation-meta">${escapeHtml(formatDate(item.updated_at))} · ${item.message_count} 条消息</div>
    </article>`).join("");
}

async function openConversation(conversationId) {
  state.activeConversationId = conversationId;
  renderConversationList();
  if (!conversationId) {
    state.messages = [];
    state.selectedAssistantMessageId = null;
    renderAll();
    setStatus("已新建空白对话。", "idle");
    els.messageInput.focus();
    return;
  }
  setStatus("正在加载会话内容…", "loading");
  try {
    state.messages = await apiFetch(`/api/conversations/${conversationId}/messages`);
    const last = [...state.messages].reverse().find((message) => message.role === "assistant");
    state.selectedAssistantMessageId = last?.message_id || null;
    renderAll();
    setStatus(`会话已加载，共 ${state.messages.length} 条消息。`, "idle");
  } catch (error) {
    setStatus(error.message || "会话加载失败。", "error");
    showToast(error.message || "会话加载失败。", "error");
  }
}

async function deleteConversation(conversationId, button) {
  if (!window.confirm("确定删除该会话及全部消息吗？此操作无法撤销。")) return;
  const card = button.closest(".conversation-item");
  card?.classList.add("busy");
  setButtonBusy(button, true, "删除中…");
  try {
    await apiFetch(`/api/conversations/${conversationId}`, { method: "DELETE" });
    if (state.activeConversationId === conversationId) {
      state.activeConversationId = null;
      state.messages = [];
      state.selectedAssistantMessageId = null;
    }
    await refreshConversations();
    renderAll();
    showToast("会话已删除。", "success");
  } catch (error) {
    card?.classList.remove("busy");
    setButtonBusy(button, false, "");
    showToast(error.message || "会话删除失败。", "error");
  }
}

function messageEvidence(message) {
  return {
    citations: message.metadata?.citations || [],
    rendered: message.metadata?.rendered_citations || [],
    cards: message.metadata?.source_cards || [],
    claims: message.metadata?.claim_items || [],
    sufficiency: message.metadata?.evidence_sufficiency || "insufficient",
    confidence: message.metadata?.confidence_score ?? 0,
    researchRoute: message.metadata?.research_route || "standard",
  };
}

function deliveryRow(message) {
  if (!message.local_state) return "";
  const labels = { sending: "发送中", sent: "已发送", failed: "发送失败" };
  const retry = message.local_state === "failed"
    ? `<button class="retry-button" type="button" data-retry-id="${escapeHtml(message.message_id)}">重试</button>`
    : "";
  return `<div class="delivery-row"><span class="delivery-state ${message.local_state}">${labels[message.local_state]}</span>${retry}</div>`;
}

function renderThread() {
  const hasMessages = state.messages.length > 0;
  els.welcomePanel.classList.toggle("hidden", hasMessages);
  els.threadEmpty.classList.toggle("hidden", hasMessages);
  els.typingIndicator.classList.toggle("hidden", !state.sending);
  els.thread.innerHTML = state.messages.map((message) => {
    const assistant = message.role === "assistant";
    const evidence = assistant ? messageEvidence(message) : null;
    const failedClass = message.local_state === "failed" ? "failed" : "";
    const assistantMeta = assistant
      ? `<div class="assistant-meta"><span class="mini-badge">${escapeHtml(evidence.sufficiency)}</span><span class="mini-badge">置信度 ${Number(evidence.confidence).toFixed(2)}</span><span class="mini-badge">${escapeHtml(ROUTE_LABELS[evidence.researchRoute] || evidence.researchRoute)}</span></div>`
      : deliveryRow(message);
    return `<article class="message ${escapeHtml(message.role)} ${failedClass} ${message.message_id === state.selectedAssistantMessageId ? "selected" : ""}" data-message-id="${escapeHtml(message.message_id)}">
      <div class="avatar">${assistant ? "AI" : "你"}</div>
      <div class="bubble-wrap"><div class="bubble">${assistant ? renderMarkdown(message.content) : `<p>${escapeHtml(message.content)}</p>`}</div>${assistantMeta}</div>
    </article>`;
  }).join("");
  if (state.sending) els.thread.scrollTop = els.thread.scrollHeight;
}

function renderInspector() {
  const message = state.messages.find((item) => item.message_id === state.selectedAssistantMessageId);
  if (!message || message.role !== "assistant") {
    els.inspectorEmpty.classList.remove("hidden");
    els.inspectorContent.classList.add("hidden");
    return;
  }
  const evidence = messageEvidence(message);
  els.inspectorEmpty.classList.add("hidden");
  els.inspectorContent.classList.remove("hidden");
  els.inspectorCaption.textContent = `所选回答使用 ${ROUTE_LABELS[evidence.researchRoute] || evidence.researchRoute} 路线。`;
  els.inspectorBadges.innerHTML = `<span class="mini-badge">${evidence.citations.length} 条证据</span><span class="mini-badge">置信度 ${Number(evidence.confidence).toFixed(2)}</span>`;
  els.claimList.innerHTML = evidence.claims.map((claim) => `<article class="claim-card"><strong>${escapeHtml(claim.claim_id)}</strong><p>${escapeHtml(claim.text)}</p><div>${escapeHtml((claim.citations || []).join(" "))}</div></article>`).join("") || `<div class="empty-card">暂无声明级信息。</div>`;
  els.citationReferenceList.innerHTML = evidence.rendered.map((item) => `<span class="pill">${escapeHtml(item)}</span>`).join("");
  els.evidenceList.innerHTML = evidence.citations.map((item) => `<article class="evidence-card"><strong>[${escapeHtml(item.label)}] ${escapeHtml(item.source_file)}</strong><div>${item.page != null ? `第 ${item.page} 页` : "页码未知"}</div><p>${escapeHtml(item.snippet || "")}</p></article>`).join("") || `<div class="empty-card">暂无证据条目。</div>`;
  els.sourceCardList.innerHTML = evidence.cards.map((card) => `<article class="source-card"><h3>${escapeHtml(card.title || card.source_file)}</h3><div>${escapeHtml(card.source_file)} · ${(card.pages || []).join(", ")}</div><p>${escapeHtml(card.snippet_preview || "")}</p></article>`).join("") || `<div class="empty-card">暂无来源卡片。</div>`;
}

function renderAll() {
  renderSettingsSummary();
  renderConversationList();
  renderThread();
  renderInspector();
}

async function submitMessage(message, retryId = null) {
  if (!message || state.sending) return;
  let optimistic;
  if (retryId) {
    optimistic = state.messages.find((item) => item.message_id === retryId);
    if (!optimistic) return;
    optimistic.local_state = "sending";
    delete optimistic.local_error;
  } else {
    optimistic = {
      message_id: `local-user-${Date.now()}`,
      role: "user",
      content: message,
      metadata: {},
      created_at: new Date().toISOString(),
      local_state: "sending",
    };
    state.messages.push(optimistic);
  }
  state.sending = true;
  els.sendButton.disabled = true;
  setStatus(`正在使用 ${ROUTE_LABELS[state.settings.researchRoute] || state.settings.researchRoute} 路线处理问题…`, "loading");
  renderThread();
  try {
    const response = await apiFetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...runtimePayload(), conversation_id: state.activeConversationId, message }),
    });
    optimistic.local_state = "sent";
    state.activeConversationId = response.conversation_id;
    state.messages.push({
      message_id: response.message_id,
      role: "assistant",
      content: response.answer,
      metadata: {
        citations: response.citations || [],
        rendered_citations: response.rendered_citations || [],
        source_cards: response.source_cards || [],
        claim_items: response.claim_items || [],
        evidence_sufficiency: response.evidence_sufficiency,
        confidence_score: response.confidence_score,
        diagnostics: response.diagnostics || {},
        retrieval_trace_summary: response.retrieval_trace_summary || {},
        research_route: state.settings.researchRoute,
      },
      created_at: new Date().toISOString(),
    });
    state.selectedAssistantMessageId = response.message_id;
    els.messageInput.value = "";
    await refreshConversations();
    setStatus(`回答完成，耗时 ${Number(response.timings_ms?.total || 0).toFixed(0)} ms。`, "idle");
  } catch (error) {
    optimistic.local_state = "failed";
    optimistic.local_error = error.message || "发送失败。";
    setStatus(optimistic.local_error, "error");
    showToast(`${optimistic.local_error} 问题已保留，可直接重试。`, "error");
  } finally {
    state.sending = false;
    els.sendButton.disabled = false;
    renderAll();
  }
}

function sendMessage(event) {
  event.preventDefault();
  submitMessage(els.messageInput.value.trim());
}

// 上传与增量入库
function newFileEntry(file) {
  return { file, status: "ready", detail: "等待上传" };
}

function addFiles(files) {
  const incoming = Array.from(files);
  if (state.selectedFiles.length && state.selectedFiles.every((item) => ["completed", "saved", "failed"].includes(item.status))) {
    state.selectedFiles = [];
  }
  incoming.forEach((file) => {
    if (!state.selectedFiles.some((current) => current.file.name === file.name && current.file.size === file.size)) {
      state.selectedFiles.push(newFileEntry(file));
    }
  });
  renderUploadFiles();
}

function renderUploadFiles() {
  const hasFiles = state.selectedFiles.length > 0;
  const activeUpload = state.selectedFiles.some((item) => ["uploading", "queued", "saving", "indexing"].includes(item.status));
  const uploadable = state.selectedFiles.some((item) => ["ready", "failed"].includes(item.status));
  els.uploadFileList.classList.toggle("hidden", !hasFiles);
  els.uploadActions.classList.toggle("hidden", !hasFiles);
  const totalBytes = state.selectedFiles.reduce((total, item) => total + item.file.size, 0);
  const completed = state.selectedFiles.filter((item) => item.status === "completed").length;
  const failed = state.selectedFiles.filter((item) => item.status === "failed").length;
  els.uploadSummary.textContent = hasFiles
    ? `${state.selectedFiles.length} 个文件 · ${formatFileSize(totalBytes)}${completed ? ` · ${completed} 个可检索` : ""}${failed ? ` · ${failed} 个失败` : ""}`
    : "尚未选择文件。";
  els.uploadStart.disabled = activeUpload || !uploadable;
  els.uploadClear.disabled = activeUpload;
  els.uploadFileList.innerHTML = state.selectedFiles.map((entry, index) => `
    <div class="upload-file-item">
      <div class="upload-file-copy"><span class="file-name">${escapeHtml(entry.file.name)}</span><span class="file-detail">${formatFileSize(entry.file.size)} · ${escapeHtml(entry.detail || "")}</span></div>
      <span class="file-state ${escapeHtml(entry.status)}">${escapeHtml(FILE_STATE_LABELS[entry.status] || entry.status)}</span>
      <button type="button" class="file-remove" data-index="${index}" ${activeUpload ? "disabled" : ""}>移除</button>
    </div>`).join("");
  els.uploadFileList.querySelectorAll(".file-remove").forEach((button) => button.addEventListener("click", () => {
    state.selectedFiles.splice(Number(button.dataset.index), 1);
    renderUploadFiles();
  }));
}

function setFilesStatus(status, detail, predicate = () => true) {
  state.selectedFiles.forEach((entry) => {
    if (predicate(entry)) {
      entry.status = status;
      entry.detail = detail;
    }
  });
  renderUploadFiles();
}

function setTaskProgress(progress, label) {
  els.uploadProgress.classList.remove("hidden");
  els.uploadProgressBar.style.width = `${Math.max(0, Math.min(100, progress))}%`;
  els.uploadProgressLabel.textContent = label;
}

function applyTaskToFiles(task) {
  const failedNames = (task.errors || []).map((item) => String(item).split(":", 1)[0]);
  state.selectedFiles.forEach((entry) => {
    if (failedNames.includes(entry.file.name)) {
      entry.status = "failed";
      entry.detail = (task.errors || []).find((item) => String(item).startsWith(`${entry.file.name}:`)) || task.message;
      return;
    }
    if (task.stage === "saving") {
      entry.status = "saving";
      entry.detail = "服务端正在校验并保存";
    } else if (task.stage === "indexing" || task.stage === "preparing") {
      entry.status = "indexing";
      entry.detail = "正在解析、分块、向量化并更新索引";
    } else if (task.stage === "saved") {
      entry.status = "saved";
      entry.detail = "文件已保存，尚未写入索引";
    } else if (task.status === "completed") {
      entry.status = "completed";
      entry.detail = "已写入当前知识库，可用于后续检索";
    } else if (task.status === "failed") {
      entry.status = "failed";
      entry.detail = task.message;
    } else {
      entry.status = "queued";
      entry.detail = task.message;
    }
  });
  renderUploadFiles();
}

async function pollTask(taskId, target = "upload") {
  const deadline = Date.now() + 30 * 60 * 1000;
  while (Date.now() < deadline) {
    const task = await apiFetch(`/api/admin/tasks/${taskId}`);
    const label = `${task.message}（${task.progress}%）`;
    if (target === "upload") {
      setTaskProgress(task.progress, label);
      applyTaskToFiles(task);
    } else {
      els.ingestStatus.textContent = label;
    }
    if (["completed", "failed"].includes(task.status)) {
      const node = target === "upload" ? els.uploadStatus : els.ingestStatus;
      node.className = `${target === "upload" ? "upload-status" : "ingest-status"} ${task.status === "completed" ? "success" : "error"}`;
      const errorCount = (task.errors || []).length;
      node.textContent = `${task.message}${errorCount ? `；${errorCount} 个文件失败。` : ""}`;
      if (task.status === "completed") {
        await loadKBStats();
        await loadDocumentList();
        showToast(target === "upload" ? "新参考文献已经可以检索。" : "知识库摄入完成。", "success");
      } else {
        showToast(task.message, "error");
      }
      return task;
    }
    await new Promise((resolve) => window.setTimeout(resolve, 800));
  }
  throw new Error("任务仍在后台运行，请稍后在知识库统计中确认结果。");
}

function startUpload() {
  const entries = state.selectedFiles.filter((item) => ["ready", "failed"].includes(item.status));
  if (!entries.length) return;
  const form = new FormData();
  entries.forEach((entry) => form.append("files", entry.file));
  const xhr = new XMLHttpRequest();
  xhr.open("POST", apiUrl(`/api/admin/documents/upload?auto_ingest=${els.uploadAutoIngest.checked}`));
  setButtonBusy(els.uploadStart, true, "正在上传…");
  setFilesStatus("uploading", "正在向服务端传输", (entry) => entries.includes(entry));
  setTaskProgress(2, "正在传输文件…");
  els.uploadStatus.className = "upload-status loading";
  els.uploadStatus.textContent = `正在上传 ${entries.length} 个文件。`;
  xhr.upload.onprogress = (event) => {
    if (!event.lengthComputable) return;
    const percent = Math.round(event.loaded / event.total * 100);
    setTaskProgress(Math.max(2, Math.round(percent * 0.3)), `文件传输 ${percent}%`);
    setFilesStatus("uploading", `文件传输 ${percent}%`, (entry) => entries.includes(entry));
  };
  xhr.onload = async () => {
    try {
      const payload = JSON.parse(xhr.responseText || "{}");
      if (xhr.status < 200 || xhr.status >= 300) throw new Error(payload.detail || "上传请求失败。");
      setFilesStatus("queued", "文件已传输，等待后台处理", (entry) => entries.includes(entry));
      await pollTask(payload.task_id, "upload");
    } catch (error) {
      setFilesStatus("failed", error.message || "上传失败", (entry) => entries.includes(entry) && entry.status !== "completed");
      els.uploadStatus.className = "upload-status error";
      els.uploadStatus.textContent = error.message || "上传失败。";
      showToast(els.uploadStatus.textContent, "error");
    } finally {
      setButtonBusy(els.uploadStart, false, "");
      renderUploadFiles();
    }
  };
  xhr.onerror = () => {
    setFilesStatus("failed", "网络连接失败，可点击重试", (entry) => entries.includes(entry));
    setButtonBusy(els.uploadStart, false, "");
    els.uploadStatus.className = "upload-status error";
    els.uploadStatus.textContent = "网络连接失败，文件尚未保存。";
    showToast(els.uploadStatus.textContent, "error");
  };
  xhr.send(form);
}

async function loadDocumentList() {
  els.docList.innerHTML = `<div class="empty-card">正在加载文档列表…</div>`;
  try {
    const documents = await apiFetch("/api/admin/documents");
    els.docCount.textContent = `共 ${documents.length} 个文档`;
    els.docList.innerHTML = documents.map((document) => `<div class="doc-item"><div class="doc-info"><div class="doc-name">${escapeHtml(document.title || document.source_file)}</div><div class="doc-meta">${escapeHtml(document.source_file)} · ${formatFileSize(document.size)} · ${escapeHtml(document.status)}</div></div><button class="doc-delete" data-id="${escapeHtml(document.document_id)}" type="button">删除</button></div>`).join("") || `<div class="empty-card">暂无文档，可先在“上传与入库”中添加参考文献。</div>`;
    els.docList.querySelectorAll(".doc-delete").forEach((button) => button.addEventListener("click", async () => {
      if (!window.confirm("确定删除该文档和全部索引分块吗？")) return;
      setButtonBusy(button, true, "删除中…");
      try {
        await apiFetch(`/api/admin/documents/${button.dataset.id}`, { method: "DELETE" });
        await loadDocumentList();
        await loadKBStats();
        showToast("文档及其索引分块已删除。", "success");
      } catch (error) {
        setButtonBusy(button, false, "");
        showToast(error.message || "文档删除失败。", "error");
      }
    }));
  } catch (error) {
    els.docList.innerHTML = `<div class="empty-card">${escapeHtml(error.message)}</div>`;
  }
}

async function loadKBStats() {
  els.kbStats.innerHTML = `<div class="empty-card">正在读取知识库状态…</div>`;
  try {
    const stats = await apiFetch("/api/admin/stats");
    els.kbStats.innerHTML = `<div class="stats-grid"><div class="stat-card"><div class="stat-value">${stats.total_documents}</div><div class="stat-label">文档</div></div><div class="stat-card"><div class="stat-value">${stats.active_documents}</div><div class="stat-label">活跃</div></div><div class="stat-card"><div class="stat-value">${stats.total_chunks}</div><div class="stat-label">分块</div></div></div><div class="stat-extra"><strong>后端：</strong>${escapeHtml(stats.backend)}<br/><strong>Embedding：</strong>${escapeHtml(stats.embedding_model)}<br/><strong>最近摄入：</strong>${escapeHtml(stats.last_ingestion || "暂无")}</div>`;
  } catch (error) {
    els.kbStats.innerHTML = `<div class="empty-card">${escapeHtml(error.message)}</div>`;
  }
}

async function startIngestion() {
  setButtonBusy(els.ingestStart, true, "正在摄入…");
  els.ingestStatus.className = "ingest-status loading";
  els.ingestStatus.textContent = "任务正在排队。";
  try {
    const task = await apiFetch(`/api/admin/ingest?rebuild=${els.ingestRebuild.checked}&force=${els.ingestForce.checked}`, { method: "POST" });
    await pollTask(task.task_id, "ingest");
  } catch (error) {
    els.ingestStatus.className = "ingest-status error";
    els.ingestStatus.textContent = error.message;
    showToast(error.message || "知识库摄入失败。", "error");
  } finally {
    setButtonBusy(els.ingestStart, false, "");
  }
}

function switchManageTab(tabId) {
  document.querySelectorAll(".manage-tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === tabId));
  document.querySelectorAll(".manage-tab-panel").forEach((panel) => panel.classList.toggle("hidden", panel.id !== tabId));
  if (tabId === "tab-docs") loadDocumentList();
  if (tabId === "tab-stats") loadKBStats();
}

// 事件与启动
function bindEvents() {
  els.form.addEventListener("submit", sendMessage);
  els.messageInput.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      els.form.requestSubmit();
    }
  });
  els.thread.addEventListener("click", (event) => {
    const retry = event.target.closest("[data-retry-id]");
    if (retry) {
      const message = state.messages.find((item) => item.message_id === retry.dataset.retryId);
      if (message) submitMessage(message.content, message.message_id);
      return;
    }
    const node = event.target.closest("[data-message-id]");
    if (!node) return;
    const message = state.messages.find((item) => item.message_id === node.dataset.messageId);
    if (message?.role === "assistant") {
      state.selectedAssistantMessageId = message.message_id;
      renderThread();
      renderInspector();
    }
  });
  els.conversationList.addEventListener("click", (event) => {
    const deleteButton = event.target.closest("[data-delete-id]");
    if (deleteButton) {
      deleteConversation(deleteButton.dataset.deleteId, deleteButton);
      return;
    }
    const item = event.target.closest("[data-conversation-id]");
    if (item) openConversation(item.dataset.conversationId);
  });
  els.newChatTop.addEventListener("click", () => openConversation(null));
  els.newChatSidebar.addEventListener("click", () => openConversation(null));
  els.refreshConversations.addEventListener("click", () => refreshConversations().catch((error) => showToast(error.message, "error")));
  els.settingsToggle.addEventListener("click", openSettings);
  els.settingsClose.addEventListener("click", closeSettings);
  els.settingsBackdrop.addEventListener("click", closeSettings);
  els.settingsForm.addEventListener("submit", (event) => saveSettings(event).catch((error) => {
    setStatus(error.message, "error");
    showToast(error.message, "error");
  }));
  els.settingsReset.addEventListener("click", () => resetSettings().catch((error) => showToast(error.message, "error")));
  Object.entries(CAPABILITIES).forEach(([capability, control]) => {
    control.mode.addEventListener("change", () => {
      state.settings[`${capability}Mode`] = control.mode.value;
      state.settings[`${capability}Provider`] = providerOptions(capability, control.mode.value)[0] || "";
      syncCapability(capability);
      renderProviderSummary();
    });
    [control.provider, control.baseUrl, control.apiKey, control.model].forEach((node) => node.addEventListener("input", () => {
      state.settings[`${capability}Provider`] = control.provider.value;
      state.settings[`${capability}BaseUrl`] = control.baseUrl.value.trim();
      state.settings[`${capability}ApiKey`] = control.apiKey.value.trim();
      state.settings[`${capability}Model`] = control.model.value.trim();
      renderCapabilityStatus(capability);
      renderProviderSummary();
    }));
  });
  [els.settingRerank, els.settingLlm].forEach((node) => node.addEventListener("change", () => {
    state.settings.rerank = els.settingRerank.checked;
    state.settings.llm = els.settingLlm.checked;
    Object.keys(CAPABILITIES).forEach(renderCapabilityStatus);
    renderProviderSummary();
  }));
  els.manageToggle.addEventListener("click", openManage);
  els.manageClose.addEventListener("click", closeManage);
  els.manageBackdrop.addEventListener("click", closeManage);
  document.querySelectorAll(".manage-tab").forEach((tab) => tab.addEventListener("click", () => switchManageTab(tab.dataset.tab)));
  els.uploadDropzone.addEventListener("click", () => els.fileInput.click());
  els.uploadBtn.addEventListener("click", (event) => { event.stopPropagation(); els.fileInput.click(); });
  els.fileInput.addEventListener("change", () => { addFiles(els.fileInput.files); els.fileInput.value = ""; });
  els.uploadDropzone.addEventListener("dragover", (event) => { event.preventDefault(); els.uploadDropzone.classList.add("drag-over"); });
  els.uploadDropzone.addEventListener("dragleave", () => els.uploadDropzone.classList.remove("drag-over"));
  els.uploadDropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    els.uploadDropzone.classList.remove("drag-over");
    addFiles(event.dataTransfer.files);
  });
  els.uploadClear.addEventListener("click", () => {
    state.selectedFiles = [];
    els.uploadProgress.classList.add("hidden");
    els.uploadStatus.textContent = "";
    renderUploadFiles();
  });
  els.uploadStart.addEventListener("click", startUpload);
  els.refreshDocs.addEventListener("click", loadDocumentList);
  els.ingestStart.addEventListener("click", startIngestion);
  $("sample-question")?.addEventListener("click", () => {
    els.messageInput.value = "蒸汽气化在1200K附近如何影响NH3和HCN的生成？";
    els.messageInput.focus();
  });
  document.querySelectorAll(".sample-chip[data-sample]").forEach((button) => button.addEventListener("click", () => {
    els.messageInput.value = button.dataset.sample;
    els.messageInput.focus();
  }));
}

async function bootstrap() {
  bindEvents();
  renderUploadFiles();
  setStatus("正在连接后端服务…", "loading");
  try {
    await apiFetch("/health");
    state.defaults = await apiFetch("/api/settings/runtime");
    state.settings = mergeSettings(state.defaults);
    syncSettingsForm();
    renderSettingsSummary();
    await refreshConversations();
    if (state.conversations.length) await openConversation(state.conversations[0].conversation_id);
    else renderAll();
    setStatus(`服务已连接 · ${state.settings.backend} 后端。`, "idle");
  } catch (error) {
    setStatus(error.message || "初始化失败。", "error");
    showToast(error.message || "无法连接后端服务。", "error");
  }
}

bootstrap();
