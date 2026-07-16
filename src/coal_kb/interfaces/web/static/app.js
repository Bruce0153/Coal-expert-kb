const STORAGE_KEYS = {
  settings: "coal-kb-ui-settings",
  secrets: "coal-kb-ui-session-secrets",
};

const state = {
  defaults: null,
  settings: null,
  settingsDirty: false,
  settingsSaving: false,
  conversations: [],
  conversationsLoading: false,
  activeConversationId: null,
  messages: [],
  selectedAssistantMessageId: null,
  pendingMessage: null,
  lastFailedMessage: null,
  selectedFiles: [],
  sending: false,
  upload: {
    taskId: null,
    status: "idle",
    stage: "idle",
    progress: 0,
    message: "选择文件后开始上传。",
    saved: [],
    errors: [],
    stats: null,
    autoIngest: true,
  },
};

const $ = (id) => document.getElementById(id);
const els = {
  backendPill: $("backend-pill"), threadStatus: $("thread-status"), conversationEmpty: $("conversation-empty"),
  conversationList: $("conversation-list"), conversationContext: $("conversation-context"), welcomePanel: $("welcome-panel"),
  threadEmpty: $("thread-empty"), thread: $("message-thread"), typingIndicator: $("typing-indicator"), form: $("chat-form"),
  messageInput: $("message-input"), composerNote: $("composer-note"), sendButton: $("send-button"),
  settingsSummary: $("settings-summary"), inspectorEmpty: $("inspector-empty"), inspectorContent: $("inspector-content"),
  inspectorCaption: $("inspector-caption"), inspectorBadges: $("inspector-badges"), claimList: $("claim-list"),
  citationReferenceList: $("citation-reference-list"), evidenceList: $("evidence-list"), sourceCardList: $("source-card-list"),
  newChatTop: $("new-chat-top"), newChatSidebar: $("new-chat-sidebar"), refreshConversations: $("refresh-conversations"),
  settingsToggle: $("settings-toggle"), settingsModal: $("settings-modal"), settingsBackdrop: $("settings-backdrop"),
  settingsClose: $("settings-close"), settingsForm: $("settings-form"), settingsReset: $("settings-reset"),
  settingsSave: $("settings-save"), settingsApplyStatus: $("settings-apply-status"), settingsNotes: $("settings-notes"),
  settingApiBaseUrl: $("setting-api-base-url"), settingBackend: $("setting-backend"), settingMode: $("setting-mode"),
  settingK: $("setting-k"), settingRerank: $("setting-rerank"), settingLlm: $("setting-llm"), settingDebug: $("setting-debug"),
  manageToggle: $("manage-toggle"), manageModal: $("manage-modal"), manageBackdrop: $("manage-backdrop"), manageClose: $("manage-close"),
  uploadDropzone: $("upload-dropzone"), fileInput: $("file-input"), uploadBtn: $("upload-btn"),
  uploadSelectionSummary: $("upload-selection-summary"), uploadFileList: $("upload-file-list"), uploadActions: $("upload-actions"),
  uploadClear: $("upload-clear"), uploadStart: $("upload-start"), uploadAutoIngest: $("upload-auto-ingest"),
  uploadStatus: $("upload-status"), uploadProgress: $("upload-progress"), uploadProgressBar: $("upload-progress-bar"),
  uploadProgressLabel: $("upload-progress-label"), uploadStageList: $("upload-stage-list"), uploadTaskDetails: $("upload-task-details"),
  refreshDocs: $("refresh-docs"), docCount: $("doc-count"), docList: $("doc-list"), kbStats: $("kb-stats"),
  ingestRebuild: $("ingest-rebuild"), ingestForce: $("ingest-force"), ingestStart: $("ingest-start"), ingestStatus: $("ingest-status"),
};

const CAPABILITIES = {
  tokenizer: {
    defaultsKey: "tokenizer", label: "Tokenizer", prefix: "Tokenizer",
    card: $("provider-tokenizer-card"), status: $("provider-tokenizer-status"), help: $("provider-tokenizer-help"),
    mode: $("setting-tokenizer-mode"), provider: $("setting-tokenizer-provider"),
    baseUrl: $("setting-tokenizer-base-url"), apiKey: $("setting-tokenizer-api-key"), model: $("setting-tokenizer-model"),
  },
  embeddings: {
    defaultsKey: "embeddings", label: "Embeddings", prefix: "Embedding",
    card: $("provider-embeddings-card"), status: $("provider-embeddings-status"), help: $("provider-embeddings-help"),
    mode: $("setting-embedding-mode"), provider: $("setting-embedding-provider"),
    baseUrl: $("setting-embedding-base-url"), apiKey: $("setting-embedding-api-key"), model: $("setting-embedding-model"),
  },
  rerank: {
    defaultsKey: "rerank_config", label: "Rerank", prefix: "Rerank",
    card: $("provider-rerank-card"), status: $("provider-rerank-status"), help: $("provider-rerank-help"),
    mode: $("setting-rerank-mode"), provider: $("setting-rerank-provider"),
    baseUrl: $("setting-rerank-base-url"), apiKey: $("setting-rerank-api-key"), model: $("setting-rerank-model"),
  },
  llm: {
    defaultsKey: "llm_config", label: "LLM", prefix: "Llm",
    card: $("provider-llm-card"), status: $("provider-llm-status"), help: $("provider-llm-help"),
    mode: $("setting-llm-mode"), provider: $("setting-llm-provider"),
    baseUrl: $("setting-llm-base-url"), apiKey: $("setting-llm-api-key"), model: $("setting-llm-model"),
  },
};

const UPLOAD_STAGES = ["transfer", "saving", "indexing", "completed"];

function escapeHtml(value) {
  return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}

function formatDate(value) {
  if (!value) return "";
  return new Intl.DateTimeFormat("zh-CN", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(new Date(value));
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
  const pillLabels = { error: "连接异常", loading: "处理中", success: "已完成", idle: "就绪" };
  els.threadStatus.className = `thread-status ${kind}`;
  els.threadStatus.innerHTML = `<span class="status-dot" aria-hidden="true"></span><span>${escapeHtml(message)}</span>`;
  els.backendPill.textContent = pillLabels[kind] || pillLabels.idle;
  els.backendPill.className = `status-pill ${kind}`;
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
    k: Number(saved.k ?? defaults.k), rerank: saved.rerank ?? defaults.rerank,
    llm: saved.llm ?? defaults.llm, debug: saved.debug ?? defaults.debug,
  };
  Object.keys(CAPABILITIES).forEach((capability) => {
    const current = activeCapability(defaults, capability);
    const prefix = CAPABILITIES[capability].prefix;
    merged[`${capability}Mode`] = saved[`${capability}Mode`] ?? current.mode;
    merged[`${capability}Provider`] = saved[`${capability}Provider`] ?? current.provider ?? "";
    merged[`${capability}BaseUrl`] = saved[`${capability}BaseUrl`] ?? current.base_url ?? "";
    merged[`${capability}Model`] = saved[`${capability}Model`] ?? current.model ?? "";
    merged[`${capability}ApiKey`] = secrets[`${capability}ApiKey`] ?? "";
    merged[`${prefix}Mode`] = merged[`${capability}Mode`];
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

function capabilityFormValue(capability) {
  const control = CAPABILITIES[capability];
  return {
    mode: control.mode.value,
    provider: control.provider.value.trim(),
    baseUrl: control.baseUrl.value.trim(),
    model: control.model.value.trim(),
    apiKey: control.apiKey.value.trim(),
  };
}

function capabilityEnabled(capability) {
  if (capability === "llm") return els.settingLlm.checked;
  if (capability === "rerank") return els.settingRerank.checked;
  return true;
}

function renderProviderStatus(capability) {
  const control = CAPABILITIES[capability];
  const value = capabilityFormValue(capability);
  const enabled = capabilityEnabled(capability);
  let kind = "ready";
  let text = "已配置";
  let help = value.mode === "remote" ? "远程调用会使用当前地址、模型和会话密钥。" : "本地模式不会读取远程 API Key。";

  if (!enabled) {
    kind = "muted";
    text = "当前停用";
    help = capability === "llm" ? "关闭 LLM 后仅返回检索证据。" : "关闭重排序后跳过该能力。";
  } else if (!value.provider || !value.model) {
    kind = "warning";
    text = "配置未完整";
    help = "请至少选择 Provider 并填写模型名称。";
  } else if (value.mode === "remote" && !value.apiKey) {
    kind = "attention";
    text = "密钥未填写";
    help = "可继续保存；仅当远程服务要求鉴权时需要 API Key。";
  }

  control.card.className = `provider-card ${kind}`;
  control.status.className = `provider-status ${kind}`;
  control.status.textContent = text;
  control.help.textContent = help;
  control.apiKey.disabled = value.mode !== "remote";
  control.apiKey.closest("label")?.classList.toggle("field-disabled", value.mode !== "remote");
}

function syncCapability(capability) {
  const control = CAPABILITIES[capability];
  const mode = state.settings[`${capability}Mode`];
  control.mode.value = mode;
  populateSelect(control.provider, providerOptions(capability, mode), state.settings[`${capability}Provider`]);
  control.baseUrl.value = state.settings[`${capability}BaseUrl`] || "";
  control.model.value = state.settings[`${capability}Model`] || "";
  control.apiKey.value = state.settings[`${capability}ApiKey`] || "";
  renderProviderStatus(capability);
}

function renderSettingsApplyStatus(message = "当前配置已与运行态同步。", kind = "ready") {
  els.settingsApplyStatus.className = `settings-apply-status ${kind}`;
  els.settingsApplyStatus.textContent = message;
  els.settingsSave.disabled = state.settingsSaving;
  els.settingsSave.textContent = state.settingsSaving ? "正在应用…" : state.settingsDirty ? "保存并应用更改" : "保存并应用";
}

function syncSettingsForm() {
  populateSelect(els.settingBackend, state.defaults.backend_options, state.settings.backend);
  populateSelect(els.settingMode, state.defaults.mode_options, state.settings.retrievalMode);
  els.settingApiBaseUrl.value = state.settings.apiBaseUrl;
  els.settingK.value = state.settings.k;
  els.settingRerank.checked = state.settings.rerank;
  els.settingLlm.checked = state.settings.llm;
  els.settingDebug.checked = state.settings.debug;
  Object.keys(CAPABILITIES).forEach(syncCapability);
  els.settingsNotes.innerHTML = (state.defaults.notes || []).map((note) => `<div>· ${escapeHtml(note)}</div>`).join("");
  state.settingsDirty = false;
  renderSettingsApplyStatus();
}

function collectSettings() {
  const next = {
    apiBaseUrl: els.settingApiBaseUrl.value.trim(), backend: els.settingBackend.value,
    retrievalMode: els.settingMode.value, k: Math.max(1, Number(els.settingK.value) || 5),
    rerank: els.settingRerank.checked, llm: els.settingLlm.checked, debug: els.settingDebug.checked,
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

function validateSettings(settings) {
  const missing = [];
  Object.keys(CAPABILITIES).forEach((capability) => {
    const enabled = capability === "llm" ? settings.llm : capability === "rerank" ? settings.rerank : true;
    if (!enabled) return;
    if (!settings[`${capability}Provider`]) missing.push(`${CAPABILITIES[capability].label} Provider`);
    if (!settings[`${capability}Model`]) missing.push(`${CAPABILITIES[capability].label} 模型`);
  });
  return missing;
}

function runtimePayload(settings = state.settings) {
  return {
    llm: settings.llm, debug: settings.debug, backend: settings.backend,
    mode: settings.retrievalMode, k: Number(settings.k), rerank: settings.rerank,
    tokenizer_mode: settings.tokenizerMode, tokenizer_provider: settings.tokenizerProvider,
    tokenizer_base_url: settings.tokenizerBaseUrl || null, tokenizer_api_key: settings.tokenizerApiKey || null,
    tokenizer_model: settings.tokenizerModel,
    embedding_mode: settings.embeddingsMode, embedding_provider: settings.embeddingsProvider,
    embedding_base_url: settings.embeddingsBaseUrl || null, embedding_api_key: settings.embeddingsApiKey || null,
    embedding_model: settings.embeddingsModel,
    rerank_mode: settings.rerankMode, rerank_provider: settings.rerankProvider,
    rerank_base_url: settings.rerankBaseUrl || null, rerank_api_key: settings.rerankApiKey || null,
    rerank_model: settings.rerankModel,
    llm_mode: settings.llmMode, llm_provider: settings.llm ? settings.llmProvider : "none",
    llm_base_url: settings.llmBaseUrl || null, llm_api_key: settings.llmApiKey || null,
    llm_model: settings.llmModel,
  };
}

function renderSettingsSummary() {
  const settings = state.settings;
  if (!settings) return;
  const chips = [
    `后端: ${settings.backend}`, `检索: ${settings.retrievalMode}`, `Top-K: ${settings.k}`,
    `Embedding: ${settings.embeddingsMode}/${settings.embeddingsProvider}`,
    settings.rerank ? `Rerank: ${settings.rerankMode}/${settings.rerankProvider}` : "Rerank: 关闭",
    settings.llm ? `LLM: ${settings.llmMode}/${settings.llmProvider}` : "回答: 仅证据",
  ];
  els.settingsSummary.innerHTML = chips.map((chip) => `<span class="settings-chip">${escapeHtml(chip)}</span>`).join("");
  els.composerNote.textContent = state.sending
    ? "正在组织检索证据与回答，请勿重复提交。"
    : settings.debug ? "诊断模式已开启。按 Ctrl/⌘ + Enter 可发送。" : "回答基于可溯源证据。按 Ctrl/⌘ + Enter 可发送。";
}

function markSettingsDirty() {
  state.settingsDirty = true;
  Object.keys(CAPABILITIES).forEach(renderProviderStatus);
  renderSettingsApplyStatus("有未保存的配置更改。", "dirty");
}

async function saveSettings(event) {
  event.preventDefault();
  const next = collectSettings();
  const missing = validateSettings(next);
  if (missing.length) {
    renderSettingsApplyStatus(`请补充：${missing.join("、")}。`, "error");
    return;
  }
  state.settingsSaving = true;
  renderSettingsApplyStatus("正在写入后端运行配置…", "saving");
  try {
    const response = await apiFetch(
      "/api/settings/runtime",
      { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(runtimePayload(next)) },
      next.apiBaseUrl,
    );
    state.defaults = response;
    state.settings = next;
    state.settingsDirty = false;
    saveBrowserSettings();
    renderSettingsSummary();
    renderSettingsApplyStatus("已应用到后续问答与增量入库。", "success");
    setStatus("模型与 Provider 设置已应用。", "success");
    window.setTimeout(closeSettings, 420);
  } catch (error) {
    renderSettingsApplyStatus(error.message || "设置保存失败。", "error");
    throw error;
  } finally {
    state.settingsSaving = false;
    if (state.settingsDirty) renderSettingsApplyStatus("有未保存的配置更改。", "dirty");
    else if (!els.settingsModal.classList.contains("hidden")) renderSettingsApplyStatus("已应用到后续问答与增量入库。", "success");
  }
}

async function resetSettings() {
  state.settingsSaving = true;
  renderSettingsApplyStatus("正在恢复后端默认配置…", "saving");
  try {
    const response = await apiFetch("/api/settings/runtime", { method: "DELETE" });
    localStorage.removeItem(STORAGE_KEYS.settings);
    sessionStorage.removeItem(STORAGE_KEYS.secrets);
    state.defaults = response;
    state.settings = mergeSettings(response);
    syncSettingsForm();
    renderSettingsSummary();
    setStatus("已恢复默认运行配置。", "success");
  } finally {
    state.settingsSaving = false;
    renderSettingsApplyStatus();
  }
}

function openSettings() { syncSettingsForm(); els.settingsModal.classList.remove("hidden"); els.settingsModal.setAttribute("aria-hidden", "false"); }
function closeSettings() { els.settingsModal.classList.add("hidden"); els.settingsModal.setAttribute("aria-hidden", "true"); }
function openManage() { renderUploadTask(); els.manageModal.classList.remove("hidden"); els.manageModal.setAttribute("aria-hidden", "false"); }
function closeManage() { els.manageModal.classList.add("hidden"); els.manageModal.setAttribute("aria-hidden", "true"); }

async function refreshConversations({ silent = false } = {}) {
  state.conversationsLoading = true;
  if (!silent) renderConversationList();
  try {
    state.conversations = await apiFetch("/api/conversations");
  } finally {
    state.conversationsLoading = false;
    renderConversationList();
  }
}

function renderConversationList() {
  if (state.conversationsLoading) {
    els.conversationEmpty.classList.add("hidden");
    els.conversationList.innerHTML = `<div class="conversation-loading"><span class="inline-spinner"></span>正在同步历史会话…</div>`;
    return;
  }
  els.conversationEmpty.classList.toggle("hidden", state.conversations.length > 0);
  els.conversationList.innerHTML = state.conversations.map((item) => `
    <article class="conversation-item ${item.conversation_id === state.activeConversationId ? "active" : ""}" data-conversation-id="${escapeHtml(item.conversation_id)}">
      <div class="conversation-title-row"><div class="conversation-title">${escapeHtml(item.title)}</div><button class="conversation-delete" data-delete-id="${escapeHtml(item.conversation_id)}" type="button" aria-label="删除会话">删除</button></div>
      <div class="conversation-meta">${escapeHtml(formatDate(item.updated_at))} · ${item.message_count} 条消息</div>
    </article>`).join("");
  const active = state.conversations.find((item) => item.conversation_id === state.activeConversationId);
  els.conversationContext.textContent = active ? `当前：${active.title} · ${active.message_count} 条历史消息` : "当前为新对话，发送后自动保存。";
}

async function openConversation(conversationId) {
  if (state.sending) return;
  state.activeConversationId = conversationId;
  state.pendingMessage = null;
  state.lastFailedMessage = null;
  if (!conversationId) {
    state.messages = [];
    state.selectedAssistantMessageId = null;
    renderAll();
    setStatus("已新建对话，可以开始提问。", "idle");
    els.messageInput.focus();
    return;
  }
  setStatus("正在加载历史会话…", "loading");
  try {
    state.messages = await apiFetch(`/api/conversations/${conversationId}/messages`);
    const last = [...state.messages].reverse().find((message) => message.role === "assistant");
    state.selectedAssistantMessageId = last?.message_id || null;
    renderAll();
    setStatus("历史会话已加载。", "idle");
  } catch (error) {
    setStatus(error.message || "历史会话加载失败。", "error");
  }
}

async function deleteConversation(conversationId) {
  if (!confirm("确定删除该会话及其全部消息吗？")) return;
  setStatus("正在删除会话…", "loading");
  await apiFetch(`/api/conversations/${conversationId}`, { method: "DELETE" });
  if (state.activeConversationId === conversationId) await openConversation(null);
  await refreshConversations({ silent: true });
  setStatus("会话已删除。", "success");
}

function messageEvidence(message) {
  return {
    citations: message.metadata?.citations || [], rendered: message.metadata?.rendered_citations || [],
    cards: message.metadata?.source_cards || [], claims: message.metadata?.claim_items || [],
    sufficiency: message.metadata?.evidence_sufficiency || "insufficient",
    confidence: message.metadata?.confidence_score ?? 0,
  };
}

function renderMessage(message) {
  const assistant = message.role === "assistant";
  const evidence = assistant ? messageEvidence(message) : null;
  const selected = assistant && message.message_id === state.selectedAssistantMessageId;
  return `<article class="message ${escapeHtml(message.role)} ${selected ? "selected" : ""}" data-message-id="${escapeHtml(message.message_id)}" data-role="${escapeHtml(message.role)}">
    <div class="avatar">${assistant ? "AI" : "你"}</div><div class="bubble-wrap"><div class="bubble">${assistant ? renderMarkdown(message.content) : `<p>${escapeHtml(message.content)}</p>`}</div>
    ${assistant ? `<div class="assistant-meta"><span class="mini-badge ${escapeHtml(evidence.sufficiency)}">${escapeHtml(evidence.sufficiency)}</span><span class="mini-badge">置信度 ${Number(evidence.confidence).toFixed(2)}</span></div>` : ""}</div></article>`;
}

function renderThread() {
  const transient = [];
  if (state.pendingMessage) transient.push({ ...state.pendingMessage, state: "pending" });
  if (state.lastFailedMessage) transient.push({ ...state.lastFailedMessage, state: "failed" });
  const hasMessages = state.messages.length > 0 || transient.length > 0;
  els.welcomePanel.classList.toggle("hidden", hasMessages);
  els.threadEmpty.classList.toggle("hidden", hasMessages);
  els.typingIndicator.classList.toggle("hidden", !state.sending);
  const persisted = state.messages.map(renderMessage).join("");
  const pending = state.pendingMessage ? `<article class="message user pending"><div class="avatar">你</div><div class="bubble-wrap"><div class="bubble"><p>${escapeHtml(state.pendingMessage.content)}</p></div><div class="message-state"><span class="inline-spinner"></span>正在发送并等待检索结果…</div></div></article>` : "";
  const failed = state.lastFailedMessage ? `<article class="message user failed"><div class="avatar">你</div><div class="bubble-wrap"><div class="bubble"><p>${escapeHtml(state.lastFailedMessage.content)}</p></div><div class="message-error"><span>${escapeHtml(state.lastFailedMessage.error)}</span><button type="button" class="text-button" data-retry-message>重试</button></div></div></article>` : "";
  els.thread.innerHTML = persisted + pending + failed;
}

function renderInspector() {
  const message = state.messages.find((item) => item.message_id === state.selectedAssistantMessageId && item.role === "assistant");
  if (!message) {
    els.inspectorEmpty.classList.remove("hidden"); els.inspectorContent.classList.add("hidden"); return;
  }
  const evidence = messageEvidence(message);
  els.inspectorEmpty.classList.add("hidden"); els.inspectorContent.classList.remove("hidden");
  els.inspectorCaption.textContent = "所选回答的声明、引用和来源。";
  els.inspectorBadges.innerHTML = `<span class="mini-badge">${evidence.citations.length} 条证据</span><span class="mini-badge">置信度 ${Number(evidence.confidence).toFixed(2)}</span>`;
  els.claimList.innerHTML = evidence.claims.map((claim) => `<article class="claim-card"><strong>${escapeHtml(claim.claim_id)}</strong><p>${escapeHtml(claim.text)}</p><div>${escapeHtml((claim.citations || []).join(" "))}</div></article>`).join("") || `<div class="empty-card">暂无声明级信息。</div>`;
  els.citationReferenceList.innerHTML = evidence.rendered.map((item) => `<span class="pill">${escapeHtml(item)}</span>`).join("") || `<span class="section-hint">暂无格式化引用。</span>`;
  els.evidenceList.innerHTML = evidence.citations.map((item) => `<article class="evidence-card"><strong>[${escapeHtml(item.label)}] ${escapeHtml(item.source_file)}</strong><div>${item.page != null ? `第 ${item.page} 页` : "页码未知"}</div><p>${escapeHtml(item.snippet || "")}</p></article>`).join("") || `<div class="empty-card">当前回答未返回证据条目。</div>`;
  els.sourceCardList.innerHTML = evidence.cards.map((card) => `<article class="source-card"><h3>${escapeHtml(card.title || card.source_file)}</h3><div>${escapeHtml(card.source_file)} · ${(card.pages || []).join(", ")}</div><p>${escapeHtml(card.snippet_preview || "")}</p></article>`).join("") || `<div class="empty-card">暂无来源卡片。</div>`;
}

function renderAll() { renderSettingsSummary(); renderConversationList(); renderThread(); renderInspector(); }

function scrollThreadToBottom() {
  window.requestAnimationFrame(() => { els.thread.parentElement?.scrollTo({ top: els.thread.parentElement.scrollHeight, behavior: "smooth" }); });
}

async function submitMessage(message) {
  if (!message || state.sending) return;
  state.sending = true;
  state.pendingMessage = { message_id: `pending-${Date.now()}`, role: "user", content: message };
  state.lastFailedMessage = null;
  els.sendButton.disabled = true;
  els.sendButton.textContent = "生成中…";
  els.messageInput.disabled = true;
  setStatus("正在检索证据并生成回答…", "loading");
  renderAll();
  scrollThreadToBottom();
  try {
    const response = await apiFetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...runtimePayload(), conversation_id: state.activeConversationId, message }),
    });
    state.activeConversationId = response.conversation_id;
    state.messages.push(
      { message_id: `local-${Date.now()}`, role: "user", content: message, metadata: {}, created_at: new Date().toISOString() },
      { message_id: response.message_id, role: "assistant", content: response.answer, metadata: {
        citations: response.citations || [], rendered_citations: response.rendered_citations || [], source_cards: response.source_cards || [],
        claim_items: response.claim_items || [], evidence_sufficiency: response.evidence_sufficiency, confidence_score: response.confidence_score,
      }, created_at: new Date().toISOString() },
    );
    state.selectedAssistantMessageId = response.message_id;
    state.pendingMessage = null;
    els.messageInput.value = "";
    await refreshConversations({ silent: true });
    setStatus(`回答完成，耗时 ${Number(response.timings_ms?.total || 0).toFixed(0)} ms。`, "success");
  } catch (error) {
    state.pendingMessage = null;
    state.lastFailedMessage = { message_id: `failed-${Date.now()}`, role: "user", content: message, error: error.message || "发送失败。" };
    setStatus(error.message || "发送失败。", "error");
  } finally {
    state.sending = false;
    els.sendButton.disabled = false;
    els.sendButton.textContent = "发送";
    els.messageInput.disabled = false;
    renderAll();
    scrollThreadToBottom();
    els.messageInput.focus();
  }
}

async function sendMessage(event) {
  event.preventDefault();
  await submitMessage(els.messageInput.value.trim());
}

function addFiles(files) {
  Array.from(files).forEach((file) => {
    if (!state.selectedFiles.some((current) => current.name === file.name && current.size === file.size)) state.selectedFiles.push(file);
  });
  renderUploadFiles();
}

function renderUploadFiles() {
  const hasFiles = state.selectedFiles.length > 0;
  const totalSize = state.selectedFiles.reduce((sum, file) => sum + file.size, 0);
  els.uploadFileList.classList.toggle("hidden", !hasFiles);
  els.uploadActions.classList.toggle("hidden", !hasFiles);
  els.uploadSelectionSummary.textContent = hasFiles ? `已选择 ${state.selectedFiles.length} 个文件，共 ${formatFileSize(totalSize)}。` : "尚未选择文件。";
  els.uploadFileList.innerHTML = state.selectedFiles.map((file, index) => `<div class="upload-file-item"><span class="file-name">${escapeHtml(file.name)}</span><span class="file-size">${formatFileSize(file.size)}</span><button type="button" class="file-remove" data-index="${index}">移除</button></div>`).join("");
  els.uploadFileList.querySelectorAll(".file-remove").forEach((button) => button.addEventListener("click", () => {
    state.selectedFiles.splice(Number(button.dataset.index), 1); renderUploadFiles();
  }));
}

function uploadStageState(stageName) {
  const current = state.upload.stage;
  if (current === "failed") return stageName === state.upload.failedAt ? "failed" : "idle";
  if (current === "saved") {
    if (["transfer", "saving", "completed"].includes(stageName)) return "done";
    return "skipped";
  }
  const currentIndex = UPLOAD_STAGES.indexOf(current);
  const stageIndex = UPLOAD_STAGES.indexOf(stageName);
  if (current === "queued" && stageName === "saving") return "active";
  if (currentIndex < 0) return stageName === "transfer" && state.upload.status === "uploading" ? "active" : "idle";
  if (stageIndex < currentIndex) return "done";
  if (stageIndex === currentIndex) return current === "completed" ? "done" : "active";
  return "idle";
}

function renderUploadTask() {
  const upload = state.upload;
  const visible = upload.status !== "idle";
  els.uploadProgress.classList.toggle("hidden", !visible);
  els.uploadProgressBar.style.width = `${Math.max(0, Math.min(100, upload.progress || 0))}%`;
  els.uploadProgressLabel.textContent = upload.message || "";
  els.uploadStatus.className = `upload-status ${upload.status === "failed" ? "error" : upload.status === "completed" ? "success" : "running"}`;
  els.uploadStatus.textContent = upload.message || "";
  els.uploadStageList.querySelectorAll("[data-upload-stage]").forEach((node) => {
    const stageState = uploadStageState(node.dataset.uploadStage);
    node.className = `upload-stage ${stageState}`;
  });
  const details = [];
  if (upload.taskId) details.push(`<div><strong>任务：</strong>${escapeHtml(upload.taskId.slice(0, 8))}</div>`);
  if (upload.saved?.length) details.push(`<div><strong>已保存：</strong>${upload.saved.map(escapeHtml).join("、")}</div>`);
  if (upload.stats) details.push(`<div><strong>结果：</strong>扫描 ${Number(upload.stats.docs_scanned || 0)} 个文档，写入 ${Number(upload.stats.indexed || 0)} 个分块。</div>`);
  if (upload.errors?.length) details.push(`<div class="task-error-list"><strong>文件错误：</strong>${upload.errors.map((error) => `<div>${escapeHtml(error)}</div>`).join("")}</div>`);
  if (upload.status === "running" || upload.status === "queued") details.push("<div class=\"task-hint\">任务已进入单线程队列，关闭管理窗口不会中断后端处理。</div>");
  els.uploadTaskDetails.innerHTML = details.join("");
  els.uploadTaskDetails.classList.toggle("hidden", details.length === 0);
  els.uploadStart.disabled = upload.status === "uploading" || upload.status === "running" || upload.status === "queued";
  els.uploadStart.textContent = els.uploadStart.disabled ? "处理中…" : "上传并处理";
}

function updateUpload(patch) {
  state.upload = { ...state.upload, ...patch };
  renderUploadTask();
}

function mapBackendStage(task) {
  if (task.status === "failed") return "failed";
  if (task.stage === "queued" || task.stage === "preparing") return "queued";
  if (task.stage === "saving") return "saving";
  if (task.stage === "indexing") return "indexing";
  if (task.stage === "saved") return "saved";
  if (task.stage === "completed") return "completed";
  return task.stage || "queued";
}

async function pollTask(taskId, target = "upload") {
  while (true) {
    const task = await apiFetch(`/api/admin/tasks/${taskId}`);
    const label = `${task.message}（${task.progress}%）`;
    if (target === "upload") {
      const mappedStage = mapBackendStage(task);
      updateUpload({
        taskId, status: task.status, stage: mappedStage, failedAt: state.upload.stage,
        progress: task.progress, message: label, saved: task.saved || [], errors: task.errors || [], stats: task.stats || null,
      });
    } else {
      els.ingestStatus.className = `ingest-status ${task.status === "failed" ? "error" : task.status === "completed" ? "success" : "running"}`;
      els.ingestStatus.textContent = label;
    }
    if (["completed", "failed"].includes(task.status)) {
      if (task.status === "completed") { await loadKBStats(); await loadDocumentList(); }
      return task;
    }
    await new Promise((resolve) => setTimeout(resolve, 800));
  }
}

function startUpload() {
  if (!state.selectedFiles.length) return;
  const form = new FormData();
  state.selectedFiles.forEach((file) => form.append("files", file));
  const xhr = new XMLHttpRequest();
  const autoIngest = els.uploadAutoIngest.checked;
  xhr.open("POST", apiUrl(`/api/admin/documents/upload?auto_ingest=${autoIngest}`));
  updateUpload({ taskId: null, status: "uploading", stage: "transfer", failedAt: null, progress: 2, message: "正在传输文件…", saved: [], errors: [], stats: null, autoIngest });
  xhr.upload.onprogress = (event) => {
    if (event.lengthComputable) updateUpload({ progress: Math.max(2, Math.round(event.loaded / event.total * 30)), message: `正在传输文件… ${Math.round(event.loaded / event.total * 100)}%` });
  };
  xhr.onload = async () => {
    try {
      const payload = JSON.parse(xhr.responseText || "{}");
      if (xhr.status < 200 || xhr.status >= 300) throw new Error(payload.detail || "上传请求失败。");
      state.selectedFiles = [];
      renderUploadFiles();
      updateUpload({ taskId: payload.task_id, status: "queued", stage: "queued", progress: 32, message: "文件传输完成，等待后端处理。" });
      await pollTask(payload.task_id, "upload");
    } catch (error) {
      updateUpload({ status: "failed", stage: "failed", failedAt: state.upload.stage || "transfer", progress: 100, message: error.message || "上传失败。" });
    }
  };
  xhr.onerror = () => updateUpload({ status: "failed", stage: "failed", failedAt: "transfer", progress: 100, message: "网络连接失败，请检查 API 地址后重试。" });
  xhr.send(form);
}

async function loadDocumentList() {
  els.docList.innerHTML = `<div class="empty-card"><span class="inline-spinner"></span>正在加载文档…</div>`;
  try {
    const documents = await apiFetch("/api/admin/documents");
    els.docCount.textContent = `共 ${documents.length} 个文档`;
    els.docList.innerHTML = documents.map((document) => `<div class="doc-item"><div class="doc-info"><div class="doc-name">${escapeHtml(document.title || document.source_file)}</div><div class="doc-meta">${escapeHtml(document.source_file)} · ${formatFileSize(document.size)} · ${escapeHtml(document.status)}</div></div><button class="doc-delete" data-id="${escapeHtml(document.document_id)}" type="button">删除</button></div>`).join("") || `<div class="empty-card">暂无文档。</div>`;
    els.docList.querySelectorAll(".doc-delete").forEach((button) => button.addEventListener("click", async () => {
      if (!confirm("确定删除该文档和全部索引分块吗？")) return;
      await apiFetch(`/api/admin/documents/${button.dataset.id}`, { method: "DELETE" }); await loadDocumentList(); await loadKBStats();
    }));
  } catch (error) { els.docList.innerHTML = `<div class="empty-card">${escapeHtml(error.message)}</div>`; }
}

async function loadKBStats() {
  try {
    const stats = await apiFetch("/api/admin/stats");
    els.kbStats.innerHTML = `<div class="stats-grid"><div class="stat-card"><div class="stat-value">${stats.total_documents}</div><div class="stat-label">文档</div></div><div class="stat-card"><div class="stat-value">${stats.active_documents}</div><div class="stat-label">活跃</div></div><div class="stat-card"><div class="stat-value">${stats.total_chunks}</div><div class="stat-label">分块</div></div></div><div class="stat-extra"><strong>后端：</strong>${escapeHtml(stats.backend)}<br/><strong>Embedding：</strong>${escapeHtml(stats.embedding_model)}<br/><strong>最近摄入：</strong>${escapeHtml(stats.last_ingestion || "暂无")}</div>`;
  } catch (error) { els.kbStats.innerHTML = `<div class="empty-card">${escapeHtml(error.message)}</div>`; }
}

async function startIngestion() {
  els.ingestStart.disabled = true;
  els.ingestStart.textContent = "处理中…";
  try {
    const task = await apiFetch(`/api/admin/ingest?rebuild=${els.ingestRebuild.checked}&force=${els.ingestForce.checked}`, { method: "POST" });
    await pollTask(task.task_id, "ingest");
  } catch (error) { els.ingestStatus.className = "ingest-status error"; els.ingestStatus.textContent = error.message; }
  finally { els.ingestStart.disabled = false; els.ingestStart.textContent = "开始摄入"; }
}

function switchManageTab(tabId) {
  document.querySelectorAll(".manage-tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === tabId));
  document.querySelectorAll(".manage-tab-panel").forEach((panel) => panel.classList.toggle("hidden", panel.id !== tabId));
  if (tabId === "tab-docs") loadDocumentList();
  if (tabId === "tab-stats") loadKBStats();
}

function bindEvents() {
  els.form.addEventListener("submit", sendMessage);
  els.messageInput.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") { event.preventDefault(); els.form.requestSubmit(); }
  });
  els.thread.addEventListener("click", (event) => {
    const retry = event.target.closest("[data-retry-message]");
    if (retry && state.lastFailedMessage) { const content = state.lastFailedMessage.content; state.lastFailedMessage = null; submitMessage(content); return; }
    const message = event.target.closest('[data-message-id][data-role="assistant"]');
    if (message) { state.selectedAssistantMessageId = message.dataset.messageId; renderInspector(); renderThread(); }
  });
  els.conversationList.addEventListener("click", (event) => {
    const deleteButton = event.target.closest("[data-delete-id]");
    if (deleteButton) { deleteConversation(deleteButton.dataset.deleteId).catch((error) => setStatus(error.message, "error")); return; }
    const item = event.target.closest("[data-conversation-id]"); if (item) openConversation(item.dataset.conversationId);
  });
  els.newChatTop.addEventListener("click", () => openConversation(null));
  els.newChatSidebar.addEventListener("click", () => openConversation(null));
  els.refreshConversations.addEventListener("click", () => refreshConversations().catch((error) => setStatus(error.message, "error")));
  els.settingsToggle.addEventListener("click", openSettings); els.settingsClose.addEventListener("click", closeSettings); els.settingsBackdrop.addEventListener("click", closeSettings);
  els.settingsForm.addEventListener("submit", (event) => saveSettings(event).catch((error) => setStatus(error.message, "error")));
  els.settingsReset.addEventListener("click", () => resetSettings().catch((error) => { renderSettingsApplyStatus(error.message, "error"); setStatus(error.message, "error"); }));
  els.settingsForm.addEventListener("input", markSettingsDirty);
  els.settingsForm.addEventListener("change", markSettingsDirty);
  Object.entries(CAPABILITIES).forEach(([capability, control]) => {
    control.mode.addEventListener("change", () => {
      state.settings[`${capability}Mode`] = control.mode.value;
      state.settings[`${capability}Provider`] = providerOptions(capability, control.mode.value)[0] || "";
      syncCapability(capability);
      markSettingsDirty();
    });
  });
  document.querySelectorAll("[data-key-toggle]").forEach((button) => button.addEventListener("click", () => {
    const input = $(button.dataset.keyToggle);
    const revealing = input.type === "password";
    input.type = revealing ? "text" : "password";
    button.textContent = revealing ? "隐藏" : "显示";
  }));
  els.manageToggle.addEventListener("click", openManage); els.manageClose.addEventListener("click", closeManage); els.manageBackdrop.addEventListener("click", closeManage);
  document.querySelectorAll(".manage-tab").forEach((tab) => tab.addEventListener("click", () => switchManageTab(tab.dataset.tab)));
  els.uploadDropzone.addEventListener("click", () => els.fileInput.click()); els.uploadBtn.addEventListener("click", (event) => { event.stopPropagation(); els.fileInput.click(); });
  els.fileInput.addEventListener("change", () => { addFiles(els.fileInput.files); els.fileInput.value = ""; });
  els.uploadDropzone.addEventListener("dragover", (event) => { event.preventDefault(); els.uploadDropzone.classList.add("drag-over"); });
  els.uploadDropzone.addEventListener("dragleave", () => els.uploadDropzone.classList.remove("drag-over"));
  els.uploadDropzone.addEventListener("drop", (event) => { event.preventDefault(); els.uploadDropzone.classList.remove("drag-over"); addFiles(event.dataTransfer.files); });
  els.uploadClear.addEventListener("click", () => { state.selectedFiles = []; renderUploadFiles(); }); els.uploadStart.addEventListener("click", startUpload);
  els.refreshDocs.addEventListener("click", loadDocumentList); els.ingestStart.addEventListener("click", startIngestion);
  $("sample-question")?.addEventListener("click", () => { els.messageInput.value = "蒸汽气化在1200K附近如何影响NH3和HCN的生成？"; els.messageInput.focus(); });
  document.querySelectorAll(".sample-chip[data-sample]").forEach((button) => button.addEventListener("click", () => { els.messageInput.value = button.dataset.sample; els.messageInput.focus(); }));
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    if (!els.settingsModal.classList.contains("hidden")) closeSettings();
    if (!els.manageModal.classList.contains("hidden")) closeManage();
  });
}

async function bootstrap() {
  bindEvents();
  renderUploadFiles();
  renderUploadTask();
  setStatus("正在连接后端服务…", "loading");
  try {
    await apiFetch("/health");
    state.defaults = await apiFetch("/api/settings/runtime");
    state.settings = mergeSettings(state.defaults);
    syncSettingsForm(); renderSettingsSummary();
    await refreshConversations();
    if (state.conversations.length) await openConversation(state.conversations[0].conversation_id); else renderAll();
    setStatus("服务已连接。", "idle");
  } catch (error) { setStatus(error.message || "初始化失败。", "error"); }
}

bootstrap();
