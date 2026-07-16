const STORAGE_KEYS = {
  settings: "coal-kb-ui-settings",
  secrets: "coal-kb-ui-session-secrets",
};

const state = {
  defaults: null,
  settings: null,
  conversations: [],
  activeConversationId: null,
  messages: [],
  selectedAssistantMessageId: null,
  selectedFiles: [],
  sending: false,
};

const $ = (id) => document.getElementById(id);
const els = {
  backendPill: $("backend-pill"), threadStatus: $("thread-status"), conversationEmpty: $("conversation-empty"),
  conversationList: $("conversation-list"), welcomePanel: $("welcome-panel"), threadEmpty: $("thread-empty"),
  thread: $("message-thread"), typingIndicator: $("typing-indicator"), form: $("chat-form"),
  messageInput: $("message-input"), composerNote: $("composer-note"), sendButton: $("send-button"),
  settingsSummary: $("settings-summary"), inspectorEmpty: $("inspector-empty"), inspectorContent: $("inspector-content"),
  inspectorCaption: $("inspector-caption"), inspectorBadges: $("inspector-badges"), claimList: $("claim-list"),
  citationReferenceList: $("citation-reference-list"), evidenceList: $("evidence-list"), sourceCardList: $("source-card-list"),
  newChatTop: $("new-chat-top"), newChatSidebar: $("new-chat-sidebar"), refreshConversations: $("refresh-conversations"),
  settingsToggle: $("settings-toggle"), settingsModal: $("settings-modal"), settingsBackdrop: $("settings-backdrop"),
  settingsClose: $("settings-close"), settingsForm: $("settings-form"), settingsReset: $("settings-reset"),
  settingsNotes: $("settings-notes"), settingApiBaseUrl: $("setting-api-base-url"), settingBackend: $("setting-backend"),
  settingMode: $("setting-mode"), settingK: $("setting-k"), settingRerank: $("setting-rerank"),
  settingLlm: $("setting-llm"), settingDebug: $("setting-debug"), manageToggle: $("manage-toggle"),
  manageModal: $("manage-modal"), manageBackdrop: $("manage-backdrop"), manageClose: $("manage-close"),
  uploadDropzone: $("upload-dropzone"), fileInput: $("file-input"), uploadBtn: $("upload-btn"),
  uploadFileList: $("upload-file-list"), uploadActions: $("upload-actions"), uploadClear: $("upload-clear"),
  uploadStart: $("upload-start"), uploadAutoIngest: $("upload-auto-ingest"), uploadStatus: $("upload-status"),
  uploadProgress: $("upload-progress"), uploadProgressBar: $("upload-progress-bar"), uploadProgressLabel: $("upload-progress-label"),
  refreshDocs: $("refresh-docs"), docCount: $("doc-count"), docList: $("doc-list"), kbStats: $("kb-stats"),
  ingestRebuild: $("ingest-rebuild"), ingestForce: $("ingest-force"), ingestStart: $("ingest-start"), ingestStatus: $("ingest-status"),
};

const CAPABILITIES = {
  tokenizer: {
    defaultsKey: "tokenizer", prefix: "Tokenizer",
    mode: $("setting-tokenizer-mode"), provider: $("setting-tokenizer-provider"),
    baseUrl: $("setting-tokenizer-base-url"), apiKey: $("setting-tokenizer-api-key"), model: $("setting-tokenizer-model"),
  },
  embeddings: {
    defaultsKey: "embeddings", prefix: "Embedding",
    mode: $("setting-embedding-mode"), provider: $("setting-embedding-provider"),
    baseUrl: $("setting-embedding-base-url"), apiKey: $("setting-embedding-api-key"), model: $("setting-embedding-model"),
  },
  rerank: {
    defaultsKey: "rerank_config", prefix: "Rerank",
    mode: $("setting-rerank-mode"), provider: $("setting-rerank-provider"),
    baseUrl: $("setting-rerank-base-url"), apiKey: $("setting-rerank-api-key"), model: $("setting-rerank-model"),
  },
  llm: {
    defaultsKey: "llm_config", prefix: "Llm",
    mode: $("setting-llm-mode"), provider: $("setting-llm-provider"),
    baseUrl: $("setting-llm-base-url"), apiKey: $("setting-llm-api-key"), model: $("setting-llm-model"),
  },
};

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
  els.threadStatus.textContent = message;
  els.backendPill.textContent = kind === "error" ? "连接异常" : kind === "loading" ? "处理中…" : "就绪";
  els.backendPill.className = `status-pill${kind === "error" ? " error" : ""}`;
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

function syncCapability(capability) {
  const control = CAPABILITIES[capability];
  const mode = state.settings[`${capability}Mode`];
  control.mode.value = mode;
  populateSelect(control.provider, providerOptions(capability, mode), state.settings[`${capability}Provider`]);
  control.baseUrl.value = state.settings[`${capability}BaseUrl`] || "";
  control.model.value = state.settings[`${capability}Model`] || "";
  control.apiKey.value = state.settings[`${capability}ApiKey`] || "";
  control.apiKey.disabled = mode !== "remote";
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
    `后端: ${settings.backend}`, `模式: ${settings.retrievalMode}`, `Top-K: ${settings.k}`,
    `Embedding: ${settings.embeddingsProvider}/${settings.embeddingsModel}`,
    settings.llm ? `LLM: ${settings.llmProvider}/${settings.llmModel}` : "仅证据",
  ];
  els.settingsSummary.innerHTML = chips.map((chip) => `<span class="settings-chip">${escapeHtml(chip)}</span>`).join("");
  els.composerNote.textContent = settings.debug ? "诊断模式已开启。" : "回答基于可溯源的知识库证据。";
}

async function saveSettings(event) {
  event.preventDefault();
  const next = collectSettings();
  const response = await apiFetch(
    "/api/settings/runtime",
    { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(runtimePayload(next)) },
    next.apiBaseUrl,
  );
  state.defaults = response;
  state.settings = next;
  saveBrowserSettings();
  renderSettingsSummary();
  closeSettings();
  setStatus("模型与 Provider 设置已应用。", "idle");
}

async function resetSettings() {
  const response = await apiFetch("/api/settings/runtime", { method: "DELETE" });
  localStorage.removeItem(STORAGE_KEYS.settings);
  sessionStorage.removeItem(STORAGE_KEYS.secrets);
  state.defaults = response;
  state.settings = mergeSettings(response);
  syncSettingsForm();
  renderSettingsSummary();
}

function openSettings() { syncSettingsForm(); els.settingsModal.classList.remove("hidden"); }
function closeSettings() { els.settingsModal.classList.add("hidden"); }
function openManage() { els.manageModal.classList.remove("hidden"); }
function closeManage() { els.manageModal.classList.add("hidden"); }

async function refreshConversations() {
  state.conversations = await apiFetch("/api/conversations");
  renderConversationList();
}

function renderConversationList() {
  els.conversationEmpty.classList.toggle("hidden", state.conversations.length > 0);
  els.conversationList.innerHTML = state.conversations.map((item) => `
    <article class="conversation-item ${item.conversation_id === state.activeConversationId ? "active" : ""}" data-conversation-id="${escapeHtml(item.conversation_id)}">
      <div class="conversation-title-row"><div class="conversation-title">${escapeHtml(item.title)}</div><button class="conversation-delete" data-delete-id="${escapeHtml(item.conversation_id)}" type="button">删除</button></div>
      <div class="conversation-meta">${escapeHtml(formatDate(item.updated_at))} · ${item.message_count} 条消息</div>
    </article>`).join("");
}

async function openConversation(conversationId) {
  state.activeConversationId = conversationId;
  state.messages = conversationId ? await apiFetch(`/api/conversations/${conversationId}/messages`) : [];
  const last = [...state.messages].reverse().find((message) => message.role === "assistant");
  state.selectedAssistantMessageId = last?.message_id || null;
  renderAll();
}

async function deleteConversation(conversationId) {
  await apiFetch(`/api/conversations/${conversationId}`, { method: "DELETE" });
  if (state.activeConversationId === conversationId) await openConversation(null);
  await refreshConversations();
}

function messageEvidence(message) {
  return {
    citations: message.metadata?.citations || [], rendered: message.metadata?.rendered_citations || [],
    cards: message.metadata?.source_cards || [], claims: message.metadata?.claim_items || [],
    sufficiency: message.metadata?.evidence_sufficiency || "insufficient",
    confidence: message.metadata?.confidence_score ?? 0,
  };
}

function renderThread() {
  const hasMessages = state.messages.length > 0;
  els.welcomePanel.classList.toggle("hidden", hasMessages);
  els.threadEmpty.classList.toggle("hidden", hasMessages);
  els.typingIndicator.classList.toggle("hidden", !state.sending);
  els.thread.innerHTML = state.messages.map((message) => {
    const assistant = message.role === "assistant";
    const evidence = assistant ? messageEvidence(message) : null;
    return `<article class="message ${escapeHtml(message.role)} ${message.message_id === state.selectedAssistantMessageId ? "selected" : ""}" data-message-id="${escapeHtml(message.message_id)}">
      <div class="avatar">${assistant ? "AI" : "你"}</div><div class="bubble-wrap"><div class="bubble">${assistant ? renderMarkdown(message.content) : `<p>${escapeHtml(message.content)}</p>`}</div>
      ${assistant ? `<div class="assistant-meta"><span class="mini-badge">${escapeHtml(evidence.sufficiency)}</span><span class="mini-badge">置信度 ${Number(evidence.confidence).toFixed(2)}</span></div>` : ""}</div></article>`;
  }).join("");
}

function renderInspector() {
  const message = state.messages.find((item) => item.message_id === state.selectedAssistantMessageId);
  if (!message) {
    els.inspectorEmpty.classList.remove("hidden"); els.inspectorContent.classList.add("hidden"); return;
  }
  const evidence = messageEvidence(message);
  els.inspectorEmpty.classList.add("hidden"); els.inspectorContent.classList.remove("hidden");
  els.inspectorCaption.textContent = "所选回答的声明、引用和来源。";
  els.inspectorBadges.innerHTML = `<span class="mini-badge">${evidence.citations.length} 条证据</span><span class="mini-badge">置信度 ${Number(evidence.confidence).toFixed(2)}</span>`;
  els.claimList.innerHTML = evidence.claims.map((claim) => `<article class="claim-card"><strong>${escapeHtml(claim.claim_id)}</strong><p>${escapeHtml(claim.text)}</p><div>${escapeHtml((claim.citations || []).join(" "))}</div></article>`).join("") || `<div class="empty-card">暂无声明级信息。</div>`;
  els.citationReferenceList.innerHTML = evidence.rendered.map((item) => `<span class="pill">${escapeHtml(item)}</span>`).join("");
  els.evidenceList.innerHTML = evidence.citations.map((item) => `<article class="evidence-card"><strong>[${escapeHtml(item.label)}] ${escapeHtml(item.source_file)}</strong><div>${item.page != null ? `第 ${item.page} 页` : "页码未知"}</div><p>${escapeHtml(item.snippet || "")}</p></article>`).join("");
  els.sourceCardList.innerHTML = evidence.cards.map((card) => `<article class="source-card"><h3>${escapeHtml(card.title || card.source_file)}</h3><div>${escapeHtml(card.source_file)} · ${(card.pages || []).join(", ")}</div><p>${escapeHtml(card.snippet_preview || "")}</p></article>`).join("");
}

function renderAll() { renderSettingsSummary(); renderConversationList(); renderThread(); renderInspector(); }

async function sendMessage(event) {
  event.preventDefault();
  const message = els.messageInput.value.trim();
  if (!message || state.sending) return;
  state.sending = true; els.sendButton.disabled = true; setStatus("正在检索和生成回答…", "loading"); renderThread();
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
    els.messageInput.value = "";
    await refreshConversations();
    setStatus(`回答完成，耗时 ${Number(response.timings_ms?.total || 0).toFixed(0)} ms。`);
  } catch (error) { setStatus(error.message || "发送失败。", "error"); }
  finally { state.sending = false; els.sendButton.disabled = false; renderAll(); }
}

function addFiles(files) {
  Array.from(files).forEach((file) => {
    if (!state.selectedFiles.some((current) => current.name === file.name && current.size === file.size)) state.selectedFiles.push(file);
  });
  renderUploadFiles();
}

function renderUploadFiles() {
  const hasFiles = state.selectedFiles.length > 0;
  els.uploadFileList.classList.toggle("hidden", !hasFiles);
  els.uploadActions.classList.toggle("hidden", !hasFiles);
  els.uploadFileList.innerHTML = state.selectedFiles.map((file, index) => `<div class="upload-file-item"><span class="file-name">${escapeHtml(file.name)}</span><span class="file-size">${formatFileSize(file.size)}</span><button type="button" class="file-remove" data-index="${index}">移除</button></div>`).join("");
  els.uploadFileList.querySelectorAll(".file-remove").forEach((button) => button.addEventListener("click", () => {
    state.selectedFiles.splice(Number(button.dataset.index), 1); renderUploadFiles();
  }));
}

function setTaskProgress(progress, label) {
  els.uploadProgress.classList.remove("hidden");
  els.uploadProgressBar.style.width = `${Math.max(0, Math.min(100, progress))}%`;
  els.uploadProgressLabel.textContent = label;
}

async function pollTask(taskId, target = "upload") {
  while (true) {
    const task = await apiFetch(`/api/admin/tasks/${taskId}`);
    const label = `${task.message}（${task.progress}%）`;
    if (target === "upload") setTaskProgress(task.progress, label);
    else els.ingestStatus.textContent = label;
    if (["completed", "failed"].includes(task.status)) {
      const node = target === "upload" ? els.uploadStatus : els.ingestStatus;
      node.className = `${target === "upload" ? "upload-status" : "ingest-status"} ${task.status === "completed" ? "success" : "error"}`;
      node.textContent = task.message;
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
  xhr.open("POST", apiUrl(`/api/admin/documents/upload?auto_ingest=${els.uploadAutoIngest.checked}`));
  els.uploadStart.disabled = true;
  setTaskProgress(2, "正在传输文件…");
  xhr.upload.onprogress = (event) => {
    if (event.lengthComputable) setTaskProgress(Math.max(2, Math.round(event.loaded / event.total * 30)), "正在传输文件…");
  };
  xhr.onload = async () => {
    try {
      const payload = JSON.parse(xhr.responseText || "{}");
      if (xhr.status < 200 || xhr.status >= 300) throw new Error(payload.detail || "上传请求失败。");
      state.selectedFiles = []; renderUploadFiles();
      await pollTask(payload.task_id, "upload");
    } catch (error) { els.uploadStatus.className = "upload-status error"; els.uploadStatus.textContent = error.message; }
    finally { els.uploadStart.disabled = false; }
  };
  xhr.onerror = () => { els.uploadStart.disabled = false; els.uploadStatus.className = "upload-status error"; els.uploadStatus.textContent = "网络连接失败。"; };
  xhr.send(form);
}

async function loadDocumentList() {
  els.docList.innerHTML = `<div class="empty-card">加载中…</div>`;
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
  try {
    const task = await apiFetch(`/api/admin/ingest?rebuild=${els.ingestRebuild.checked}&force=${els.ingestForce.checked}`, { method: "POST" });
    await pollTask(task.task_id, "ingest");
  } catch (error) { els.ingestStatus.className = "ingest-status error"; els.ingestStatus.textContent = error.message; }
  finally { els.ingestStart.disabled = false; }
}

function switchManageTab(tabId) {
  document.querySelectorAll(".manage-tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === tabId));
  document.querySelectorAll(".manage-tab-panel").forEach((panel) => panel.classList.toggle("hidden", panel.id !== tabId));
  if (tabId === "tab-docs") loadDocumentList();
  if (tabId === "tab-stats") loadKBStats();
}

function bindEvents() {
  els.form.addEventListener("submit", sendMessage);
  els.thread.addEventListener("click", (event) => {
    const message = event.target.closest("[data-message-id]");
    if (message) { state.selectedAssistantMessageId = message.dataset.messageId; renderInspector(); }
  });
  els.conversationList.addEventListener("click", (event) => {
    const deleteButton = event.target.closest("[data-delete-id]");
    if (deleteButton) { deleteConversation(deleteButton.dataset.deleteId); return; }
    const item = event.target.closest("[data-conversation-id]"); if (item) openConversation(item.dataset.conversationId);
  });
  els.newChatTop.addEventListener("click", () => openConversation(null));
  els.newChatSidebar.addEventListener("click", () => openConversation(null));
  els.refreshConversations.addEventListener("click", refreshConversations);
  els.settingsToggle.addEventListener("click", openSettings); els.settingsClose.addEventListener("click", closeSettings); els.settingsBackdrop.addEventListener("click", closeSettings);
  els.settingsForm.addEventListener("submit", (event) => saveSettings(event).catch((error) => setStatus(error.message, "error")));
  els.settingsReset.addEventListener("click", () => resetSettings().catch((error) => setStatus(error.message, "error")));
  Object.entries(CAPABILITIES).forEach(([capability, control]) => control.mode.addEventListener("change", () => {
    state.settings[`${capability}Mode`] = control.mode.value; state.settings[`${capability}Provider`] = providerOptions(capability, control.mode.value)[0] || ""; syncCapability(capability);
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
  $("sample-question")?.addEventListener("click", () => { els.messageInput.value = "蒸汽气化在1200K附近如何影响NH3和HCN的生成？"; });
  document.querySelectorAll(".sample-chip[data-sample]").forEach((button) => button.addEventListener("click", () => { els.messageInput.value = button.dataset.sample; }));
}

async function bootstrap() {
  bindEvents(); setStatus("正在连接后端服务…", "loading");
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
