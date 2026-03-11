const STORAGE_KEYS = {
  settings: "coal-kb-ui-settings",
  theme: "coal-kb-ui-theme",
};

const state = {
  defaults: null,
  settings: null,
  conversations: [],
  activeConversationId: null,
  messages: [],
  selectedAssistantMessageId: null,
  sending: false,
  error: "",
};

const els = {
  body: document.body,
  backendPill: document.getElementById("backend-pill"),
  threadStatus: document.getElementById("thread-status"),
  conversationEmpty: document.getElementById("conversation-empty"),
  conversationList: document.getElementById("conversation-list"),
  welcomePanel: document.getElementById("welcome-panel"),
  threadEmpty: document.getElementById("thread-empty"),
  thread: document.getElementById("message-thread"),
  typingIndicator: document.getElementById("typing-indicator"),
  form: document.getElementById("chat-form"),
  messageInput: document.getElementById("message-input"),
  composerNote: document.getElementById("composer-note"),
  sendButton: document.getElementById("send-button"),
  settingsSummary: document.getElementById("settings-summary"),
  inspectorEmpty: document.getElementById("inspector-empty"),
  inspectorContent: document.getElementById("inspector-content"),
  inspectorCaption: document.getElementById("inspector-caption"),
  inspectorBadges: document.getElementById("inspector-badges"),
  claimList: document.getElementById("claim-list"),
  citationReferenceList: document.getElementById("citation-reference-list"),
  evidenceList: document.getElementById("evidence-list"),
  sourceCardList: document.getElementById("source-card-list"),
  diagnostics: document.getElementById("diagnostics"),
  sampleQuestion: document.getElementById("sample-question"),
  sampleChips: Array.from(document.querySelectorAll(".sample-chip")),
  themeToggle: document.getElementById("theme-toggle"),
  newChatTop: document.getElementById("new-chat-top"),
  newChatSidebar: document.getElementById("new-chat-sidebar"),
  refreshConversations: document.getElementById("refresh-conversations"),
  settingsToggle: document.getElementById("settings-toggle"),
  settingsModal: document.getElementById("settings-modal"),
  settingsBackdrop: document.getElementById("settings-backdrop"),
  settingsClose: document.getElementById("settings-close"),
  settingsForm: document.getElementById("settings-form"),
  settingsReset: document.getElementById("settings-reset"),
  settingsNotes: document.getElementById("settings-notes"),
  settingApiBaseUrl: document.getElementById("setting-api-base-url"),
  settingProviderBaseUrl: document.getElementById("setting-provider-base-url"),
  settingApiKey: document.getElementById("setting-api-key"),
  settingLlmProvider: document.getElementById("setting-llm-provider"),
  settingLlmModel: document.getElementById("setting-llm-model"),
  settingEmbeddingModel: document.getElementById("setting-embedding-model"),
  settingBackend: document.getElementById("setting-backend"),
  settingMode: document.getElementById("setting-mode"),
  settingK: document.getElementById("setting-k"),
  settingRerank: document.getElementById("setting-rerank"),
  settingLlm: document.getElementById("setting-llm"),
  settingDebug: document.getElementById("setting-debug"),
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function formatDate(value) {
  if (!value) {
    return "";
  }
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function renderMarkdown(markdown) {
  const lines = String(markdown || "").split(/\r?\n/);
  let html = "";
  let listOpen = false;

  const flushList = () => {
    if (listOpen) {
      html += "</ul>";
      listOpen = false;
    }
  };

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) {
      flushList();
      continue;
    }

    const safeLine = escapeHtml(line).replace(/\[(E\d+)\]/g, "<code>[$1]</code>");
    if (line.startsWith("## ")) {
      flushList();
      html += `<h2>${escapeHtml(line.slice(3))}</h2>`;
      continue;
    }
    if (line.startsWith("- ")) {
      if (!listOpen) {
        html += "<ul>";
        listOpen = true;
      }
      html += `<li>${safeLine.slice(2)}</li>`;
      continue;
    }
    flushList();
    html += `<p>${safeLine}</p>`;
  }

  flushList();
  return html;
}

function mergeSettings(defaults, saved) {
  return {
    apiBaseUrl: saved?.apiBaseUrl ?? defaults.api_base_url ?? "",
    providerBaseUrl: saved?.providerBaseUrl ?? defaults.provider_base_url ?? "",
    apiKey: saved?.apiKey ?? "",
    llmProvider: saved?.llmProvider ?? defaults.llm_provider ?? "none",
    llmModel: saved?.llmModel ?? defaults.llm_model ?? "",
    embeddingModel: saved?.embeddingModel ?? defaults.embedding_model ?? "",
    backend: saved?.backend ?? defaults.backend ?? "elastic",
    mode: saved?.mode ?? defaults.mode ?? "balanced",
    k: Number(saved?.k ?? defaults.k ?? 6),
    rerank: saved?.rerank ?? defaults.rerank ?? true,
    llm: saved?.llm ?? defaults.llm ?? false,
    debug: saved?.debug ?? defaults.debug ?? false,
  };
}

function loadSavedSettings() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEYS.settings) || "null");
  } catch {
    return null;
  }
}

function saveSettings() {
  localStorage.setItem(STORAGE_KEYS.settings, JSON.stringify(state.settings));
}

function loadTheme() {
  const theme = localStorage.getItem(STORAGE_KEYS.theme) || "dark";
  els.body.dataset.theme = theme;
}

function toggleTheme() {
  const nextTheme = els.body.dataset.theme === "dark" ? "light" : "dark";
  els.body.dataset.theme = nextTheme;
  localStorage.setItem(STORAGE_KEYS.theme, nextTheme);
}

function normalizedApiBaseUrl() {
  const raw = (state.settings?.apiBaseUrl || "").trim();
  return raw ? raw.replace(/\/+$/, "") : "";
}

function apiUrl(path) {
  return `${normalizedApiBaseUrl()}${path}`;
}

async function apiFetch(path, options = {}) {
  const response = await fetch(apiUrl(path), options);
  if (!response.ok) {
    const errorPayload = await response.json().catch(() => ({}));
    throw new Error(errorPayload.detail || `Request failed with status ${response.status}.`);
  }
  return response.json();
}

function setStatus(message, kind = "idle") {
  els.threadStatus.textContent = message;
  els.backendPill.textContent = kind === "error" ? "API issue" : kind === "loading" ? "Working…" : "API ready";
  els.backendPill.className = kind === "error" ? "status-pill error" : "status-pill";
}

function setError(message) {
  state.error = message;
  setStatus(message, "error");
}

function clearError() {
  state.error = "";
}

function populateSelect(node, options, selectedValue) {
  node.innerHTML = options
    .map((option) => `<option value="${escapeHtml(option)}" ${option === selectedValue ? "selected" : ""}>${escapeHtml(option)}</option>`)
    .join("");
}

function syncSettingsForm() {
  const { settings, defaults } = state;
  if (!settings || !defaults) {
    return;
  }

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
  els.settingsNotes.innerHTML = defaults.notes.map((note) => `<div>• ${escapeHtml(note)}</div>`).join("");
}

function renderSettingsSummary() {
  if (!state.settings) {
    return;
  }
  const chips = [
    `backend: ${state.settings.backend}`,
    `mode: ${state.settings.mode}`,
    `top-k: ${state.settings.k}`,
    state.settings.rerank ? "rerank on" : "rerank off",
    state.settings.llm ? `llm: ${state.settings.llmModel || "enabled"}` : "evidence-only",
  ];
  els.settingsSummary.innerHTML = chips.map((chip) => `<span class="settings-chip">${escapeHtml(chip)}</span>`).join("");
  els.composerNote.textContent = state.settings.debug
    ? "Debug mode is on. Retrieval trace and context diagnostics will be available in the evidence panel."
    : "Answers stay evidence-first, with citations, source cards, and a developer-friendly evidence panel.";
}

function getVisibleMessages(conversationId) {
  return state.activeConversationId === conversationId ? state.messages : [];
}

function getConversationPreview(conversationId) {
  const messages = getVisibleMessages(conversationId);
  const lastAssistant = [...messages].reverse().find((message) => message.role === "assistant");
  if (!lastAssistant) {
    return "No assistant answer yet.";
  }
  return lastAssistant.content.replace(/\s+/g, " ").slice(0, 88);
}

function renderConversationList() {
  els.conversationEmpty.classList.toggle("hidden", state.conversations.length > 0);
  els.conversationList.innerHTML = state.conversations
    .map((conversation) => {
      const activeClass = conversation.conversation_id === state.activeConversationId ? "active" : "";
      return `
        <article class="conversation-item ${activeClass}" data-conversation-id="${escapeHtml(conversation.conversation_id)}">
          <div class="conversation-title-row">
            <div class="conversation-title">${escapeHtml(conversation.title)}</div>
            <button class="conversation-delete" type="button" data-delete-id="${escapeHtml(conversation.conversation_id)}">Delete</button>
          </div>
          <p class="conversation-preview">${escapeHtml(getConversationPreview(conversation.conversation_id))}</p>
          <div class="conversation-meta">${escapeHtml(formatDate(conversation.updated_at))} · ${escapeHtml(String(conversation.message_count))} messages</div>
        </article>
      `;
    })
    .join("");
}

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
  if (!state.selectedAssistantMessageId) {
    return null;
  }
  return state.messages.find((message) => message.message_id === state.selectedAssistantMessageId) || null;
}

function renderThread() {
  const hasMessages = state.messages.length > 0;
  els.welcomePanel.classList.toggle("hidden", hasMessages);
  els.threadEmpty.classList.toggle("hidden", hasMessages);
  els.thread.innerHTML = state.messages
    .map((message) => {
      const isAssistant = message.role === "assistant";
      const selectedClass = message.message_id === state.selectedAssistantMessageId ? "selected" : "";
      const evidence = isAssistant ? messageEvidence(message) : null;
      const citationChips = (evidence?.renderedCitations || []).slice(0, 3);
      const badges = isAssistant
        ? `
          <div class="assistant-meta">
            <span class="mini-badge ${escapeHtml(evidence.evidenceSufficiency)}">${escapeHtml(evidence.evidenceSufficiency)}</span>
            <span class="mini-badge">confidence ${escapeHtml(String(Number(evidence.confidenceScore || 0).toFixed(2)))}</span>
            <span class="conversation-meta">${escapeHtml(formatDate(message.created_at))}</span>
          </div>
        `
        : `<div class="assistant-meta"><span class="conversation-meta">${escapeHtml(formatDate(message.created_at))}</span></div>`;
      const citationRow = isAssistant && citationChips.length
        ? `<div class="pill-list">${citationChips.map((item) => `<span class="citation-pill">${escapeHtml(item)}</span>`).join("")}</div>`
        : "";

      return `
        <article class="message ${escapeHtml(message.role)} ${selectedClass}" data-message-id="${escapeHtml(message.message_id)}">
          <div class="avatar">${isAssistant ? "AI" : "YOU"}</div>
          <div class="bubble-wrap">
            <div class="bubble">${isAssistant ? renderMarkdown(message.content) : `<p>${escapeHtml(message.content)}</p>`}</div>
            <div class="assistant-footer">
              ${badges}
              ${citationRow}
            </div>
          </div>
        </article>
      `;
    })
    .join("");
  els.typingIndicator.classList.toggle("hidden", !state.sending);
}

function renderInspector() {
  const message = selectedAssistantMessage();
  if (!message) {
    els.inspectorEmpty.classList.remove("hidden");
    els.inspectorContent.classList.add("hidden");
    els.inspectorCaption.textContent = "Select an assistant message to inspect its supporting evidence and retrieval details.";
    els.inspectorBadges.innerHTML = "";
    return;
  }

  const evidence = messageEvidence(message);
  const summary = evidence.retrievalTraceSummary || {};
  els.inspectorEmpty.classList.add("hidden");
  els.inspectorContent.classList.remove("hidden");
  els.inspectorCaption.textContent = summary.retrieval_query ? `Retrieval query: ${summary.retrieval_query}` : "Inspecting the selected assistant turn.";
  els.inspectorBadges.innerHTML = `
    <span class="mini-badge ${escapeHtml(evidence.evidenceSufficiency)}">${escapeHtml(evidence.evidenceSufficiency)}</span>
    <span class="mini-badge">confidence ${escapeHtml(String(Number(evidence.confidenceScore || 0).toFixed(2)))}</span>
    <span class="mini-badge">${escapeHtml(String((evidence.citations || []).length))} evidence items</span>
  `;
  els.claimList.innerHTML = (evidence.claimItems || [])
    .map((claim) => {
      const labels = (claim.citations || []).map((label) => `[${label}]`).join(" ");
      return `
        <article class="claim-card">
          <div class="claim-support">${escapeHtml(claim.claim_id || "Claim")} · ${escapeHtml(claim.support || "supported")}</div>
          <p class="evidence-snippet">${escapeHtml(claim.text || "")}</p>
          <div class="pill-list">${labels ? `<span class="pill">${escapeHtml(labels)}</span>` : ""}</div>
        </article>
      `;
    })
    .join("");

  els.citationReferenceList.innerHTML = (evidence.renderedCitations || [])
    .map((item) => `<span class="pill">${escapeHtml(item)}</span>`)
    .join("");

  els.evidenceList.innerHTML = (evidence.citations || [])
    .map((item) => {
      const page = item.page !== null && item.page !== undefined ? `page ${item.page}` : "page n/a";
      const heading = item.heading_path ? item.heading_path : "heading n/a";
      return `
        <article class="evidence-card">
          <div class="evidence-header">
            <div class="evidence-label">[${escapeHtml(item.label)}]</div>
            <div class="evidence-source">
              <strong>${escapeHtml(item.source_file || item.title || "unknown source")}</strong><br />
              ${escapeHtml(page)} · ${escapeHtml(heading)}
            </div>
          </div>
          <p class="evidence-snippet">${escapeHtml(item.snippet || "No snippet available.")}</p>
        </article>
      `;
    })
    .join("");

  els.sourceCardList.innerHTML = (evidence.sourceCards || [])
    .map((card) => {
      const pages = (card.pages || []).join(", ") || "n/a";
      const headings = (card.headings || []).join(", ") || "n/a";
      const labels = (card.evidence_labels || []).map((label) => `[${label}]`).join(" ");
      return `
        <article class="source-card">
          <div class="source-card-header">
            <div>
              <h3>${escapeHtml(card.title || card.source_file || "Unknown source")}</h3>
              <div class="source-card-meta">${escapeHtml(card.source_file || "unknown file")} · pages ${escapeHtml(pages)}</div>
              <div class="source-card-meta">${escapeHtml(headings)}</div>
            </div>
            <div class="mini-badge">${escapeHtml(String(card.evidence_count || 0))} snippets</div>
          </div>
          <p class="source-card-preview">${escapeHtml(card.snippet_preview || "No preview available.")}</p>
          <div class="pill-list">${labels ? `<span class="pill">${escapeHtml(labels)}</span>` : ""}</div>
        </article>
      `;
    })
    .join("");

  els.diagnostics.textContent = JSON.stringify(
    {
      retrieval_trace_summary: evidence.retrievalTraceSummary || {},
      diagnostics: evidence.diagnostics || {},
    },
    null,
    2,
  );
}

function renderAll() {
  renderSettingsSummary();
  renderConversationList();
  renderThread();
  renderInspector();
}

async function loadSettingsDefaults() {
  state.defaults = await apiFetch("/api/settings/defaults");
  state.settings = mergeSettings(state.defaults, loadSavedSettings());
  syncSettingsForm();
  renderSettingsSummary();
}

async function refreshConversations() {
  state.conversations = await apiFetch("/api/conversations");
  if (state.activeConversationId && !state.conversations.some((item) => item.conversation_id === state.activeConversationId)) {
    state.activeConversationId = null;
    state.messages = [];
    state.selectedAssistantMessageId = null;
  }
}

function chooseAssistantSelection() {
  const lastAssistant = [...state.messages].reverse().find((message) => message.role === "assistant");
  state.selectedAssistantMessageId = lastAssistant?.message_id || null;
}

async function loadMessages(conversationId) {
  if (!conversationId) {
    state.messages = [];
    state.selectedAssistantMessageId = null;
    renderAll();
    return;
  }
  setStatus("Loading conversation…", "loading");
  state.messages = await apiFetch(`/api/conversations/${conversationId}/messages`);
  chooseAssistantSelection();
  setStatus("Conversation loaded.", "idle");
  renderAll();
}

async function openConversation(conversationId) {
  if (!conversationId) {
    state.activeConversationId = null;
    state.messages = [];
    state.selectedAssistantMessageId = null;
    setStatus("Started a fresh conversation. Send the first message when you're ready.", "idle");
    renderAll();
    return;
  }
  state.activeConversationId = conversationId;
  await loadMessages(conversationId);
}

async function deleteConversation(conversationId) {
  const response = await fetch(apiUrl(`/api/conversations/${conversationId}`), { method: "DELETE" });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || "Failed to delete conversation.");
  }
  if (state.activeConversationId === conversationId) {
    state.activeConversationId = null;
    state.messages = [];
    state.selectedAssistantMessageId = null;
  }
  await refreshConversations();
  renderAll();
  setStatus("Conversation deleted.", "idle");
}

function buildChatPayload(message) {
  return {
    conversation_id: state.activeConversationId,
    message,
    llm: state.settings.llm,
    debug: state.settings.debug,
    backend: state.settings.backend,
    mode: state.settings.mode,
    k: Number(state.settings.k),
    rerank: state.settings.rerank,
    llm_provider: state.settings.llmProvider,
    api_key: state.settings.apiKey || null,
    provider_base_url: state.settings.providerBaseUrl || null,
    llm_model: state.settings.llmModel || null,
    embedding_model: state.settings.embeddingModel || null,
  };
}

async function sendMessage(event) {
  event.preventDefault();
  const message = els.messageInput.value.trim();
  if (!message || state.sending) {
    return;
  }

  clearError();
  state.sending = true;
  els.sendButton.disabled = true;
  setStatus("Running retrieval and composing a grounded answer…", "loading");
  renderAll();

  try {
    const payload = await apiFetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildChatPayload(message)),
    });

    state.activeConversationId = payload.conversation_id;
    state.messages = [
      ...state.messages,
      {
        message_id: `local-user-${Date.now()}`,
        conversation_id: payload.conversation_id,
        role: "user",
        content: message,
        metadata: {},
        created_at: new Date().toISOString(),
      },
      {
        message_id: payload.message_id,
        conversation_id: payload.conversation_id,
        role: "assistant",
        content: payload.answer,
        metadata: {
          citations: payload.citations || [],
          rendered_citations: payload.rendered_citations || [],
          source_cards: payload.source_cards || [],
          claim_items: payload.claim_items || [],
          diagnostics: payload.diagnostics || {},
          retrieval_trace_summary: payload.retrieval_trace_summary || {},
          evidence_sufficiency: payload.evidence_sufficiency || "insufficient",
          confidence_score: payload.confidence_score ?? 0,
        },
        created_at: new Date().toISOString(),
      },
    ];
    state.selectedAssistantMessageId = payload.message_id;
    els.messageInput.value = "";
    await refreshConversations();
    setStatus(`Answer ready in ${Number(payload.timings_ms?.total || 0).toFixed(2)} ms.`, "idle");
  } catch (error) {
    setError(error.message || "Something went wrong while sending the message.");
  } finally {
    state.sending = false;
    els.sendButton.disabled = false;
    renderAll();
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

function resetSettings() {
  state.settings = mergeSettings(state.defaults, null);
  saveSettings();
  syncSettingsForm();
  renderSettingsSummary();
}

function saveSettingsFromForm(event) {
  event.preventDefault();
  state.settings = {
    apiBaseUrl: els.settingApiBaseUrl.value.trim(),
    providerBaseUrl: els.settingProviderBaseUrl.value.trim(),
    apiKey: els.settingApiKey.value.trim(),
    llmProvider: els.settingLlmProvider.value,
    llmModel: els.settingLlmModel.value.trim(),
    embeddingModel: els.settingEmbeddingModel.value.trim(),
    backend: els.settingBackend.value,
    mode: els.settingMode.value,
    k: Math.max(1, Number(els.settingK.value) || 6),
    rerank: els.settingRerank.checked,
    llm: els.settingLlm.checked,
    debug: els.settingDebug.checked,
  };
  saveSettings();
  renderSettingsSummary();
  closeSettings();
  setStatus("Settings saved locally for future chat requests.", "idle");
}

function handleThreadClick(event) {
  const messageNode = event.target.closest("[data-message-id]");
  if (!messageNode) {
    return;
  }
  const message = state.messages.find((item) => item.message_id === messageNode.getAttribute("data-message-id"));
  if (message?.role !== "assistant") {
    return;
  }
  state.selectedAssistantMessageId = message.message_id;
  renderAll();
}

function handleConversationClick(event) {
  const deleteButton = event.target.closest("[data-delete-id]");
  if (deleteButton) {
    deleteConversation(deleteButton.getAttribute("data-delete-id")).catch((error) => setError(error.message || "Failed to delete conversation."));
    return;
  }

  const conversationNode = event.target.closest("[data-conversation-id]");
  if (!conversationNode) {
    return;
  }
  openConversation(conversationNode.getAttribute("data-conversation-id")).catch((error) => setError(error.message || "Failed to load conversation."));
}

function applySample(text) {
  els.messageInput.value = text;
  els.messageInput.focus();
}

async function bootstrap() {
  loadTheme();
  setStatus("Connecting to the backend…", "loading");

  try {
    await apiFetch("/health");
    await loadSettingsDefaults();
    await refreshConversations();
    if (state.conversations.length) {
      await openConversation(state.conversations[0].conversation_id);
    } else {
      renderAll();
      setStatus("No saved conversations yet. Start a new grounded chat when ready.", "idle");
    }
  } catch (error) {
    setError(error.message || "Failed to initialize the application.");
  }
}

els.form.addEventListener("submit", sendMessage);
els.thread.addEventListener("click", handleThreadClick);
els.conversationList.addEventListener("click", handleConversationClick);
els.themeToggle.addEventListener("click", toggleTheme);
els.newChatTop.addEventListener("click", () => openConversation(null));
els.newChatSidebar.addEventListener("click", () => openConversation(null));
els.refreshConversations.addEventListener("click", () => {
  refreshConversations().then(renderAll).catch((error) => setError(error.message || "Failed to refresh conversations."));
});
els.sampleQuestion.addEventListener("click", () => applySample("How does steam gasification influence NH3 and HCN formation near 1200 K?"));
els.sampleChips.forEach((chip) => chip.addEventListener("click", () => applySample(chip.dataset.sample || "")));
els.settingsToggle.addEventListener("click", openSettings);
els.settingsClose.addEventListener("click", closeSettings);
els.settingsBackdrop.addEventListener("click", closeSettings);
els.settingsReset.addEventListener("click", resetSettings);
els.settingsForm.addEventListener("submit", saveSettingsFromForm);

bootstrap();
