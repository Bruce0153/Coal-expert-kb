const form = document.getElementById("ask-form");
const queryInput = document.getElementById("query");
const llmInput = document.getElementById("llm");
const debugInput = document.getElementById("debug");
const kInput = document.getElementById("k");
const modeInput = document.getElementById("mode");
const statusNode = document.getElementById("status");
const answerEmpty = document.getElementById("answer-empty");
const answerContent = document.getElementById("answer-content");
const evidenceEmpty = document.getElementById("evidence-empty");
const evidenceList = document.getElementById("evidence-list");
const diagEmpty = document.getElementById("diag-empty");
const diagnosticsNode = document.getElementById("diagnostics");
const sampleQuestionButton = document.getElementById("sample-question");
const themeToggle = document.getElementById("theme-toggle");

function setStatus(message, kind = "empty") {
  statusNode.textContent = message;
  statusNode.className = `status ${kind}`;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function renderMarkdownSections(markdown) {
  const lines = markdown.split(/\r?\n/);
  let html = "";
  let listOpen = false;

  const closeList = () => {
    if (listOpen) {
      html += "</ul>";
      listOpen = false;
    }
  };

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) {
      closeList();
      continue;
    }
    if (line.startsWith("## ")) {
      closeList();
      html += `<h2>${escapeHtml(line.slice(3))}</h2>`;
      continue;
    }
    if (line.startsWith("- ")) {
      if (!listOpen) {
        html += "<ul>";
        listOpen = true;
      }
      html += `<li>${escapeHtml(line.slice(2)).replace(/\[(E\d+)\]/g, "<code>[$1]</code>")}</li>`;
      continue;
    }
    closeList();
    html += `<p>${escapeHtml(line).replace(/\[(E\d+)\]/g, "<code>[$1]</code>")}</p>`;
  }

  closeList();
  return html;
}

function renderAnswer(answer) {
  answerEmpty.classList.add("hidden");
  answerContent.classList.remove("hidden");
  answerContent.innerHTML = renderMarkdownSections(answer);
}

function renderEvidence(citations) {
  evidenceEmpty.classList.add("hidden");
  evidenceList.classList.remove("hidden");
  evidenceList.innerHTML = citations
    .map((citation) => {
      const usedClass = citation.referenced_in_answer ? "used" : "";
      const heading = citation.heading_path ? ` | ${escapeHtml(citation.heading_path)}` : "";
      const page = citation.page !== null && citation.page !== undefined ? ` | page ${citation.page}` : "";
      return `
        <article class="evidence-card">
          <div class="evidence-meta">
            <div class="evidence-label ${usedClass}">[${escapeHtml(citation.label)}]</div>
            <div class="evidence-source">
              <strong>${escapeHtml(citation.source_file)}</strong>${page}${heading}
            </div>
          </div>
          <p class="evidence-snippet">${escapeHtml(citation.snippet || "No snippet available.")}</p>
        </article>
      `;
    })
    .join("");
}

function renderDiagnostics(payload) {
  diagEmpty.classList.add("hidden");
  diagnosticsNode.classList.remove("hidden");
  diagnosticsNode.textContent = JSON.stringify(payload, null, 2);
}

async function submitQuery(event) {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (!query) {
    setStatus("Please enter a question.", "error");
    return;
  }

  setStatus("Running retrieval and grounding the answer...", "loading");
  answerContent.classList.add("hidden");
  evidenceList.classList.add("hidden");
  diagnosticsNode.classList.add("hidden");
  answerEmpty.classList.remove("hidden");
  evidenceEmpty.classList.remove("hidden");
  diagEmpty.classList.remove("hidden");

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        llm: llmInput.checked,
        debug: debugInput.checked,
        k: Number(kInput.value) || null,
        mode: modeInput.value || null,
      }),
    });

    if (!response.ok) {
      const errorPayload = await response.json().catch(() => ({}));
      throw new Error(errorPayload.detail || "Request failed.");
    }

    const payload = await response.json();
    renderAnswer(payload.answer);
    renderEvidence(payload.citations || []);
    renderDiagnostics(payload.diagnostics || {});
    setStatus(`Done in ${payload.timings_ms.total.toFixed(2)} ms`, "empty");
  } catch (error) {
    setStatus(error.message || "Something went wrong.", "error");
  }
}

sampleQuestionButton.addEventListener("click", () => {
  queryInput.value = "How does steam gasification influence NH3 and HCN formation near 1200 K?";
  queryInput.focus();
});

themeToggle.addEventListener("click", () => {
  document.body.classList.toggle("dark");
});

form.addEventListener("submit", submitQuery);
