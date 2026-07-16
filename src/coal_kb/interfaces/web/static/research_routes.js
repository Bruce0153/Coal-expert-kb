"use strict";

(() => {
  const STORAGE_KEY = "coal-kb-research-route";
  const ROUTES = [
    ["standard", "Standard：标准检索"],
    ["graph", "Graph：证据关系重排"],
    ["multimodal", "Multimodal：文本/表格/图像证据"],
    ["agent", "Agent：受控多步路线"],
  ];
  let activeRoute = localStorage.getItem(STORAGE_KEY) || "standard";
  if (!ROUTES.some(([value]) => value === activeRoute)) activeRoute = "standard";

  const settingsGrid = document.querySelector("#settings-form .settings-grid");
  const label = document.createElement("label");
  label.className = "field";
  label.innerHTML = '<span>研究路线</span><select id="setting-research-route"></select>';
  settingsGrid?.appendChild(label);
  const select = document.getElementById("setting-research-route");
  if (select) {
    select.innerHTML = ROUTES.map(
      ([value, text]) => `<option value="${value}">${text}</option>`,
    ).join("");
    select.value = activeRoute;
    select.addEventListener("change", () => {
      activeRoute = select.value;
      localStorage.setItem(STORAGE_KEY, activeRoute);
      renderRouteChip();
    });
  }

  const nativeFetch = window.fetch.bind(window);
  window.fetch = (input, init = {}) => {
    const url = typeof input === "string" ? input : input.url;
    const method = String(init.method || "GET").toUpperCase();
    const isJsonWrite = method !== "GET" && typeof init.body === "string";
    const acceptsRoute = /\/api\/(chat|ask|settings\/runtime)(?:\?|$)/.test(url);
    if (isJsonWrite && acceptsRoute) {
      try {
        const payload = JSON.parse(init.body);
        payload.research_route = activeRoute;
        init = { ...init, body: JSON.stringify(payload) };
      } catch (_) {
        // Non-JSON requests remain unchanged.
      }
    }
    return nativeFetch(input, init);
  };

  document.getElementById("settings-form")?.addEventListener("submit", () => {
    activeRoute = select?.value || "standard";
    localStorage.setItem(STORAGE_KEY, activeRoute);
    renderRouteChip();
  });
  document.getElementById("settings-reset")?.addEventListener("click", () => {
    activeRoute = "standard";
    localStorage.setItem(STORAGE_KEY, activeRoute);
    if (select) select.value = activeRoute;
    renderRouteChip();
  });

  function renderRouteChip() {
    const summary = document.getElementById("settings-summary");
    if (!summary) return;
    let chip = summary.querySelector("[data-research-route-chip]");
    if (!chip) {
      chip = document.createElement("span");
      chip.className = "settings-chip";
      chip.dataset.researchRouteChip = "true";
      summary.appendChild(chip);
    }
    const labelText = ROUTES.find(([value]) => value === activeRoute)?.[1] || activeRoute;
    const nextText = `路线: ${labelText.split("：")[0]}`;
    if (chip.textContent !== nextText) chip.textContent = nextText;
  }

  const summary = document.getElementById("settings-summary");
  if (summary) new MutationObserver(renderRouteChip).observe(summary, { childList: true });
  renderRouteChip();
})();
