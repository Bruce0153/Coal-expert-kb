"use strict";

const nativeFetch = window.fetch.bind(window);

window.fetch = async (input, options = {}) => {
  const method = String(options.method || "GET").toUpperCase();
  const url = typeof input === "string" ? input : input.url;
  if (method === "GET" && url.endsWith("/api/settings/runtime")) {
    const response = await nativeFetch(input, options);
    if (response.status !== 401 && response.status !== 403) return response;
    return nativeFetch("/api/public/config", options);
  }
  return nativeFetch(input, options);
};

async function syncAdminControls() {
  const controls = [
    document.getElementById("manage-toggle"),
    document.getElementById("settings-toggle"),
    document.getElementById("settings-summary"),
  ].filter(Boolean);
  try {
    const response = await nativeFetch("/api/auth/admin/status");
    const payload = await response.json();
    const authenticated = response.ok && Boolean(payload.authenticated);
    controls.forEach((node) => node.classList.toggle("hidden", !authenticated));
    document.documentElement.dataset.adminAuthenticated = String(authenticated);
  } catch (_) {
    controls.forEach((node) => node.classList.add("hidden"));
    document.documentElement.dataset.adminAuthenticated = "false";
  }
}

syncAdminControls();
