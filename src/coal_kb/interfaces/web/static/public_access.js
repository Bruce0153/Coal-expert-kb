"use strict";

async function syncAdminControls() {
  const controls = [
    document.getElementById("manage-toggle"),
    document.getElementById("settings-toggle"),
    document.getElementById("settings-summary"),
  ].filter(Boolean);
  try {
    const response = await fetch("/api/auth/admin/status");
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
