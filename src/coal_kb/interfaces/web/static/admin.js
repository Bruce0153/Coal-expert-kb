"use strict";

const statusNode = document.getElementById("admin-status");
const form = document.getElementById("admin-login-form");
const passwordInput = document.getElementById("admin-password");
const loginButton = document.getElementById("admin-login-button");
const actions = document.getElementById("admin-actions");
const logoutButton = document.getElementById("admin-logout-button");

async function request(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `请求失败，状态码 ${response.status}。`);
  return payload;
}

function render(authenticated, message, kind = "") {
  statusNode.textContent = message;
  statusNode.className = `status${kind ? ` ${kind}` : ""}`;
  form.classList.toggle("hidden", authenticated);
  actions.classList.toggle("hidden", !authenticated);
  passwordInput.value = "";
}

async function refreshStatus() {
  try {
    const result = await request("/api/auth/admin/status");
    render(
      result.authenticated,
      result.authenticated ? "管理员已登录，可以进入主站执行维护操作。" : "当前未登录管理员账号。",
      result.authenticated ? "success" : "",
    );
  } catch (error) {
    render(false, error.message, "error");
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  loginButton.disabled = true;
  statusNode.textContent = "正在登录…";
  try {
    await request("/api/auth/admin/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password: passwordInput.value }),
    });
    render(true, "登录成功，可以进入主站执行维护操作。", "success");
  } catch (error) {
    render(false, error.message, "error");
  } finally {
    loginButton.disabled = false;
  }
});

logoutButton.addEventListener("click", async () => {
  logoutButton.disabled = true;
  try {
    await request("/api/auth/admin/logout", { method: "POST" });
    render(false, "已退出管理员登录。", "");
  } catch (error) {
    render(true, error.message, "error");
  } finally {
    logoutButton.disabled = false;
  }
});

refreshStatus();
