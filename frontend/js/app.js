/* =========================================================
   Shared utilities: API client, toasts, modals, nav highlighting
   ========================================================= */
const API_BASE = "/api";

async function apiRequest(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!res.ok) {
    let detail = "Something went wrong. Please try again.";
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch (_) {
      if (res.status === 0) detail = "Cannot reach the server. Is the backend running?";
    }
    throw new Error(detail);
  }

  const contentType = res.headers.get("content-type") || "";
  if (contentType.includes("application/json")) return res.json();
  return res;
}

const api = {
  get: (path) => apiRequest(path),
  post: (path, body) => apiRequest(path, { method: "POST", body: JSON.stringify(body) }),
  put: (path, body) => apiRequest(path, { method: "PUT", body: JSON.stringify(body) }),
  del: (path) => apiRequest(path, { method: "DELETE" }),
};

function showToast(message, type = "info") {
  let container = document.getElementById("toast-container");
  if (!container) {
    container = document.createElement("div");
    container.id = "toast-container";
    document.body.appendChild(container);
  }
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}

function confirmModal({ title, message, confirmLabel = "Confirm", danger = true }) {
  return new Promise((resolve) => {
    const overlay = document.createElement("div");
    overlay.className = "modal-overlay";
    overlay.innerHTML = `
      <div class="modal-box">
        <h3>${title}</h3>
        <p style="color:var(--slate-600); font-size:0.9rem;">${message}</p>
        <div class="modal-actions">
          <button class="btn btn-outline" data-action="cancel">Cancel</button>
          <button class="btn ${danger ? "btn-danger" : "btn-gold"}" data-action="ok">${confirmLabel}</button>
        </div>
      </div>`;
    document.body.appendChild(overlay);
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay || e.target.dataset.action === "cancel") {
        overlay.remove();
        resolve(false);
      }
      if (e.target.dataset.action === "ok") {
        overlay.remove();
        resolve(true);
      }
    });
  });
}

function setActiveNavLink() {
  const current = window.location.pathname.split("/").pop() || "dashboard.html";
  document.querySelectorAll(".nav-list a").forEach((a) => {
    if (a.getAttribute("href") === current) a.classList.add("active");
  });
}

function setTodayDateChip() {
  const chip = document.getElementById("today-date-chip");
  if (chip) {
    chip.textContent = new Date().toLocaleDateString(undefined, {
      weekday: "long", year: "numeric", month: "long", day: "numeric",
    });
  }
}

function skeletonRows(cols, rows = 4) {
  let html = "";
  for (let r = 0; r < rows; r++) {
    html += "<tr>";
    for (let c = 0; c < cols; c++) {
      html += `<td><div class="skeleton" style="height:14px;width:${60 + Math.random() * 30}%"></div></td>`;
    }
    html += "</tr>";
  }
  return html;
}

async function loadSidebar() {
  const slot = document.getElementById("sidebar-slot");
  if (!slot) return;
  const res = await fetch("partials/sidebar.html");
  slot.innerHTML = await res.text();

  const teacherName = sessionStorage.getItem("teacher_name") || "Admin";
  const usernameEl = document.getElementById("sidebar-username");
  const avatarEl = document.getElementById("sidebar-avatar");
  if (usernameEl) usernameEl.textContent = teacherName;
  if (avatarEl) avatarEl.textContent = teacherName.charAt(0).toUpperCase();

  document.getElementById("logout-link")?.addEventListener("click", (e) => {
    e.preventDefault();
    sessionStorage.removeItem("teacher_name");
    window.location.href = "index.html";
  });

  setActiveNavLink();
}

document.addEventListener("DOMContentLoaded", async () => {
  await loadSidebar();
  setActiveNavLink();
  setTodayDateChip();
});
