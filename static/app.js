async function health() {
  const status = document.getElementById("status");
  try {
    const r = await fetch("/api/health");
    const data = await r.json();
    if (!data.ok) {
      status.textContent = "Ollama: not reachable";
      status.className = "status bad";
      return;
    }
    status.textContent = "Ollama: OK";
    status.className = "status ok";
  } catch (e) {
    status.textContent = "Ollama: error";
    status.className = "status bad";
  }
}

function escapeHtml(str) {
  return str.replace(/[&<>"']/g, (m) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
  }[m]));
}

async function ask() {
  const q = document.getElementById("q");
  const out = document.getElementById("out");
  const sourcesBox = document.getElementById("sources");
  const btn = document.getElementById("ask");

  const question = q.value.trim();
  if (!question) return;

  btn.disabled = true;
  out.innerHTML = `<div class="muted">Thinking…</div>`;
  sourcesBox.textContent = "—";

  try {
    const r = await fetch("/api/ask", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question})
    });

    const data = await r.json();
    if (!r.ok) {
      out.innerHTML = `<div class="error">${escapeHtml(data.error || "Request failed")}</div>`;
      btn.disabled = false;
      return;
    }

    out.innerHTML = `<pre>${escapeHtml(data.answer || "")}</pre>`;

    const sources = data.sources || [];
    if (sources.length === 0) {
      sourcesBox.textContent = "—";
    } else {
      sourcesBox.innerHTML = sources.map(s => `<div class="src">${escapeHtml(s)}</div>`).join("");
    }
  } catch (e) {
    out.innerHTML = `<div class="error">Network error</div>`;
  } finally {
    btn.disabled = false;
  }
}

document.getElementById("ask").addEventListener("click", ask);
document.getElementById("q").addEventListener("keydown", (e) => {
  if (e.key === "Enter") ask();
});

health();