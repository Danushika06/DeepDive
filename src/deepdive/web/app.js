/* DeepDive frontend logic — no build step, plain ES modules-free JS. */
"use strict";

// ---------- tiny DOM helpers ----------
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html != null) n.innerHTML = html;
  return n;
};
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmtMs = (ms) => (ms >= 1000 ? (ms / 1000).toFixed(1) + "s" : Math.round(ms) + "ms");
const fmtUsd = (v) => (v == null ? "—" : "$" + Number(v).toFixed(4));
const pct = (v) => (v == null ? "—" : (Number(v) * 100).toFixed(0) + "%");

const NODE_TAGS = {
  planner: "Planning", searcher: "Tool use", retriever: "RAG",
  extractor: "Extract", critic: "Reflection", synthesizer: "Synthesis",
};

const SAMPLES = [
  "What are the main benefits and risks of intermittent fasting?",
  "How would I build a rocket company to compete with SpaceX?",
  "Compare solid-state and lithium-ion batteries.",
];

let evtSource = null;
let nodeRuns = {}; // node -> count

// ---------- tabs ----------
function initTabs() {
  $$(".tab").forEach((t) =>
    t.addEventListener("click", () => {
      $$(".tab").forEach((x) => x.classList.remove("active"));
      $$(".view").forEach((x) => x.classList.remove("active"));
      t.classList.add("active");
      $("#view-" + t.dataset.view).classList.add("active");
      if (t.dataset.view === "history") loadHistory();
      if (t.dataset.view === "evals" && !$("#eval-results").dataset.loaded) loadEvals();
    })
  );
}

// ---------- health ----------
async function pollHealth() {
  const pill = $("#health-pill");
  try {
    const r = await fetch("/health");
    const j = await r.json();
    pill.textContent = `${j.status} · ${j.reasoning_model.split("-").slice(0, 2).join("-")} · ${j.search_provider}`;
    pill.className = "status-pill ok";
  } catch {
    pill.textContent = "offline";
    pill.className = "status-pill bad";
  }
}

// ---------- flow graph ----------
function buildFlow(nodes) {
  const flow = $("#flow");
  flow.innerHTML = "";
  nodeRuns = {};
  nodes.forEach((name, i) => {
    const n = el("div", "node");
    n.id = "node-" + name;
    n.innerHTML = `
      <div class="node-runs" id="runs-${name}"></div>
      <div class="node-top">
        <span class="node-num">${i + 1}</span>
        <div><div class="node-name">${name}</div><div class="node-tag">${NODE_TAGS[name] || ""}</div></div>
      </div>
      <div class="node-meta" id="meta-${name}"></div>`;
    flow.appendChild(n);
  });
}

function markNode(span) {
  const name = span.node;
  const node = $("#node-" + name);
  if (!node) return;
  nodeRuns[name] = (nodeRuns[name] || 0) + 1;
  if (nodeRuns[name] > 1) {
    const r = $("#runs-" + name);
    r.style.display = "block";
    r.textContent = "×" + nodeRuns[name];
  }
  const meta = $("#meta-" + name);
  const cost = span.est_cost_usd != null ? " · " + fmtUsd(span.est_cost_usd) : "";
  meta.textContent = fmtMs(span.latency_ms) + cost;
  // pulse active, then settle to done/error
  node.classList.remove("done");
  node.classList.add(span.error ? "error" : "active");
  if (!span.error) {
    setTimeout(() => { node.classList.remove("active"); node.classList.add("done"); }, 450);
  }
}

// ---------- live trace ----------
function addTraceRow(step) {
  const trace = $("#trace");
  const empty = $(".trace-empty", trace);
  if (empty) empty.remove();
  const s = step.span;
  const row = el("div", "trace-row" + (s.error ? " err" : ""));
  const tok = s.tokens_in != null ? `<span><b>${s.tokens_in}</b>+${s.tokens_out || 0} tok</span>` : "";
  const cost = s.est_cost_usd != null ? `<span>${fmtUsd(s.est_cost_usd)}</span>` : "";
  row.innerHTML = `
    <span class="tr-idx">${step.index}</span>
    <span class="tr-node">${esc(s.node)}</span>
    <span class="tr-detail">${esc(s.error || s.detail || "")}</span>
    <span class="tr-stats"><span><b>${fmtMs(s.latency_ms)}</b></span>${tok}${cost}</span>`;
  trace.appendChild(row);
  trace.scrollTop = trace.scrollHeight;
}

// ---------- metrics ----------
function metricCard(label, value, cls = "") {
  return `<div class="metric"><div class="m-label">${label}</div><div class="m-value ${cls}">${value}</div></div>`;
}
function renderMetrics(report) {
  const m = report.metrics, t = report.trace;
  const fcls = m.citation_faithfulness >= 0.9 ? "good" : m.citation_faithfulness != null ? "warn" : "";
  $("#metrics").innerHTML =
    metricCard("Sources", `${m.n_unique_sources}/${m.n_sources_retrieved}`) +
    metricCard("Findings", m.n_findings) +
    metricCard("Conflicts", m.n_conflicts, m.n_conflicts > 0 ? "warn" : "") +
    metricCard("Extract valid", pct(m.extraction_valid_rate)) +
    metricCard("Citation faith.", pct(m.citation_faithfulness), fcls) +
    metricCard("Total cost", fmtUsd(t.total_cost_usd));
  $("#trace-summary").textContent =
    `${t.spans.length} steps · ${fmtMs(t.total_latency_ms)} · ${t.total_tokens_in + t.total_tokens_out} tok · ${fmtUsd(t.total_cost_usd)}`;
}

// ---------- conflicts ----------
function renderConflicts(conflicts) {
  const panel = $("#conflicts-panel");
  if (!conflicts || !conflicts.length) { panel.style.display = "none"; return; }
  panel.style.display = "block";
  $("#conflicts-count").textContent = conflicts.length;
  $("#conflicts").innerHTML = conflicts.map((c) => `
    <div class="conflict ${c.severity === "high" ? "high" : ""}">
      <div class="conflict-top">
        <span class="badge type">${esc(c.conflict_type)}</span>
        <span class="badge ${c.severity === "high" ? "danger" : "warn"}">${esc(c.severity)}</span>
      </div>
      <div class="conflict-summary">${esc(c.summary)}</div>
      <div class="conflict-src">sources: ${esc((c.source_ids || []).join(", "))}</div>
    </div>`).join("");
}

// ---------- report ----------
function renderMarkdown(md) {
  try {
    if (window.marked && window.DOMPurify) return DOMPurify.sanitize(marked.parse(md));
  } catch { /* fall through */ }
  return "<pre style='white-space:pre-wrap'>" + esc(md) + "</pre>";
}
function renderReport(report) {
  $("#report-panel").style.display = "block";
  $("#report").innerHTML = renderMarkdown(report.report_markdown || "");
  $("#report-meta").textContent = `${report.sub_questions.length} sub-questions`;
  const cites = report.citations || [];
  $("#citations").innerHTML = cites.length
    ? "<div class='muted' style='margin-bottom:6px'>Citations</div>" + cites.map((c) =>
        `<div class="citation-item"><span class="citation-marker">${esc(c.marker)}</span>
         <a href="${esc(c.url)}" target="_blank" rel="noopener">${esc(c.title || c.url)}</a></div>`).join("")
    : "";
}

// ---------- run research (SSE) ----------
function setRunning(on) {
  $("#run-btn").disabled = on;
  $("#stop-btn").disabled = !on;
  $("#flow-status").innerHTML = on ? '<span class="spinner"></span> running' : "done";
}
function stopRun() {
  if (evtSource) { evtSource.close(); evtSource = null; }
  setRunning(false);
}
function runResearch() {
  const q = $("#question").value.trim();
  if (q.length < 3) { $("#question").focus(); return; }
  const maxSrc = $("#max-sources").value || 4;

  // reset UI
  $("#error-banner").classList.add("hidden");
  $("#trace").innerHTML = "<div class='trace-empty muted'>waiting for first step…</div>";
  $("#metrics").innerHTML = "";
  $("#report-panel").style.display = "none";
  $("#conflicts-panel").style.display = "none";
  $("#trace-summary").textContent = "";
  buildFlow(Object.keys(NODE_TAGS));
  setRunning(true);

  const url = `/research/stream?question=${encodeURIComponent(q)}&max_sources=${encodeURIComponent(maxSrc)}`;
  evtSource = new EventSource(url);

  evtSource.addEventListener("start", (e) => {
    const d = JSON.parse(e.data);
    if (d.nodes) buildFlow(d.nodes);
  });
  evtSource.addEventListener("step", (e) => {
    const step = JSON.parse(e.data);
    markNode(step.span);
    addTraceRow(step);
  });
  evtSource.addEventListener("done", (e) => {
    const d = JSON.parse(e.data);
    renderMetrics(d.report);
    renderConflicts(d.report.conflicts);
    renderReport(d.report);
    stopRun();
  });
  evtSource.addEventListener("error", (e) => {
    let msg = "Stream interrupted.";
    try { if (e.data) msg = JSON.parse(e.data).message; } catch {}
    showError(msg);
    stopRun();
  });
}
function showError(msg) {
  const b = $("#error-banner");
  b.textContent = "⚠ " + msg;
  b.classList.remove("hidden");
}

// ---------- evals ----------
function renderEvals(results) {
  const root = $("#eval-results");
  root.dataset.loaded = "1";
  if (!results || !results.length) {
    root.innerHTML = "<div class='panel muted'>No results yet. Click <b>Run fresh</b> (takes a few minutes — live web + LLM judge).</div>";
    return;
  }
  const meta = new Set(["suite", "level", "n", "per_case", "error"]);
  root.innerHTML = results.map((r) => {
    if (r.error) return `<div class="panel eval-suite"><div class="panel-head"><h2>${esc(r.suite)}</h2><span class="badge danger">error</span></div><div class="muted">${esc(r.error)}</div></div>`;
    const rows = Object.entries(r).filter(([k]) => !meta.has(k)).map(([k, v]) => {
      const num = typeof v === "number";
      const bar = num && v <= 1 ? `<div class="bar"><span style="width:${Math.max(0, Math.min(1, v)) * 100}%"></span></div>` : "<div></div>";
      return `<div class="eval-metric"><span class="eval-mname">${esc(k)}</span><span class="eval-mval">${esc(v)}</span>${bar}</div>`;
    }).join("");
    return `<div class="panel eval-suite">
      <div class="panel-head"><h2>${esc(r.suite)}</h2><span class="badge type">${esc(r.level)}</span><span class="badge">n=${r.n}</span></div>
      ${rows}</div>`;
  }).join("");
}
async function loadEvals() {
  $("#eval-status").textContent = "loading…";
  try {
    const r = await fetch("/evals/last");
    const j = await r.json();
    renderEvals(j.results);
    $("#eval-status").textContent = j.results ? "showing last saved results" : "";
  } catch { $("#eval-status").textContent = "failed to load"; }
}
async function runEvals() {
  const suite = $("#eval-suite").value;
  $("#eval-status").innerHTML = '<span class="spinner"></span> running ' + suite + "…";
  $("#eval-run").disabled = true;
  try {
    const r = await fetch(`/evals/run?suite=${suite}`, { method: "POST" });
    const { job_id } = await r.json();
    const poll = setInterval(async () => {
      const jr = await fetch(`/evals/${job_id}`);
      const job = await jr.json();
      if (job.status === "done") {
        clearInterval(poll); renderEvals(job.results);
        $("#eval-status").textContent = "done"; $("#eval-run").disabled = false;
      } else if (job.status === "error") {
        clearInterval(poll); $("#eval-status").textContent = "error: " + job.error; $("#eval-run").disabled = false;
      }
    }, 2500);
  } catch (e) { $("#eval-status").textContent = "failed"; $("#eval-run").disabled = false; }
}

// ---------- history ----------
async function loadHistory() {
  const list = $("#history-list");
  list.innerHTML = "<div class='muted'>loading…</div>";
  try {
    const r = await fetch("/history");
    const items = await r.json();
    if (!items.length) { list.innerHTML = "<div class='muted'>No runs yet this session. Run a research query first.</div>"; return; }
    list.innerHTML = items.map((it) => `
      <div class="hist-item" data-id="${it.id}">
        <div class="hist-q">${esc(it.question)}</div>
        <div class="hist-meta"><span>${esc(it.created_at)}</span><span>${it.n_findings} findings</span><span>${it.n_conflicts} conflicts</span><span>faith ${pct(it.citation_faithfulness)}</span></div>
      </div>`).join("");
    $$(".hist-item", list).forEach((node) =>
      node.addEventListener("click", () => {
        $$(".hist-item").forEach((x) => x.classList.remove("sel"));
        node.classList.add("sel");
        openHistory(node.dataset.id);
      }));
  } catch { list.innerHTML = "<div class='muted'>failed to load history</div>"; }
}
async function openHistory(id) {
  const detail = $("#history-detail");
  detail.innerHTML = "<div class='muted'>loading…</div>";
  try {
    const r = await fetch("/history/" + id);
    const report = await r.json();
    const t = report.trace;
    const traceRows = t.spans.map((s, i) => `
      <div class="trace-row${s.error ? " err" : ""}">
        <span class="tr-idx">${i + 1}</span><span class="tr-node">${esc(s.node)}</span>
        <span class="tr-detail">${esc(s.error || s.detail || "")}</span>
        <span class="tr-stats"><span><b>${fmtMs(s.latency_ms)}</b></span>${s.est_cost_usd != null ? `<span>${fmtUsd(s.est_cost_usd)}</span>` : ""}</span>
      </div>`).join("");
    const cites = (report.citations || []).map((c) =>
      `<div class="citation-item"><span class="citation-marker">${esc(c.marker)}</span><a href="${esc(c.url)}" target="_blank">${esc(c.title || c.url)}</a></div>`).join("");
    $("#history-detail-meta").textContent =
      `${report.metrics.n_findings} findings · ${report.metrics.n_conflicts} conflicts · ${fmtUsd(t.total_cost_usd)}`;
    detail.innerHTML = `
      <div class="muted" style="margin-bottom:8px">Trace</div>
      <div class="trace" style="max-height:200px;margin-bottom:16px">${traceRows}</div>
      <article class="report markdown">${renderMarkdown(report.report_markdown || "")}</article>
      <div class="citations">${cites}</div>`;
  } catch { detail.innerHTML = "<div class='muted'>failed to load run</div>"; }
}

// ---------- init ----------
document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  pollHealth();
  $("#samples").innerHTML = SAMPLES.map((s) => `<span class="chip">${esc(s)}</span>`).join("");
  $$(".chip").forEach((c) => c.addEventListener("click", () => { $("#question").value = c.textContent; }));
  $("#run-btn").addEventListener("click", runResearch);
  $("#stop-btn").addEventListener("click", stopRun);
  $("#question").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) runResearch(); });
  $("#eval-load").addEventListener("click", loadEvals);
  $("#eval-run").addEventListener("click", runEvals);
  $("#history-refresh").addEventListener("click", loadHistory);
});
