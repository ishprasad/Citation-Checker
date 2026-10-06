"use strict";

/* Citation Checker UI: vanilla JS, no build step.
 *
 * Routes: "#/" start page, "#/job/<id>" progress + report for a job.
 * A job page connects to the SSE stream, applies the snapshot, then updates citation
 * markers and the detail panel incrementally as verdicts arrive.
 */

// ------------------------------------------------------------------ constants

const LABELS = {
  supported: { text: "Supported", icon: "✓" },
  overstated: { text: "Possibly overstated", icon: "!" },
  contradicted: { text: "Contradicted by cited text", icon: "✕" },
  not_found: { text: "Not found in cited text", icon: "?" },
  unverifiable: { text: "Could not be verified", icon: "–" },
};
const CHIP_TEXT = {
  contradicted: "Contradicted",
  not_found: "Not found",
  overstated: "Overstated",
  unverifiable: "Unverifiable",
  supported: "Supported",
};
const SUMMARY_ORDER = ["contradicted", "not_found", "overstated", "unverifiable", "supported"];
const SEVERITY = ["contradicted", "not_found", "overstated", "unverifiable", "supported"];
const ISSUES = new Set(["contradicted", "not_found", "overstated"]);
const FAILURE = {
  scope_inflation: ["Scope inflation", false],
  number_drift: ["Number drift", false],
  hypothesis_as_finding: ["Hypothesis stated as finding", false],
  secondary_citation: ["Secondary citation", false],
  retracted_source: ["Retracted source", true],
  off_topic_citation: ["Possibly the wrong reference", true],
  correlation_as_causation: ["Correlation stated as causation", false],
  significance_inflation: ["Significance overstated", false],
  corrected_source: ["Source has a correction", false],
  expression_of_concern: ["Expression of concern", true],
};
const STAGE_TEXT = {
  parse: "Parsing paper",
  resolve: "Resolving references",
  extract: "Extracting claims",
  retrieve: "Retrieving evidence",
  judge: "Judging claims",
  escalate: "Escalating to Ultra",
};
const ROLE_TEXT = {
  extract: "Claim extraction · Nemotron Nano",
  judge: "Judging · Nemotron Super",
  escalate: "Escalation · Nemotron Ultra",
  vision: "Scanned pages · Nemotron Nano Omni",
  embed: "Embeddings",
};
const FINISHED = new Set(["done", "partial", "failed"]);
const MUTATION_TEXT = {
  number_change: "number changed",
  scope_inflation: "scope inflated",
  hedge_removal: "hedge removed",
  citation_swap: "citation swapped to an unrelated paper",
};
const TEXT_STATUS = {
  full: "Full text",
  abstract_only: "Abstract only",
  unavailable: "Not openly available",
};

// -------------------------------------------------------------------- helpers

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ESC[c]);
const pid = (claimId, key) => `${claimId}|${key}`;
const pct = (x) => `${Math.round((x ?? 0) * 100)}%`;
const fmtTokens = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${Math.round(n / 1e3)}k` : String(n || 0));

async function api(path, options = {}) {
  const res = await fetch(path, options);
  let body = null;
  try { body = await res.json(); } catch { /* empty or non-JSON */ }
  if (!res.ok) {
    const detail = body && (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail));
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return body;
}

function toast(message) {
  const el = $("#toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (el.hidden = true), 4000);
}

function refLink(ref) {
  if (!ref) return null;
  if (ref.arxiv_id) return `https://arxiv.org/abs/${ref.arxiv_id}`;
  if (ref.doi) return `https://doi.org/${ref.doi}`;
  return ref.url || null;
}

// ---------------------------------------------------------------------- state

let S = freshState();

function freshState() {
  return {
    jobId: null,
    job: null,
    paper: null,
    text: "",
    sentences: [],
    sentById: new Map(),
    refs: new Map(),
    claimsBySentence: new Map(),
    claimById: new Map(),
    verdicts: new Map(),
    duplicates: new Set(), // pair ids merged into a sibling verdict
    demo: null, // planted-error info and score for the demo paper
    escalating: new Set(),
    coverage: null,
    filter: { labels: new Set(), issuesOnly: false, query: "" },
    selected: null, // {sid, m}
    es: null,
    poll: null,
    reportLoading: null,
    showStages: false,
  };
}

function teardown() {
  if (S.es) S.es.close();
  if (S.poll) clearInterval(S.poll);
}

// --------------------------------------------------------------------- router

function route() {
  const match = location.hash.match(/^#\/job\/([\w-]+)/);
  if (match) openJob(match[1]);
  else showStart();
}

window.addEventListener("hashchange", route);
document.addEventListener("DOMContentLoaded", () => {
  initStart();
  initJobView();
  route();
});

// ---------------------------------------------------------------- start page

let pdfFile = null;

function showStart() {
  teardown();
  S = freshState();
  $("#job").hidden = true;
  $("#start").hidden = false;
  document.title = "Citation Checker";
  setStartBusy(false);
  $("#arxiv-input").focus();
}

function setStartError(message) {
  const el = $("#start-error");
  el.textContent = message || "";
  el.hidden = !message;
}

function setStartBusy(busy) {
  $("#start-submit").disabled = busy;
  $("#demo-button").disabled = busy;
  $("#start-submit").textContent = busy ? "Starting…" : "Check citations";
}

function setPdf(file) {
  pdfFile = file;
  const zone = $("#drop-zone");
  zone.classList.toggle("has-file", !!file);
  $("#drop-label").textContent = file ? `${file.name} (${(file.size / 1e6).toFixed(1)} MB)` : "Or drop a PDF here, or click to choose one";
  if (file) $("#arxiv-input").value = "";
}

function initStart() {
  const zone = $("#drop-zone");
  const fileInput = $("#pdf-input");
  zone.addEventListener("click", () => fileInput.click());
  zone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener("change", () => setPdf(fileInput.files[0] || null));
  ["dragenter", "dragover"].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.remove("dragging"); }));
  zone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (!file) return;
    if (file.type && file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
      setStartError("Only PDF files can be uploaded.");
      return;
    }
    setStartError("");
    setPdf(file);
  });
  $("#arxiv-input").addEventListener("input", () => { if ($("#arxiv-input").value.trim()) setPdf(null); setStartError(""); });

  $("#start-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const mode = $("input[name=mode]:checked").value;
    const arxiv = $("#arxiv-input").value.trim();
    setStartError("");
    if (!arxiv && !pdfFile) {
      setStartError("Enter an arXiv ID or link, or choose a PDF.");
      $("#arxiv-input").focus();
      return;
    }
    setStartBusy(true);
    try {
      let result;
      if (pdfFile) {
        const form = new FormData();
        form.append("file", pdfFile);
        form.append("mode", mode);
        result = await api("/api/jobs", { method: "POST", body: form });
      } else {
        result = await api("/api/jobs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ arxiv_id: arxiv, mode }),
        });
      }
      location.hash = `#/job/${result.id}`;
    } catch (err) {
      setStartError(err.message);
      setStartBusy(false);
    }
  });

  $("#demo-button").addEventListener("click", async () => {
    setStartError("");
    setStartBusy(true);
    try {
      const result = await api("/api/demo", { method: "POST" });
      location.hash = `#/job/${result.id}`;
    } catch (err) {
      setStartError(err.message);
      setStartBusy(false);
    }
  });
}

// ------------------------------------------------------------------ job page

function initJobView() {
  $("#paper").addEventListener("click", (e) => {
    const btn = e.target.closest(".cite");
    if (!btn) return;
    select(btn.dataset.sid, Number(btn.dataset.m));
  });
  $("#detail").addEventListener("click", (e) => {
    const go = e.target.closest("[data-go-sid]");
    if (go) { select(go.dataset.goSid, Number(go.dataset.goM), true); return; }
    if (e.target.closest("[data-close]")) { select(null); return; }
    if (e.target.closest("[data-expand]")) expandJob();
  });
  $("#paper-head").addEventListener("click", (e) => {
    if (e.target.closest("[data-expand]")) expandJob();
  });
  $("#progress").addEventListener("click", (e) => {
    if (e.target.closest("[data-toggle-stages]")) { S.showStages = !S.showStages; renderProgress(); }
  });
  $("#summary").addEventListener("click", (e) => {
    const chip = e.target.closest("[data-label]");
    if (!chip) return;
    const label = chip.dataset.label;
    if (S.filter.labels.has(label)) S.filter.labels.delete(label);
    else S.filter.labels.add(label);
    renderSummary();
    applyFilter();
  });
  $("#issues-only").addEventListener("change", (e) => { S.filter.issuesOnly = e.target.checked; applyFilter(); });
  let searchTimer = null;
  $("#search").addEventListener("input", (e) => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => { S.filter.query = e.target.value.trim().toLowerCase(); applyFilter(); if (!S.selected) renderDetail(); }, 150);
  });
  $("#next-issue").addEventListener("click", nextIssue);
  setupMenu("#export-button", "#export-menu");
  setupMenu("#cost-button", "#cost-panel");
  document.addEventListener("keydown", (e) => {
    if ($("#job").hidden) return;
    if (e.key === "Escape" && S.selected) select(null);
    if (e.key === "j" && !["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) nextIssue();
  });
}

function setupMenu(buttonSel, panelSel) {
  const button = $(buttonSel);
  const panel = $(panelSel);
  button.addEventListener("click", (e) => {
    e.stopPropagation();
    const open = panel.hidden;
    $$(".menu-panel").forEach((p) => (p.hidden = true));
    $$(".menu > button").forEach((b) => b.setAttribute("aria-expanded", "false"));
    panel.hidden = !open;
    button.setAttribute("aria-expanded", String(open));
  });
  document.addEventListener("click", (e) => {
    if (!panel.hidden && !panel.contains(e.target)) { panel.hidden = true; button.setAttribute("aria-expanded", "false"); }
  });
}

async function openJob(jobId) {
  if (S.jobId === jobId && S.job) return;
  teardown();
  S = freshState();
  S.jobId = jobId;
  $("#start").hidden = true;
  $("#job").hidden = false;
  $("#search").value = "";
  $("#issues-only").checked = false;
  $("#export-md").href = `/api/jobs/${jobId}/export?format=md`;
  $("#export-json").href = `/api/jobs/${jobId}/export?format=json`;
  $("#paper").innerHTML = `<p class="empty">Loading…</p>`;
  $("#detail").innerHTML = "";
  $("#paper-head").innerHTML = "";
  try {
    S.job = await api(`/api/jobs/${jobId}`);
  } catch (err) {
    renderFailed("This check could not be found", err.message);
    return;
  }
  renderAllChrome();
  connect();
}

function connect() {
  if (S.es) S.es.close();
  const es = new EventSource(`/api/jobs/${S.jobId}/events`);
  S.es = es;
  const on = (name, fn) => es.addEventListener(name, (e) => fn(JSON.parse(e.data)));
  on("snapshot", applySnapshot);
  on("paper", () => loadReport());
  on("progress", (p) => {
    if (!S.job) return;
    const i = S.job.progress.findIndex((x) => x.stage === p.stage);
    if (i >= 0) S.job.progress[i] = p;
    renderProgress();
  });
  on("reference", (r) => { S.refs.set(r.key, { ...(S.refs.get(r.key) || {}), ...r }); if (S.selected) renderDetail(); });
  on("coverage", (c) => { S.coverage = c; renderHead(); });
  on("claims", ({ sentence_id, claims }) => { setClaims(sentence_id, claims); updateMarkers(sentence_id); refreshDetailFor(sentence_id); });
  on("escalating", ({ pair_id }) => { S.escalating.add(pair_id); refreshDetailFor(pair_id.split(".c")[0]); });
  on("verdict", ({ verdict, claim }) => {
    addClaim(claim);
    const id = pid(verdict.claim_id, verdict.reference_key);
    S.verdicts.set(id, verdict);
    if (verdict.duplicate_of) S.duplicates.add(id);
    else S.duplicates.delete(id);
    S.escalating.delete(id);
    updateMarkers(claim.citing_sentence_id);
    renderSummary();
    refreshDetailFor(claim.citing_sentence_id);
    if (!S.selected) renderDetail();
  });
  on("status", ({ status, message }) => {
    if (!S.job) return;
    S.job.status = status;
    S.job.message = message;
    renderAllChrome();
  });
  on("done", async ({ status, message }) => {
    es.close();
    S.job.status = status;
    S.job.message = message;
    await refreshJob();
    if (S.paper) await loadReport(true);
    else if (status === "failed") renderFailed("This paper could not be checked", message);
    renderAllChrome();
  });
  es.onerror = () => {
    // EventSource reconnects automatically; the next snapshot brings us back in sync.
  };
  if (S.poll) clearInterval(S.poll);
  S.poll = setInterval(async () => {
    if (S.job && FINISHED.has(S.job.status)) { clearInterval(S.poll); return; }
    await refreshJob();
  }, 4000);
}

async function refreshJob() {
  try {
    const job = await api(`/api/jobs/${S.jobId}`);
    S.job = job;
    renderProgress();
    renderCost();
    renderHead();
  } catch { /* transient */ }
}

function applySnapshot(snap) {
  S.job = snap.job;
  applyData(snap);
  if (snap.paper && !S.text) loadReport();
  renderAllChrome();
  if (S.text) { updateMarkers(); renderDetail(); }
  else if (!snap.paper && FINISHED.has(snap.job.status) && snap.job.status === "failed") {
    renderFailed("This paper could not be checked", snap.job.message);
  } else if (!snap.paper) {
    $("#paper").innerHTML = `<p class="empty">Fetching the paper and its bibliography…</p>`;
    renderDetail();
  }
}

function applyData(data) {
  if (data.paper) {
    S.paper = { ...(S.paper || {}), ...data.paper };
    if (data.paper.text) S.text = data.paper.text;
  }
  if (data.sentences) {
    S.sentences = data.sentences;
    S.sentById = new Map(data.sentences.map((s) => [s.id, s]));
  }
  for (const r of data.references || []) S.refs.set(r.key, r);
  if (data.claims) {
    S.claimsBySentence = new Map();
    S.claimById = new Map();
    data.claims.forEach(addClaim);
  }
  for (const v of data.verdicts || []) S.verdicts.set(pid(v.claim_id, v.reference_key), v);
  for (const d of data.duplicates || []) { S.duplicates.add(d.pair_id); S.verdicts.delete(d.pair_id); }
  if (data.summary) S.coverage = data.summary.coverage;
  if (data.demo !== undefined) S.demo = data.demo;
}

function addClaim(claim) {
  if (S.claimById.has(claim.id)) return;
  S.claimById.set(claim.id, claim);
  const list = S.claimsBySentence.get(claim.citing_sentence_id) || [];
  list.push(claim);
  S.claimsBySentence.set(claim.citing_sentence_id, list);
}

function setClaims(sid, claims) {
  S.claimsBySentence.set(sid, []);
  claims.forEach((c) => { S.claimById.delete(c.id); addClaim(c); });
}

function loadReport(force = false) {
  if (S.reportLoading && !force) return S.reportLoading;
  S.reportLoading = (async () => {
    const report = await api(`/api/jobs/${S.jobId}/report`);
    S.job = report.job;
    applyData(report);
    renderPaper();
    renderAllChrome();
    renderDetail();
  })().catch((err) => toast(err.message)).finally(() => { S.reportLoading = null; });
  return S.reportLoading;
}

async function expandJob() {
  try {
    const result = await api(`/api/jobs/${S.jobId}/expand`, { method: "POST" });
    S.job = result.job;
    toast("Checking the rest of the paper…");
    S.selected = null;
    await loadReport(true);
    connect();
  } catch (err) {
    toast(err.message);
  }
}

// ----------------------------------------------------------------- rendering

function renderAllChrome() {
  renderHead();
  renderProgress();
  renderSummary();
  renderCost();
}

function renderFailed(title, message) {
  $("#paper-head").innerHTML = "";
  $("#progress").innerHTML = "";
  $("#paper").innerHTML = `
    <div class="failed-card">
      <h2>${esc(title)}</h2>
      <p>${esc(message || "Something went wrong.")}</p>
      <p><a href="#/" class="btn primary">Check another paper</a></p>
    </div>`;
  $("#detail").innerHTML = "";
}

function renderHead() {
  const el = $("#paper-head");
  const job = S.job;
  if (!job) return;
  const p = S.paper;
  const title = p?.title || job.title || (job.arxiv_id ? `arXiv:${job.arxiv_id}` : "Checking paper…");
  document.title = `${title} · Citation Checker`;
  const authors = p?.authors?.length ? (p.authors.length > 6 ? `${p.authors.slice(0, 6).join(", ")}, et al.` : p.authors.join(", ")) : "";
  const link = p?.arxiv_id ? `<a href="https://arxiv.org/abs/${esc(p.arxiv_id)}" target="_blank" rel="noopener">arXiv:${esc(p.arxiv_id)}</a>` : "";
  const parsed = p?.parsed_from === "latex" ? "Parsed from the LaTeX source" : p?.parsed_from ? "Parsed from PDF; citation matching may be less accurate" : "";
  const notices = [];
  if (job.status === "queued") notices.push(`<div class="notice">Waiting for a free slot to start this check…</div>`);
  if (job.status === "failed" && p) notices.push(`<div class="notice bad">${esc(job.message || "This check stopped early.")}</div>`);
  if (job.status === "partial") notices.push(`<div class="notice warn">${esc(job.message || "This check stopped early; results so far are shown.")}</div>`);
  for (const note of p?.notes || []) notices.push(`<div class="notice warn">${esc(note)}</div>`);
  if (job.source === "demo" && S.demo) notices.unshift(renderDemoNotice());
  if (job.mode === "quick" && job.sentences_total > job.sentences_checked && job.sentences_checked) {
    const done = FINISHED.has(job.status);
    notices.push(`<div class="notice">Quick check: ${job.sentences_checked} of ${job.sentences_total} citing sentences checked.
      ${done ? `<button class="btn small" type="button" data-expand>Check the rest</button>` : ""}</div>`);
  }
  const cov = S.coverage && S.coverage.total ? `
    <div class="coverage">
      <span>${esc(S.coverage.line)}</span>
      <details><summary>What does this mean?</summary>
        <p>Verdicts for abstract-only and unavailable references are limited by open access, not by the checker.
        Citations to those references are marked “Could not be verified” when the available text cannot settle the claim.</p>
      </details>
    </div>` : "";
  el.innerHTML = `
    <h1>${esc(title)}</h1>
    <div class="paper-meta">${[esc(authors), link, esc(parsed)].filter(Boolean).join(" · ")}</div>
    ${cov}
    ${notices.length ? `<div class="notices">${notices.join("")}</div>` : ""}`;
}

function renderProgress() {
  const el = $("#progress");
  const job = S.job;
  if (!job || !job.progress) return;
  const finished = FINISHED.has(job.status);
  el.classList.toggle("compact", finished && !S.showStages);
  const stages = job.progress.map((p) => {
    const name = STAGE_TEXT[p.stage] || p.label;
    const total = p.total || 0;
    const ratio = p.state === "done" ? 1 : total ? Math.min(1, p.done / total) : 0;
    let count;
    if (p.stage === "parse") count = p.state === "done" ? "Done" : p.state === "running" ? "Working…" : "Waiting";
    else if (p.stage === "escalate" && p.state === "done" && !total) count = "None needed";
    else count = total || p.state !== "pending" ? `${p.done} / ${total}` : "Waiting";
    return `
      <div class="stage ${p.state}" aria-label="${esc(name)}: ${esc(count)}">
        <div class="stage-name"><span class="stage-icon" aria-hidden="true"></span>${esc(name)}</div>
        <div class="stage-count">${esc(count)}</div>
        <div class="bar"><span style="width:${(ratio * 100).toFixed(1)}%"></span></div>
      </div>`;
  }).join("");
  const verdictCount = liveVerdicts().length;
  const escalated = liveVerdicts().filter((v) => v.escalated).length;
  const line = finished
    ? `<div class="progress-line">${job.status === "done" ? "Check complete" : job.status === "partial" ? "Check stopped early" : "Check failed"}
        · ${verdictCount} citation verdicts · ${escalated} escalated to Ultra
        <button class="btn link" type="button" data-toggle-stages>${S.showStages ? "Hide" : "Show"} stages</button></div>`
    : "";
  el.innerHTML = `${line}<div class="stages">${stages}</div>`;
}

function liveVerdicts() {
  return Array.from(S.verdicts.values()).filter((v) => !v.duplicate_of);
}

function verdictCounts() {
  const counts = Object.fromEntries(SUMMARY_ORDER.map((l) => [l, 0]));
  for (const v of liveVerdicts()) counts[v.label] = (counts[v.label] || 0) + 1;
  return counts;
}

function renderSummary() {
  const counts = verdictCounts();
  $("#summary").innerHTML = SUMMARY_ORDER.map((label) => `
    <button type="button" class="chip" data-label="${label}" aria-pressed="${S.filter.labels.has(label)}"
            title="Show only: ${esc(LABELS[label].text)}">
      <span class="dot v-${label}" aria-hidden="true"></span>${esc(CHIP_TEXT[label])} <span class="n">${counts[label]}</span>
    </button>`).join("");
}

function renderCost() {
  const usage = S.job?.usage;
  const per = usage?.per_model || [];
  const tokens = (usage?.total_prompt_tokens || 0) + (usage?.total_completion_tokens || 0);
  const cost = usage?.total_estimated_cost_usd || 0;
  $("#cost-summary").textContent = tokens ? `${fmtTokens(tokens)} tokens${cost ? ` · $${cost.toFixed(3)}` : ""}` : "Cost";
  const order = ["extract", "judge", "escalate", "vision", "embed"];
  const rows = per.slice().sort((a, b) => order.indexOf(a.role) - order.indexOf(b.role)).map((m) => `
    <tr><td title="${esc(m.model)}">${esc(ROLE_TEXT[m.role] || m.role)}</td><td>${m.calls}</td>
      <td>${fmtTokens(m.prompt_tokens)}</td><td>${fmtTokens(m.completion_tokens)}</td>
      <td>${cost ? `$${m.estimated_cost_usd.toFixed(4)}` : "–"}</td></tr>`).join("");
  const verdicts = liveVerdicts();
  const escalated = verdicts.filter((v) => v.escalated);
  const changed = escalated.filter((v) => v.original_label && v.original_label !== v.label);
  $("#cost-panel").innerHTML = per.length ? `
    <h3>Tokens and estimated cost for this check</h3>
    <table>
      <thead><tr><th>Model</th><th>Calls</th><th>In</th><th>Out</th><th>Cost</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <p>The cheap model handles every claim; the large model reads full papers only for the
      ${escalated.length} of ${verdicts.length} verdicts that needed a second look
      (it changed the label in ${changed.length}).</p>
    ${cost ? "" : "<p>Set the PRICE_* values in .env to see estimated costs.</p>"}
    <p>Repeated checks reuse cached model responses.</p>`
    : `<p>No model calls yet.</p>`;
}

// ---------------------------------------------------------------- paper text

function paragraphs(text) {
  const out = [];
  const re = /\n\s*\n/g;
  let start = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > start) out.push([start, m.index]);
    start = m.index + m[0].length;
  }
  if (start < text.length) out.push([start, text.length]);
  return out;
}

function renderPaper() {
  const text = S.text;
  if (!text) return;
  const headings = new Map((S.paper?.sections || []).filter((s) => s.title).map((s) => [s.start, s.title]));
  const sentences = S.sentences.slice().sort((a, b) => a.start - b.start);
  let si = 0;
  const html = [];
  for (const [ps, pe] of paragraphs(text)) {
    const raw = text.slice(ps, pe);
    if (headings.has(ps) && raw.trim() === headings.get(ps)) {
      html.push(`<h2>${esc(raw.trim())}</h2>`);
      continue;
    }
    let out = "";
    let pos = ps;
    while (si < sentences.length && sentences[si].start < pe) {
      const s = sentences[si++];
      if (s.start < ps) continue;
      out += esc(text.slice(pos, s.start));
      out += `<span class="sent" data-sid="${s.id}">`;
      let q = s.start;
      s.marker_spans.forEach(([a, b], m) => {
        if (a < q || b > s.end) return;
        out += esc(text.slice(q, a));
        // Keep trailing punctuation on the marker's line ("[3]." never wraps before the period).
        // Long author-year markers must stay free to wrap.
        const punct = b - a <= 24 ? text.slice(b, Math.min(s.end, b + 2)).match(/^[.,;:)\]]+/) : null;
        const tail = punct ? punct[0] : "";
        const button = `<button type="button" class="cite" data-sid="${s.id}" data-m="${m}">${esc(text.slice(a, b))}</button>`;
        out += tail ? `<span class="nowrap">${button}${esc(tail)}</span>` : button;
        q = b + tail.length;
      });
      out += `${esc(text.slice(q, s.end))}</span>`;
      pos = s.end;
    }
    out += esc(text.slice(pos, pe));
    html.push(`<p>${out}</p>`);
  }
  $("#paper").innerHTML = html.join("");
  updateMarkers();
  if (S.selected) markSelected();
}

// --------------------------------------------------------------- marker state

function markerItems(sid, m) {
  const s = S.sentById.get(sid);
  if (!s) return { status: "unchecked", items: [] };
  if (!s.checked) return { status: "unchecked", items: [] };
  const keys = (s.marker_keys && s.marker_keys[m]) || s.reference_keys;
  const claims = S.claimsBySentence.get(sid);
  if (!claims) return { status: "pending", items: keys.map((k) => ({ claim: null, key: k, verdict: null })) };
  const items = [];
  for (const c of claims) {
    for (const k of c.reference_keys) {
      if (!keys.includes(k)) continue;
      const id = pid(c.id, k);
      const verdict = S.verdicts.get(id);
      if (S.duplicates.has(id) || (verdict && verdict.duplicate_of)) continue; // restates a sibling verdict
      items.push({ claim: c, key: k, verdict });
    }
  }
  const labels = items.filter((x) => x.verdict).map((x) => x.verdict.label);
  const pending = items.length === 0 || items.some((x) => !x.verdict);
  const worst = SEVERITY.find((l) => labels.includes(l));
  let status = worst || "pending";
  if (pending && (!worst || worst === "supported") && !FINISHED.has(S.job?.status)) status = "pending";
  return { status, items, pending };
}

function updateMarkers(sid) {
  const sel = sid ? `.cite[data-sid="${sid}"]` : ".cite";
  for (const btn of $$(sel, $("#paper"))) {
    const { status } = markerItems(btn.dataset.sid, Number(btn.dataset.m));
    btn.className = `cite ${status === "unchecked" ? "unchecked" : `v-${status}`}`;
    btn.dataset.status = status;
    const label = status === "unchecked" ? "Not checked in this quick check" : status === "pending" ? "Checking…" : LABELS[status].text;
    btn.setAttribute("aria-label", `Citation ${btn.textContent}: ${label}`);
    btn.title = label;
    if (status === "unchecked") btn.tabIndex = -1;
  }
  applyFilter(sid);
  if (S.selected) markSelected();
}

function matchesFilter(sid, m, status, items) {
  const f = S.filter;
  if (status === "unchecked") return !f.labels.size && !f.issuesOnly && !f.query;
  if (f.labels.size && !items.some((x) => x.verdict && f.labels.has(x.verdict.label))) return false;
  if (f.issuesOnly && !items.some((x) => x.verdict && ISSUES.has(x.verdict.label))) return false;
  if (f.query) {
    const s = S.sentById.get(sid);
    const hay = [s?.text, ...items.flatMap((x) => {
      const ref = S.refs.get(x.key);
      return [x.claim?.claim_text, x.verdict?.attributed_claim, ref?.title, ref?.label, x.verdict?.explanation];
    })].join(" ").toLowerCase();
    if (!hay.includes(f.query)) return false;
  }
  return true;
}

function applyFilter(sid) {
  const sel = sid ? `.cite[data-sid="${sid}"]` : ".cite";
  const active = S.filter.labels.size || S.filter.issuesOnly || S.filter.query;
  for (const btn of $$(sel, $("#paper"))) {
    const m = Number(btn.dataset.m);
    const { status, items } = markerItems(btn.dataset.sid, m);
    btn.classList.toggle("dim", !!active && !matchesFilter(btn.dataset.sid, m, status, items));
  }
}

function markSelected() {
  $$(".cite.selected, .sent.selected", $("#paper")).forEach((el) => el.classList.remove("selected"));
  if (!S.selected) return;
  const { sid, m } = S.selected;
  $(`.cite[data-sid="${sid}"][data-m="${m}"]`, $("#paper"))?.classList.add("selected");
  $(`.sent[data-sid="${sid}"]`, $("#paper"))?.classList.add("selected");
}

function select(sid, m, scroll = false) {
  S.selected = sid ? { sid, m } : null;
  markSelected();
  renderDetail();
  if (sid && scroll) {
    $(`.cite[data-sid="${sid}"][data-m="${m}"]`, $("#paper"))?.scrollIntoView({ block: "center", behavior: "smooth" });
  }
  if (sid) {
    $("#detail").scrollTop = 0;
    // Single-column layout: the detail panel sits below the paper, so bring it into view.
    if (window.matchMedia("(max-width: 900px)").matches) $("#detail").scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function refreshDetailFor(sid) {
  if (S.selected && S.selected.sid === sid) renderDetail();
}

function nextIssue() {
  const buttons = $$(".cite", $("#paper")).filter((b) => {
    const { status, items } = markerItems(b.dataset.sid, Number(b.dataset.m));
    return ISSUES.has(status) && matchesFilter(b.dataset.sid, Number(b.dataset.m), status, items);
  });
  if (!buttons.length) { toast("No citations need a human check so far."); return; }
  let idx = 0;
  if (S.selected) {
    const all = $$(".cite", $("#paper"));
    const current = all.findIndex((b) => b.dataset.sid === S.selected.sid && Number(b.dataset.m) === S.selected.m);
    idx = buttons.findIndex((b) => all.indexOf(b) > current);
    if (idx < 0) idx = 0;
  }
  const btn = buttons[idx];
  select(btn.dataset.sid, Number(btn.dataset.m), true);
}

// ------------------------------------------------------------- detail panel

function renderDetail() {
  const el = $("#detail");
  if (!S.job) { el.innerHTML = ""; return; }
  if (S.selected) el.innerHTML = renderCitation(S.selected.sid, S.selected.m);
  else el.innerHTML = renderOverview();
}

function issueEntries() {
  const entries = [];
  for (const btn of $$(".cite", $("#paper"))) {
    const sid = btn.dataset.sid;
    const m = Number(btn.dataset.m);
    const { status, items } = markerItems(sid, m);
    if (status === "unchecked" || status === "pending") continue;
    entries.push({ sid, m, status, items, marker: btn.textContent });
  }
  return entries;
}

function renderOverview() {
  const legend = SUMMARY_ORDER.concat(["pending"]).map((l) => `
    <div class="legend-row"><span class="cite v-${l}">[1]</span>
      <span>${l === "pending" ? "Being checked" : esc(LABELS[l].text)}</span></div>`).join("");
  const entries = issueEntries();
  const query = S.filter.query;
  let listTitle;
  let list;
  if (query) {
    list = entries.filter((e) => matchesFilter(e.sid, e.m, e.status, e.items));
    listTitle = `${list.length} citation${list.length === 1 ? "" : "s"} matching “${esc(query)}”`;
  } else {
    list = entries.filter((e) => ISSUES.has(e.status));
    listTitle = `Needs human check (${list.length})`;
  }
  const items = list.slice(0, 200).map((e) => {
    const first = e.items.find((x) => x.verdict && x.verdict.label === e.status) || e.items[0];
    const what = first?.verdict?.attributed_claim || first?.claim?.claim_text || S.sentById.get(e.sid)?.text || "";
    return `<li><button type="button" data-go-sid="${e.sid}" data-go-m="${e.m}">
      <span class="dot v-${e.status}" aria-hidden="true"></span>
      <span><strong>${esc(e.marker)}</strong> ${esc(LABELS[e.status]?.text || "")}${S.demo?.planted_by_sentence?.[e.sid] ? ` <span class="badge planted-badge">Planted</span>` : ""}<br>
      <span class="what">${esc(what.length > 150 ? `${what.slice(0, 150)}…` : what)}</span></span></button></li>`;
  }).join("");
  const running = !FINISHED.has(S.job.status);
  return `
    <h2>How to read this report</h2>
    <p class="muted">Each citation in the paper is coloured by what the cited work says about the claim attached to it.
      Select a citation to see the claim and the evidence side by side.</p>
    <div class="legend">${legend}</div>
    <h3>${listTitle}</h3>
    ${items ? `<ul class="issue-list">${items}</ul>` : `<p class="muted">${running ? "Results appear here as citations are checked." : query ? "No citations match." : "No citations were flagged."}</p>`}`;
}

function renderCitation(sid, m) {
  const s = S.sentById.get(sid);
  const { status, items } = markerItems(sid, m);
  const planted = S.demo?.planted_by_sentence?.[sid];
  const plantedHtml = planted ? `
    <div class="planted">
      <div class="planted-title">Planted error · ${esc(MUTATION_TEXT[planted.mutation] || planted.mutation)}</div>
      <div>${esc(planted.description)}.</div>
      <details><summary>Original sentence</summary><p>${esc(planted.original)}</p></details>
    </div>` : "";
  const head = `<button type="button" class="btn small close" data-close>← Overview</button>
    <h2>Citation ${esc(S.text.slice(...s.marker_spans[m]))}</h2>
    ${plantedHtml}
    <p class="sentence">${esc(s.text)}</p>
    <p class="muted" style="font-size:12.5px;margin:0">${esc(s.section || "")}</p>`;
  if (status === "unchecked") {
    return `${head}<div class="card"><p>This citation was not part of the quick check.</p>
      ${FINISHED.has(S.job.status) ? `<button class="btn small" type="button" data-expand>Check the rest of the paper</button>` : ""}</div>`;
  }
  if (!items.length || items.every((x) => !x.claim)) {
    return `${head}<div class="card"><p class="muted"><span class="spinner"></span> Extracting the claim behind this citation…</p></div>`;
  }
  // One card per cited work; each claim attached to it is a numbered section of that card.
  const groups = new Map();
  for (const item of items) {
    if (!groups.has(item.key)) groups.set(item.key, []);
    groups.get(item.key).push(item);
  }
  return head + Array.from(groups, ([key, group]) => renderReferenceCard(key, group)).join("");
}

function renderReferenceCard(key, group) {
  const ref = S.refs.get(key) || { key, label: key };
  const link = refLink(ref);
  const title = ref.title || ref.raw || key;
  const statusBadge = ref.fulltext_status ? `<span class="badge ${ref.fulltext_status}">${esc(TEXT_STATUS[ref.fulltext_status] || ref.fulltext_status)}</span>` : "";
  const claims = group.filter((x) => x.claim);
  const count = claims.length > 1 ? `<span>${claims.length} claims checked</span>` : "";
  return `<div class="card">
    <div class="ref-title">[${esc(ref.label || key)}] ${link ? `<a href="${esc(link)}" target="_blank" rel="noopener">${esc(title)}</a>` : esc(title)}</div>
    <div class="ref-meta">${ref.year ? `<span>${esc(ref.year)}</span>` : ""}${statusBadge}
      ${ref.retracted ? `<span class="badge" style="color:var(--bad);background:var(--bad-bg)">Retracted</span>` : ""}${count}</div>
    ${claims.map((item, i) => renderClaimSection(item, ref, i, claims.length)).join("")}
  </div>`;
}

function renderPill(label, confidence) {
  const info = LABELS[label];
  return `<span class="verdict-pill v-${label}"><span aria-hidden="true">${info.icon}</span> ${esc(info.text)}</span>
    <span class="confidence">${pct(confidence)} confidence</span>`;
}

function renderClaimSection({ claim, key, verdict }, ref, index, total) {
  const heading = total > 1 ? `<div class="claim-index">Claim ${index + 1} of ${total}</div>` : "";
  const claimText = verdict?.attributed_claim || claim.claim_text;
  const claimHtml = `
    <div class="field"><div class="field-name">Claim checked</div>
      <div class="claim-text">${esc(claimText)}</div>
      ${claim.claim_type === "background" ? `<div class="muted" style="font-size:12.5px;margin-top:3px">A background citation (a pointer or list of examples), so it is checked leniently.</div>` : ""}
    </div>`;

  if (!verdict) {
    const escalating = S.escalating.has(pid(claim.id, key));
    return `<section class="claim-section">${heading}${claimHtml}
      <div class="verdict-line"><span class="verdict-pill v-pending"><span class="spinner"></span>
        ${escalating ? "Getting a second opinion from Nemotron Ultra…" : "Checking against the cited paper…"}</span></div></section>`;
  }

  // One verdict per claim. When Ultra re-checked it, show how the verdict was reached.
  const verdictHtml = verdict.escalated && verdict.original_label
    ? `<ol class="trail" aria-label="How this verdict was reached">
        <li><div class="trail-step">First check · Nemotron Super, from retrieved passages</div>
          <div class="verdict-line">${renderPill(verdict.original_label, verdict.original_confidence)}</div></li>
        <li><div class="trail-step">Second opinion · Nemotron Ultra, read the full paper${verdict.original_label === verdict.label ? " (agreed)" : ""}</div>
          <div class="verdict-line">${renderPill(verdict.label, verdict.confidence)}</div></li>
      </ol>`
    : `<div class="verdict-line">${renderPill(verdict.label, verdict.confidence)}</div>`;
  const needsCheck = verdict.label !== "supported"
    ? `<div class="human-check ${["contradicted", "not_found"].includes(verdict.label) ? "bad" : ""}">Needs human check</div>` : "";
  const modes = verdict.failure_modes || [];
  const chips = modes.length ? `
    <div class="chips">${modes.map((mode) => `<span class="mode-chip ${FAILURE[mode]?.[1] ? "serious" : ""}" title="${esc(verdict.failure_notes?.[mode] || "")}">${esc(FAILURE[mode]?.[0] || mode)}</span>`).join("")}</div>
    <ul class="mode-notes">${modes.map((mode) => verdict.failure_notes?.[mode] ? `<li>${esc(verdict.failure_notes[mode])}</li>` : "").join("")}</ul>` : "";

  let evidence = "";
  if (verdict.quote_verified && verdict.quote_context) {
    const ctx = verdict.quote_context;
    const a = verdict.quote_start ?? 0;
    const b = verdict.quote_end ?? 0;
    evidence = `<blockquote class="quote">${a > 0 ? "…" : ""}${esc(ctx.slice(0, a))}<mark>${esc(ctx.slice(a, b))}</mark>${esc(ctx.slice(b))}…</blockquote>`;
  } else if (verdict.quote) {
    evidence = `<blockquote class="quote unverified">${esc(verdict.quote)}</blockquote>
      <div class="quote-note">This quote could not be found verbatim in the cited text, so the verdict's confidence was lowered.</div>`;
  } else if (ref.fulltext_status && ref.fulltext_status !== "full") {
    evidence = `<p class="muted">${esc(ref.status_reason || (ref.fulltext_status === "unavailable" ? "The full text of this paper is not openly available." : "Only the abstract of this paper is openly available."))}</p>`;
  } else {
    evidence = `<p class="muted">No passage in the cited work supports this claim.</p>`;
  }

  return `<section class="claim-section">
    ${heading}
    ${claimHtml}
    ${verdictHtml}
    ${needsCheck}
    ${chips}
    <div class="field"><div class="field-name">From the cited paper</div>${evidence}</div>
    <div class="field"><div class="field-name">Why</div><div class="explanation">${esc(verdict.explanation)}</div></div>
  </section>`;
}

// ------------------------------------------------------------------- demo

function renderDemoNotice() {
  const d = S.demo;
  const kinds = "numbers changed, scope inflated, hedges removed, citations swapped";
  const evalResult = d.eval;
  if (!evalResult) {
    return `<div class="notice demo"><strong>Demo paper: ${d.planted} planted errors.</strong>
      This is a real paper with ${d.planted} citations deliberately altered (${kinds}).
      The checker is not told which ones.</div>`;
  }
  const rows = Object.entries(evalResult.by_mutation)
    .map(([m, r]) => `<li>${esc(MUTATION_TEXT[m] || m)}: caught ${r.caught} of ${r.planted}</li>`).join("");
  return `<div class="notice demo"><strong>Demo paper: ${esc(evalResult.summary_line)}.</strong>
    <details><summary>Details</summary>
      <ul class="demo-breakdown">${rows}</ul>
      <p>Planted citations are marked “Planted” in the list on the right. The other flags are on citations
      that were not altered. A human review of them found ${evalResult.review.genuine} real citation problems in the
      original paper, ${evalResult.review.false_alarm + evalResult.review.unreviewed} false alarms and
      ${evalResult.review.debatable} debatable cases.</p>
    </details></div>`;
}
