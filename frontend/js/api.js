/**
 * API client for CiteQuest-RAG.
 *
 * All functions return the parsed JSON body, or throw on network / HTTP error.
 */

const API_BASE = "";

/**
 * Generic POST helper.
 */
async function apiPost(path, body, signal) {
  const resp = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

  if (!resp.ok) {
    const text = await resp.text().catch(() => "unknown error");
    throw new Error(`HTTP ${resp.status}: ${text}`);
  }

  return resp.json();
}

/**
 * Generic GET helper.
 */
async function apiGet(path, signal) {
  const resp = await fetch(`${API_BASE}${path}`, { signal });

  if (!resp.ok) {
    const text = await resp.text().catch(() => "unknown error");
    throw new Error(`HTTP ${resp.status}: ${text}`);
  }

  return resp.json();
}

/**
 * POST /search — perform academic search with optional AI overview.
 */
async function searchPapers({
  query,
  top_k = 10,
  mode = "hybrid",
  alpha = 0.5,
  include_overview = true,
  year_from = null,
  year_to = null,
  signal = null,
} = {}) {
  return apiPost("/search", {
    query,
    top_k,
    mode,
    alpha,
    include_overview,
    year_from,
    year_to,
  }, signal);
}

/**
 * POST /ask — ask a question with citation-grounded RAG.
 */
async function askQuestion({
  question,
  top_k = 8,
  retrieval_mode = "hybrid",
  alpha = 0.5,
  signal = null,
} = {}) {
  return apiPost("/ask", {
    question,
    top_k,
    retrieval_mode,
    alpha,
  }, signal);
}

/**
 * GET /health — check API and index status.
 */
async function checkHealth(signal) {
  return apiGet("/health", signal);
}
