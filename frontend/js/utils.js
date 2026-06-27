/**
 * Utility functions for CiteQuest-RAG frontend.
 */

/**
 * Escape HTML special characters to prevent XSS.
 */
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

/**
 * Format latency in a human-readable way.
 *   < 1000ms → "342ms"
 *   >= 1000ms → "1.23s"
 */
function formatLatency(ms) {
  if (ms == null) return "?";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

/**
 * Highlight query terms in a title string with <mark> tags.
 * Safe: escapes the title first, then wraps matching words.
 */
function highlightTitle(title, query) {
  if (!title || !query) return escapeHtml(title || "");
  let html = escapeHtml(title);
  const words = query.split(/[\s]+/).filter(w => w.length > 1);
  for (const word of words) {
    const escaped = escapeHtml(word).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    html = html.replace(
      new RegExp(`(${escaped})`, "gi"),
      "<mark>$1</mark>"
    );
  }
  return html;
}
