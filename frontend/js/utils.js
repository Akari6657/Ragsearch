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
