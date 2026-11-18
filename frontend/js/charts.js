/**
 * Inline SVG metric bar visualizations (no external chart library).
 */

/**
 * Render a dual horizontal bar comparing Classifier vs HMM for a metric.
 *
 * @param {string} label - Metric display label
 * @param {number} clfVal - Classifier value (0 to 1 for accuracies, or float)
 * @param {number|null} hmmVal - HMM value (or null if HMM unavailable)
 * @param {boolean} isPercentage - Whether value is a 0..1 ratio to format as %
 * @param {boolean} lowerIsBetter - True for mismatch fraction (where lower = better)
 */
export function renderMetricComparisonBar(label, clfVal, hmmVal, isPercentage = true, lowerIsBetter = false) {
  const cVal = clfVal != null ? clfVal : 0;
  const hVal = hmmVal != null ? hmmVal : null;

  const cDisplay = isPercentage ? `${(cVal * 100).toFixed(1)}%` : cVal.toFixed(3);
  const hDisplay = hVal != null ? (isPercentage ? `${(hVal * 100).toFixed(1)}%` : hVal.toFixed(3)) : "N/A";

  // Calculate percentage width (0 to 100%)
  const maxScale = isPercentage ? 1.0 : Math.max(1.0, cVal, hVal || 0);
  const cWidth = Math.min(100, Math.max(2, (cVal / maxScale) * 100));
  const hWidth = hVal != null ? Math.min(100, Math.max(2, (hVal / maxScale) * 100)) : 0;

  let deltaHtml = "";
  if (hVal != null) {
    const delta = hVal - cVal;
    const isImproved = lowerIsBetter ? delta < 0 : delta > 0;
    const isDeclined = lowerIsBetter ? delta > 0 : delta < 0;
    const sign = delta > 0 ? "+" : "";
    const deltaText = isPercentage ? `${sign}${(delta * 100).toFixed(1)}%` : `${sign}${delta.toFixed(3)}`;
    const pillClass = Math.abs(delta) < 0.0001 ? "delta-pill neutral" : isImproved ? "delta-pill positive" : "delta-pill negative";
    deltaHtml = `<span class="${pillClass}">${deltaText}</span>`;
  }

  return `
    <div class="metric-comparison-row">
      <div class="metric-header">
        <span class="metric-name">${label}</span>
        ${deltaHtml}
      </div>
      <div class="bars-container">
        <!-- Classifier Bar -->
        <div class="bar-row">
          <span class="bar-label">CLF</span>
          <div class="svg-bar-wrap">
            <svg width="100%" height="10" viewBox="0 0 100 10" preserveAspectRatio="none">
              <rect x="0" y="0" width="100" height="10" rx="3" fill="#1e293b" />
              <rect x="0" y="0" width="${cWidth}" height="10" rx="3" fill="#F59E0B" />
            </svg>
          </div>
          <span class="bar-val">${cDisplay}</span>
        </div>
        <!-- HMM Bar -->
        ${hVal != null ? `
        <div class="bar-row">
          <span class="bar-label">HMM</span>
          <div class="svg-bar-wrap">
            <svg width="100%" height="10" viewBox="0 0 100 10" preserveAspectRatio="none">
              <rect x="0" y="0" width="100" height="10" rx="3" fill="#1e293b" />
              <rect x="0" y="0" width="${hWidth}" height="10" rx="3" fill="#2DD4BF" />
            </svg>
          </div>
          <span class="bar-val">${hDisplay}</span>
        </div>
        ` : `
        <div class="bar-row inactive">
          <span class="bar-label">HMM</span>
          <div class="svg-bar-wrap empty">
            <span class="unavailable-text">Decoder not active</span>
          </div>
          <span class="bar-val text-muted">-</span>
        </div>
        `}
      </div>
    </div>
  `;
}
