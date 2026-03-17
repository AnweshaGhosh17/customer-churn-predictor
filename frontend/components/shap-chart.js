/**
 * shap-chart.js
 * Renders a horizontal bar chart from SHAP top_factors data.
 */

function shapChartHTML(topFactors) {
  if (!topFactors || topFactors.length === 0) return "<p>No SHAP data available.</p>";

  const maxImpact = Math.max(...topFactors.map(f => Math.abs(f.impact)));

  const rows = topFactors.map(f => {
    const pct       = maxImpact > 0 ? (Math.abs(f.impact) / maxImpact) * 100 : 0;
    const direction = f.impact > 0 ? "positive" : "negative";
    const sign      = f.impact > 0 ? "+" : "";

    return `
      <div class="shap-row">
        <div class="shap-label" title="${f.label}">${f.label}</div>
        <div class="shap-bar-wrap">
          <div class="shap-bar ${direction}" style="width: ${pct}%"></div>
        </div>
        <div class="shap-val">${sign}${f.impact.toFixed(3)}</div>
      </div>`;
  });

  return `
    <div style="margin-bottom:8px; font-size:0.8rem; color:var(--muted);">
      <span style="color:var(--red)">▮ Increases churn risk</span>
      &nbsp;&nbsp;
      <span style="color:var(--green)">▮ Decreases churn risk</span>
    </div>
    ${rows.join("")}`;
}
