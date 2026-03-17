/**
 * risk-badge.js
 * Renders a coloured risk badge and probability ring.
 */

function riskBadgeHTML(risk) {
  const icons = { HIGH: "▲", MEDIUM: "●", LOW: "▼" };
  return `<span class="badge badge-${risk.toLowerCase()}">${icons[risk] || "●"} ${risk}</span>`;
}

function probRingHTML(probability, risk) {
  const pct    = Math.round(probability * 100);
  const colors = { HIGH: "#f87171", MEDIUM: "#fbbf24", LOW: "#34d399" };
  const color  = colors[risk] || "#6c63ff";
  const r      = 42;
  const circ   = 2 * Math.PI * r;
  const dash   = (pct / 100) * circ;

  return `
    <div class="prob-ring">
      <svg width="100" height="100" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r="${r}" fill="none" stroke="#22263a" stroke-width="8"/>
        <circle cx="50" cy="50" r="${r}" fill="none"
          stroke="${color}" stroke-width="8"
          stroke-dasharray="${dash} ${circ}"
          stroke-linecap="round"
          style="transition: stroke-dasharray 0.8s ease"/>
      </svg>
      <div class="prob-ring-label">
        ${pct}%<span>churn</span>
      </div>
    </div>`;
}
