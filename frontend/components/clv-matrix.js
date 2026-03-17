/**
 * clv-matrix.js
 * Renders the CLV × Risk priority matrix.
 */

function clvMatrixHTML(risk, clvTier, priority) {
  const cells = [
    ["HIGH",   "HIGH",   "CRITICAL"],
    ["HIGH",   "MEDIUM", "HIGH"],
    ["HIGH",   "LOW",    "MEDIUM"],
    ["MEDIUM", "HIGH",   "HIGH"],
    ["MEDIUM", "MEDIUM", "MEDIUM"],
    ["MEDIUM", "LOW",    "LOW"],
    ["LOW",    "HIGH",   "MEDIUM"],
    ["LOW",    "MEDIUM", "LOW"],
    ["LOW",    "LOW",    "LOW"],
  ];

  const colorMap = {
    CRITICAL: { bg: "rgba(248,113,113,0.2)", color: "#f87171", border: "rgba(248,113,113,0.4)" },
    HIGH:     { bg: "rgba(251,191,36,0.15)", color: "#fbbf24", border: "rgba(251,191,36,0.35)" },
    MEDIUM:   { bg: "rgba(108,99,255,0.15)", color: "#a78bfa", border: "rgba(108,99,255,0.35)" },
    LOW:      { bg: "rgba(52,211,153,0.1)",  color: "#34d399", border: "rgba(52,211,153,0.3)" },
  };

  const rows = cells.map(([r, c, p]) => {
    const isActive = r === risk && c === clvTier;
    const style    = colorMap[p];
    const activeStyle = isActive
      ? `background:${style.bg}; border:2px solid ${style.border}; font-weight:700;`
      : `opacity:0.35;`;

    return `
      <tr>
        <td>${r}</td>
        <td>${c}</td>
        <td style="${activeStyle} padding:6px 12px; border-radius:6px; color:${isActive ? style.color : 'inherit'}">
          ${isActive ? "▶ " : ""}${p}
        </td>
      </tr>`;
  });

  return `
    <table>
      <thead>
        <tr>
          <th>Churn Risk</th>
          <th>CLV Tier</th>
          <th>Retention Priority</th>
        </tr>
      </thead>
      <tbody>${rows.join("")}</tbody>
    </table>
    <p style="font-size:0.8rem;color:var(--muted);margin-top:10px;">
      Highlighted row = this customer's position
    </p>`;
}
