const tickerText = "📈 RELIANCE — Intraday + Projection";
const match1 = tickerText.match(/📈\s+([A-Z0-9_\-]+)\s*—/);
console.log("Match1:", match1 ? match1[1] : "Fail");
