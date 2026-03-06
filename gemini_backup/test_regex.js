const tickerText = "�� RELIANCE — Intraday + Projection";
const match1 = tickerText.match(/📈\s+([A-Z0-9_\-]+)\s+—/);
console.log("Match1:", match1 ? match1[1] : "Fail");

const tickerText2 = "📈 BANKNIFTY24MAR46500CE — Intraday + Projection";
const match2 = tickerText2.match(/📈\s+([A-Z0-9_\-]+)\s+—/);
console.log("Match2:", match2 ? match2[1] : "Fail");
