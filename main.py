from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import yfinance as yf
import pandas as pd

app = FastAPI(title="InvestMAC", version="9.0.0")

PEER_MAP = {
    "AAPL": "MSFT", "MSFT": "AAPL",
    "NVDA": "AMD", "AMD": "NVDA",
    "TSLA": "RIVN", "RIVN": "TSLA",
    "GGAL": "BMA", "BMA": "GGAL",
    "META": "GOOGL", "GOOGL": "META",
    "AMZN": "WMT", "WMT": "AMZN",
    "JPM": "BAC", "BAC": "JPM",
    "NFLX": "DIS", "DIS": "NFLX",
    "XOM": "CVX", "CVX": "XOM",
}

def calc_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean().iloc[-1]
    avg_loss = loss.rolling(period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)

def fetch_metrics(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    hist = t.history(period="1y")
    info = t.info or {}

    if hist.empty or len(hist) < 20:
        return None

    close = hist["Close"]
    price = round(float(close.iloc[-1]), 2)
    ret_1y = round((close.iloc[-1] / close.iloc[0] - 1) * 100, 2)
    daily_ret = close.pct_change().dropna()
    vol = round(float(daily_ret.std() * (252 ** 0.5) * 100), 2)
    rsi = calc_rsi(close)

    roll_max = close.cummax()
    drawdown = (close - roll_max) / roll_max
    max_dd = round(float(drawdown.min() * 100), 2)

    # Fundamentales desde info
    pe = info.get("trailingPE") or info.get("forwardPE")
    pe = round(pe, 2) if pe else None
    roe = info.get("returnOnEquity")
    roe = round(roe * 100, 2) if roe else None
    debt_eq = info.get("debtToEquity")
    debt_eq = round(debt_eq, 2) if debt_eq else None
    rev_growth = info.get("revenueGrowth")
    rev_growth = round(rev_growth * 100, 2) if rev_growth else None
    profit_margin = info.get("profitMargins")
    profit_margin = round(profit_margin * 100, 2) if profit_margin else None
    market_cap = info.get("marketCap")
    name = info.get("shortName") or info.get("longName") or ticker
    sector = info.get("sector") or "N/D"
    industry = info.get("industry") or "N/D"
    country = info.get("country") or "N/D"
    currency = info.get("currency") or "USD"

    # Serie semanal base100 (últimas 52 semanas)
    weekly = hist["Close"].resample("W").last().dropna()
    base = float(weekly.iloc[0])
    serie = [round(float(v) / base * 100, 2) for v in weekly]
    labels = [str(d.date()) for d in weekly.index]

    return {
        "name": name,
        "sector": sector,
        "industry": industry,
        "country": country,
        "currency": currency,
        "price": price,
        "ret_1y": ret_1y,
        "vol": vol,
        "rsi": rsi,
        "max_dd": max_dd,
        "pe": pe,
        "roe": roe,
        "debt_eq": debt_eq,
        "rev_growth": rev_growth,
        "profit_margin": profit_margin,
        "market_cap": market_cap,
        "serie": serie,
        "labels": labels,
    }

def score_metrics(m: dict, horizon: str) -> dict:
    ret = m["ret_1y"]
    vol = m["vol"]
    rsi = m["rsi"]
    pe = m.get("pe")
    roe = m.get("roe")
    debt = m.get("debt_eq")

    # Fundamental
    f = 5
    if ret > 20: f += 1.5
    elif ret > 8: f += 1
    elif ret < -10: f -= 1
    if roe and roe > 15: f += 1
    elif roe and roe < 5: f -= 0.5
    if pe and pe < 20: f += 0.5
    elif pe and pe > 40: f -= 0.5
    if debt and debt < 50: f += 0.5
    elif debt and debt > 150: f -= 0.5
    fundamental = round(min(max(f, 1), 10), 2)

    # Técnico
    t = 5
    if 45 <= rsi <= 60: t += 1.5
    elif 30 <= rsi < 45: t += 0.5
    elif rsi > 70: t -= 1.5
    elif rsi < 30: t -= 0.5
    if vol < 25: t += 0.5
    elif vol > 50: t -= 1
    tecnico = round(min(max(t, 1), 10), 2)

    # Sentimiento proxy
    s = 5
    if vol < 30: s += 1
    elif vol > 50: s -= 1
    if ret > 15: s += 0.5
    elif ret < -15: s -= 1
    sentimiento = round(min(max(s, 1), 10), 2)

    if horizon == "corto":
        total = tecnico * 0.55 + fundamental * 0.25 + sentimiento * 0.20
    elif horizon == "largo":
        total = tecnico * 0.15 + fundamental * 0.65 + sentimiento * 0.20
    else:
        total = tecnico * 0.35 + fundamental * 0.40 + sentimiento * 0.25

    total = round(total, 2)

    if total >= 6.8:
        veredicto = "COMPRAR"
    elif total <= 4.2:
        veredicto = "VENDER"
    else:
        veredicto = "MANTENER"

    return {
        "fundamental": fundamental,
        "tecnico": tecnico,
        "sentimiento": sentimiento,
        "total": total,
        "veredicto": veredicto,
    }

def detail_items(m: dict) -> list:
    items = []

    # RSI
    rsi = m["rsi"]
    if rsi > 70:
        items.append({"item": "RSI14", "value": rsi, "status": "bad",
                      "why": "Zona de sobrecompra técnica.", "action": "Esperar pullback antes de entrar."})
    elif rsi < 30:
        items.append({"item": "RSI14", "value": rsi, "status": "good",
                      "why": "Zona de sobreventa técnica.", "action": "Buscar señal de rebote confirmada."})
    else:
        items.append({"item": "RSI14", "value": rsi, "status": "neutral",
                      "why": "Zona media, sin señal extrema.", "action": "Monitorear junto a volumen."})

    # Retorno 1Y
    ret = m["ret_1y"]
    if ret > 15:
        items.append({"item": "Retorno 1Y %", "value": ret, "status": "good",
                      "why": "Fuerte desempeño anual.", "action": "Mantener sesgo alcista con stop activo."})
    elif ret < -10:
        items.append({"item": "Retorno 1Y %", "value": ret, "status": "bad",
                      "why": "Debilidad anual marcada.", "action": "Reducir exposición o esperar confirmación."})
    else:
        items.append({"item": "Retorno 1Y %", "value": ret, "status": "neutral",
                      "why": "Desempeño moderado.", "action": "Operar con prudencia."})

    # Volatilidad
    vol = m["vol"]
    if vol > 50:
        items.append({"item": "Volatilidad anual %", "value": vol, "status": "bad",
                      "why": "Alta dispersión de precio.", "action": "Reducir tamaño de posición."})
    elif vol < 25:
        items.append({"item": "Volatilidad anual %", "value": vol, "status": "good",
                      "why": "Riesgo de oscilación bajo.", "action": "Apto para perfil conservador."})
    else:
        items.append({"item": "Volatilidad anual %", "value": vol, "status": "neutral",
                      "why": "Riesgo medio.", "action": "Combinar con stop técnico."})

    # P/E
    pe = m.get("pe")
    if pe:
        if pe < 15:
            items.append({"item": "P/E Ratio", "value": pe, "status": "good",
                          "why": "Valuación baja relativa al mercado.", "action": "Favorable para largo plazo."})
        elif pe > 40:
            items.append({"item": "P/E Ratio", "value": pe, "status": "bad",
                          "why": "Valuación elevada, requiere alto crecimiento.", "action": "Exigir confirmación de earnings."})
        else:
            items.append({"item": "P/E Ratio", "value": pe, "status": "neutral",
                          "why": "Valuación en rango normal.", "action": "Comparar con sector."})

    # ROE
    roe = m.get("roe")
    if roe:
        if roe > 15:
            items.append({"item": "ROE %", "value": roe, "status": "good",
                          "why": "Alta rentabilidad sobre patrimonio.", "action": "Señal de ventaja competitiva."})
        elif roe < 5:
            items.append({"item": "ROE %", "value": roe, "status": "bad",
                          "why": "Baja eficiencia en uso del capital.", "action": "Revisar modelo de negocio."})
        else:
            items.append({"item": "ROE %", "value": roe, "status": "neutral",
                          "why": "Rentabilidad moderada.", "action": "Monitorear tendencia."})

    # Deuda
    debt = m.get("debt_eq")
    if debt is not None:
        if debt > 150:
            items.append({"item": "Deuda/Patrimonio", "value": debt, "status": "bad",
                          "why": "Nivel de apalancamiento alto.", "action": "Verificar capacidad de repago."})
        elif debt < 50:
            items.append({"item": "Deuda/Patrimonio", "value": debt, "status": "good",
                          "why": "Estructura financiera sólida.", "action": "Menor riesgo en escenarios adversos."})
        else:
            items.append({"item": "Deuda/Patrimonio", "value": debt, "status": "neutral",
                          "why": "Endeudamiento moderado.", "action": "Monitorear evolución."})

    return items

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/analyze/{ticker}")
def analyze(ticker: str, horizon: str = "mediano"):
    try:
        t = ticker.upper().strip()
        h = horizon.lower().strip()
        if h not in {"corto", "mediano", "largo"}:
            h = "mediano"

        peer = PEER_MAP.get(t, "SPY")

        m = fetch_metrics(t)
        if not m:
            return {"ok": False, "error": f"No se encontraron datos para '{t}'. Verificá el ticker."}

        m_peer = fetch_metrics(peer)
        if not m_peer:
            m_peer = None

        scores = score_metrics(m, h)
        details = detail_items(m)

        price = m["price"]
        stop_loss = round(price * 0.93, 2)
        take_profit_map = {"corto": 1.08, "mediano": 1.15, "largo": 1.25}
        take_profit = round(price * take_profit_map[h], 2)

        # Merge chart labels (usar el más largo)
        labels = m["labels"]
        ser1 = m["serie"]
        ser2 = m_peer["serie"] if m_peer else []

        # Alinear longitudes
        min_len = min(len(labels), len(ser1), len(ser2)) if ser2 else len(labels)
        labels = labels[:min_len]
        ser1 = ser1[:min_len]
        ser2 = ser2[:min_len] if ser2 else []

        comparativa = {
            t: {
                "retorno_1y_pct": m["ret_1y"],
                "volatilidad_pct": m["vol"],
                "max_drawdown_pct": m["max_dd"],
                "rsi14": m["rsi"],
                "pe": m.get("pe"),
                "roe": m.get("roe"),
            }
        }
        if m_peer:
            comparativa[peer] = {
                "retorno_1y_pct": m_peer["ret_1y"],
                "volatilidad_pct": m_peer["vol"],
                "max_drawdown_pct": m_peer["max_dd"],
                "rsi14": m_peer["rsi"],
                "pe": m_peer.get("pe"),
                "roe": m_peer.get("roe"),
            }

        return {
            "ok": True,
            "ticker": t,
            "peer": peer,
            "horizon": h,
            "company": {
                "name": m["name"],
                "sector": m["sector"],
                "industry": m["industry"],
                "country": m["country"],
                "currency": m["currency"],
                "price": price,
                "market_cap": m.get("market_cap"),
            },
            "scores": scores,
            "risk": {
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            },
            "details": details,
            "comparativa": comparativa,
            "chart": {
                "labels": labels,
                "ticker_base100": ser1,
                "peer_base100": ser2,
            },
        }

    except Exception as e:
        return {"ok": False, "error": "Error en el análisis.", "detail": str(e)}

@app.get("/", response_class=HTMLResponse)
def home():
    return """<!doctype html>
<html lang="es">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>InvestMAC</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
:root {
  --bg: #0a0c10;
  --surface: #111318;
  --surface2: #1a1d24;
  --border: #252830;
  --accent: #00e5a0;
  --accent2: #0099ff;
  --danger: #ff4560;
  --warn: #ffa500;
  --text: #e8eaf0;
  --muted: #6b7280;
  --good: #00e5a0;
  --bad: #ff4560;
  --neutral: #ffa500;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'DM Sans', sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  padding: 24px 16px;
}
.wrap { max-width: 1100px; margin: 0 auto; }

/* Header */
.header {
  display: flex;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 32px;
  padding-bottom: 20px;
  border-bottom: 1px solid var(--border);
}
.header h1 {
  font-family: 'Space Mono', monospace;
  font-size: 28px;
  color: var(--accent);
  letter-spacing: -1px;
}
.header span {
  font-size: 13px;
  color: var(--muted);
  font-family: 'Space Mono', monospace;
}

/* Input card */
.input-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 24px;
  margin-bottom: 20px;
  display: flex;
  gap: 16px;
  align-items: flex-end;
  flex-wrap: wrap;
}
.field { display: flex; flex-direction: column; gap: 6px; flex: 1; min-width: 160px; }
.field label { font-size: 11px; font-family: 'Space Mono', monospace; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }
input, select {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 10px;
  color: var(--text);
  padding: 12px 14px;
  font-size: 15px;
  font-family: 'DM Sans', sans-serif;
  outline: none;
  transition: border-color .2s;
}
input:focus, select:focus { border-color: var(--accent); }
select option { background: var(--surface2); }
button {
  background: var(--accent);
  color: #000;
  border: none;
  border-radius: 10px;
  padding: 12px 28px;
  font-size: 15px;
  font-weight: 600;
  font-family: 'DM Sans', sans-serif;
  cursor: pointer;
  transition: opacity .2s, transform .1s;
  white-space: nowrap;
}
button:hover { opacity: .85; }
button:active { transform: scale(.97); }
button:disabled { opacity: .4; cursor: not-allowed; }
.err { color: var(--danger); font-size: 13px; margin-top: 8px; font-family: 'Space Mono', monospace; }

/* Cards */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 24px;
  margin-bottom: 16px;
}
.card h3 {
  font-family: 'Space Mono', monospace;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 2px;
  color: var(--muted);
  margin-bottom: 20px;
}

/* Veredicto banner */
.verdict-banner {
  border-radius: 16px;
  padding: 24px 28px;
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 16px;
}
.verdict-comprar { background: linear-gradient(135deg, #001a0f, #003320); border: 1px solid var(--good); }
.verdict-vender  { background: linear-gradient(135deg, #1a0000, #330000); border: 1px solid var(--bad); }
.verdict-mantener{ background: linear-gradient(135deg, #1a1000, #332200); border: 1px solid var(--warn); }
.verdict-label { font-family: 'Space Mono', monospace; font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 2px; }
.verdict-text { font-family: 'Space Mono', monospace; font-size: 36px; font-weight: 700; }
.verdict-comprar .verdict-text { color: var(--good); }
.verdict-vender  .verdict-text { color: var(--bad); }
.verdict-mantener .verdict-text { color: var(--warn); }
.verdict-meta { text-align: right; }
.verdict-ticker { font-family: 'Space Mono', monospace; font-size: 20px; color: var(--text); }
.verdict-score { font-family: 'Space Mono', monospace; font-size: 13px; color: var(--muted); margin-top: 4px; }

/* KPI grid */
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 12px; margin-bottom: 16px; }
.kpi {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px;
}
.kpi-label { font-size: 10px; font-family: 'Space Mono', monospace; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
.kpi-value { font-size: 22px; font-weight: 600; font-family: 'Space Mono', monospace; }
.kpi-sub { font-size: 11px; color: var(--muted); margin-top: 4px; }

/* Scores */
.scores-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.score-item { text-align: center; }
.score-name { font-size: 11px; font-family: 'Space Mono', monospace; color: var(--muted); text-transform: uppercase; margin-bottom: 8px; }
.score-bar-wrap { background: var(--surface2); border-radius: 6px; height: 8px; overflow: hidden; margin-bottom: 6px; }
.score-bar { height: 100%; border-radius: 6px; transition: width .8s ease; }
.score-val { font-family: 'Space Mono', monospace; font-size: 18px; font-weight: 700; }

/* Table */
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th { font-family: 'Space Mono', monospace; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); padding: 10px 12px; border-bottom: 1px solid var(--border); text-align: left; }
td { padding: 12px 12px; border-bottom: 1px solid #1a1d24; vertical-align: top; }
tr:last-child td { border-bottom: none; }
.tag { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 11px; font-family: 'Space Mono', monospace; font-weight: 700; }
.tag-good    { background: #001a0f; color: var(--good); border: 1px solid var(--good); }
.tag-bad     { background: #1a0000; color: var(--bad); border: 1px solid var(--bad); }
.tag-neutral { background: #1a1000; color: var(--warn); border: 1px solid var(--warn); }

/* Risk */
.risk-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.risk-item { background: var(--surface2); border-radius: 12px; padding: 16px; }
.risk-label { font-size: 11px; font-family: 'Space Mono', monospace; color: var(--muted); margin-bottom: 6px; }
.risk-val { font-family: 'Space Mono', monospace; font-size: 20px; font-weight: 700; }
.risk-sl { color: var(--bad); }
.risk-tp { color: var(--good); }

/* Chart */
.chart-wrap { position: relative; height: 260px; }

/* Loading */
#loading { display: none; color: var(--muted); font-family: 'Space Mono', monospace; font-size: 13px; margin-top: 8px; }
.dot-anim::after { content: ''; animation: dots 1.2s infinite; }
@keyframes dots {
  0%   { content: '.'; }
  33%  { content: '..'; }
  66%  { content: '...'; }
}

.hidden { display: none; }
</style>
</head>
<body>
<div class="wrap">

  <div class="header">
    <h1>InvestMAC</h1>
    <span>// análisis de mercado</span>
  </div>

  <div class="input-card">
    <div class="field">
      <label>Ticker</label>
      <input id="ticker" placeholder="AAPL, GGAL, NVDA…" autocomplete="off" spellcheck="false"/>
    </div>
    <div class="field">
      <label>Horizonte</label>
      <select id="horizon">
        <option value="corto">Corto plazo (1–3 meses)</option>
        <option value="mediano" selected>Mediano plazo (3–12 meses)</option>
        <option value="largo">Largo plazo (1–3 años)</option>
      </select>
    </div>
    <button id="btn" onclick="run()">Analizar</button>
  </div>
  <div id="msg" class="err"></div>
  <div id="loading" class="dot-anim">Obteniendo datos reales</div>

  <!-- Veredicto -->
  <div id="verdictBanner" class="hidden verdict-banner">
    <div>
      <div class="verdict-label">Recomendación</div>
      <div class="verdict-text" id="verdictText">—</div>
    </div>
    <div class="verdict-meta">
      <div class="verdict-ticker" id="verdictTicker">—</div>
      <div class="verdict-score" id="verdictScore">Score: —</div>
    </div>
  </div>

  <!-- KPIs -->
  <div id="kpiSection" class="hidden">
    <div class="kpi-grid" id="kpiGrid"></div>
  </div>

  <!-- Scores -->
  <div id="scoresCard" class="card hidden">
    <h3>Ponderación de análisis</h3>
    <div class="scores-grid" id="scoresGrid"></div>
  </div>

  <!-- Riesgo -->
  <div id="riskCard" class="card hidden">
    <h3>Niveles de riesgo</h3>
    <div class="risk-grid">
      <div class="risk-item"><div class="risk-label">Stop Loss</div><div class="risk-val risk-sl" id="stopLoss">—</div></div>
      <div class="risk-item"><div class="risk-label">Take Profit</div><div class="risk-val risk-tp" id="takeProfit">—</div></div>
    </div>
  </div>

  <!-- Detalle -->
  <div id="detailCard" class="card hidden">
    <h3>Detalle de indicadores</h3>
    <table><thead><tr><th>Indicador</th><th>Valor</th><th>Estado</th><th>Diagnóstico</th><th>Acción</th></tr></thead>
    <tbody id="detailBody"></tbody></table>
  </div>

  <!-- Comparativa -->
  <div id="cmpCard" class="card hidden">
    <h3>Comparativa vs competidor</h3>
    <table><thead><tr><th>Métrica</th><th id="cmpH1">—</th><th id="cmpH2">—</th></tr></thead>
    <tbody id="cmpBody"></tbody></table>
  </div>

  <!-- Gráfico -->
  <div id="chartCard" class="card hidden">
    <h3>Performance relativa (base 100)</h3>
    <div class="chart-wrap"><canvas id="chart"></canvas></div>
  </div>

</div>

<script>
let chartRef = null;

function tagHtml(status) {
  const map = { good: 'tag-good', bad: 'tag-bad', neutral: 'tag-neutral' };
  const labels = { good: '▲ POSITIVO', bad: '▼ NEGATIVO', neutral: '● NEUTRO' };
  return `<span class="tag ${map[status]||'tag-neutral'}">${labels[status]||status}</span>`;
}

function scoreColor(v) {
  if (v >= 6.8) return '#00e5a0';
  if (v <= 4.2) return '#ff4560';
  return '#ffa500';
}

function fmt(v, suffix='') {
  if (v === null || v === undefined) return 'N/D';
  return v + suffix;
}

function fmtCap(v) {
  if (!v) return 'N/D';
  if (v >= 1e12) return (v/1e12).toFixed(2) + 'T';
  if (v >= 1e9) return (v/1e9).toFixed(1) + 'B';
  if (v >= 1e6) return (v/1e6).toFixed(1) + 'M';
  return v;
}

async function run() {
  const ticker = document.getElementById('ticker').value.trim();
  const horizon = document.getElementById('horizon').value;
  const msg = document.getElementById('msg');
  const loading = document.getElementById('loading');
  const btn = document.getElementById('btn');

  msg.textContent = '';
  if (!ticker) { msg.textContent = 'Ingresá un ticker.'; return; }

  btn.disabled = true;
  loading.style.display = 'block';

  // hide all sections
  ['verdictBanner','kpiSection','scoresCard','riskCard','detailCard','cmpCard','chartCard']
    .forEach(id => document.getElementById(id).classList.add('hidden'));

  try {
    const res = await fetch(`/analyze/${encodeURIComponent(ticker)}?horizon=${encodeURIComponent(horizon)}`);
    const d = await res.json();

    loading.style.display = 'none';
    btn.disabled = false;

    if (!d.ok) {
      msg.textContent = d.error + (d.detail ? ' — ' + d.detail : '');
      return;
    }

    // Veredicto
    const vb = document.getElementById('verdictBanner');
    vb.className = 'verdict-banner verdict-' + d.scores.veredicto.toLowerCase();
    vb.classList.remove('hidden');
    document.getElementById('verdictText').textContent = d.scores.veredicto;
    document.getElementById('verdictTicker').textContent = d.ticker + ' · ' + (d.company.name || d.ticker);
    document.getElementById('verdictScore').textContent = 'Score total: ' + d.scores.total + ' / 10';

    // KPIs
    const c = d.company;
    const kpis = [
      { label: 'Precio actual', value: c.currency + ' ' + c.price },
      { label: 'Retorno 1Y', value: fmt(d.comparativa[d.ticker]?.retorno_1y_pct, '%'), color: d.comparativa[d.ticker]?.retorno_1y_pct >= 0 ? '#00e5a0' : '#ff4560' },
      { label: 'Volatilidad', value: fmt(d.comparativa[d.ticker]?.volatilidad_pct, '%') },
      { label: 'RSI 14', value: fmt(d.comparativa[d.ticker]?.rsi14) },
      { label: 'Max Drawdown', value: fmt(d.comparativa[d.ticker]?.max_drawdown_pct, '%'), color: '#ff4560' },
      { label: 'Market Cap', value: fmtCap(c.market_cap) },
      { label: 'P/E Ratio', value: fmt(d.comparativa[d.ticker]?.pe) },
      { label: 'ROE', value: fmt(d.comparativa[d.ticker]?.roe, '%') },
    ];
    document.getElementById('kpiGrid').innerHTML = kpis.map(k =>
      `<div class="kpi">
        <div class="kpi-label">${k.label}</div>
        <div class="kpi-value" style="${k.color ? 'color:'+k.color : ''}">${k.value}</div>
        ${k.sub ? `<div class="kpi-sub">${k.sub}</div>` : ''}
      </div>`
    ).join('');
    document.getElementById('kpiSection').classList.remove('hidden');

    // Scores
    const sc = d.scores;
    const scoreItems = [
      { name: 'Fundamental', val: sc.fundamental },
      { name: 'Técnico', val: sc.tecnico },
      { name: 'Sentimiento', val: sc.sentimiento },
    ];
    document.getElementById('scoresGrid').innerHTML = scoreItems.map(s =>
      `<div class="score-item">
        <div class="score-name">${s.name}</div>
        <div class="score-bar-wrap">
          <div class="score-bar" style="width:${s.val*10}%;background:${scoreColor(s.val)}"></div>
        </div>
        <div class="score-val" style="color:${scoreColor(s.val)}">${s.val}</div>
      </div>`
    ).join('');
    document.getElementById('scoresCard').classList.remove('hidden');

    // Riesgo
    document.getElementById('stopLoss').textContent = c.currency + ' ' + d.risk.stop_loss;
    document.getElementById('takeProfit').textContent = c.currency + ' ' + d.risk.take_profit;
    document.getElementById('riskCard').classList.remove('hidden');

    // Detalle
    document.getElementById('detailBody').innerHTML = (d.details||[]).map(x =>
      `<tr><td><strong>${x.item}</strong></td><td style="font-family:'Space Mono',monospace">${x.value}</td><td>${tagHtml(x.status)}</td><td style="color:#9ca3af;font-size:13px">${x.why}</td><td style="font-size:13px">${x.action}</td></tr>`
    ).join('');
    document.getElementById('detailCard').classList.remove('hidden');

    // Comparativa
    const tkr = d.ticker, pr = d.peer;
    const ct = d.comparativa[tkr] || {};
    const cp = d.comparativa[pr] || {};
    document.getElementById('cmpH1').textContent = tkr;
    document.getElementById('cmpH2').textContent = pr;
    const rows = [
      ['Retorno 1Y %', ct.retorno_1y_pct, cp.retorno_1y_pct],
      ['Volatilidad %', ct.volatilidad_pct, cp.volatilidad_pct],
      ['Max Drawdown %', ct.max_drawdown_pct, cp.max_drawdown_pct],
      ['RSI 14', ct.rsi14, cp.rsi14],
      ['P/E Ratio', ct.pe ?? 'N/D', cp.pe ?? 'N/D'],
      ['ROE %', ct.roe ?? 'N/D', cp.roe ?? 'N/D'],
    ];
    document.getElementById('cmpBody').innerHTML = rows.map(([label, v1, v2]) =>
      `<tr><td>${label}</td><td style="font-family:'Space Mono',monospace;font-weight:600">${v1}</td><td style="font-family:'Space Mono',monospace;color:var(--muted)">${v2}</td></tr>`
    ).join('');
    document.getElementById('cmpCard').classList.remove('hidden');

    // Gráfico
    const ctx = document.getElementById('chart').getContext('2d');
    if (chartRef) chartRef.destroy();
    const datasets = [{
      label: tkr + ' (base 100)',
      data: d.chart.ticker_base100,
      borderColor: '#00e5a0',
      backgroundColor: 'rgba(0,229,160,0.06)',
      borderWidth: 2,
      tension: 0.3,
      fill: true,
      pointRadius: 0,
    }];
    if (d.chart.peer_base100 && d.chart.peer_base100.length > 0) {
      datasets.push({
        label: pr + ' (base 100)',
        data: d.chart.peer_base100,
        borderColor: '#0099ff',
        backgroundColor: 'rgba(0,153,255,0.04)',
        borderWidth: 2,
        tension: 0.3,
        fill: true,
        pointRadius: 0,
      });
    }
    chartRef = new Chart(ctx, {
      type: 'line',
      data: { labels: d.chart.labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#9ca3af', font: { family: 'Space Mono', size: 11 } } }
        },
        scales: {
          x: { ticks: { color: '#4b5563', maxTicksLimit: 12, font: { size: 10 } }, grid: { color: '#1a1d24' } },
          y: { ticks: { color: '#4b5563', font: { size: 10 } }, grid: { color: '#1a1d24' } }
        }
      }
    });
    document.getElementById('chartCard').classList.remove('hidden');

  } catch(e) {
    loading.style.display = 'none';
    btn.disabled = false;
    msg.textContent = 'Error de conexión: ' + e.message;
  }
}

document.getElementById('ticker').addEventListener('keydown', e => {
  if (e.key === 'Enter') run();
});
</script>
</body>
</html>"""
