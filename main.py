from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import yfinance as yf
from statistics import mean, pstdev

app = FastAPI(title="InvestMAC API", version="7.1.0")

PEER_MAP = {
    "AAPL": "MSFT", "MSFT": "AAPL",
    "NVDA": "AMD", "AMD": "NVDA",
    "TSLA": "RIVN",
    "GGAL": "BMA", "BMA": "GGAL",
    "META": "GOOGL", "GOOGL": "META",
    "AMZN": "WMT",
}

def clamp(n, lo=1, hi=10): return max(lo, min(hi, n))

def sma(values, w):
    if len(values) < w: return values[-1] if values else 0.0
    return sum(values[-w:]) / w

def rsi(values, period=14):
    if len(values) < period + 1: return 50.0
    gains, losses = [], []
    for i in range(-period, 0):
        d = values[i] - values[i - 1]
        if d >= 0: gains.append(d)
        else: losses.append(abs(d))
    avg_gain = sum(gains) / period if gains else 0.0
    avg_loss = sum(losses) / period if losses else 0.0
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def total_return(prices):
    if len(prices) < 2: return 0.0
    return (prices[-1] / prices[0]) - 1

def annualized_vol(prices):
    if len(prices) < 2: return 0.0
    rets = [(prices[i] / prices[i - 1]) - 1 for i in range(1, len(prices))]
    if len(rets) < 2: return 0.0
    return pstdev(rets) * (252 ** 0.5)

def max_drawdown(prices):
    if not prices: return 0.0
    peak = prices[0]
    mdd = 0.0
    for p in prices:
        peak = max(peak, p)
        dd = (p / peak) - 1
        mdd = min(mdd, dd)
    return mdd

def zscore_last(prices):
    if len(prices) < 5: return 0.0
    m = mean(prices)
    s = pstdev(prices)
    if s == 0: return 0.0
    return (prices[-1] - m) / s

def fmt_money(n):
    if n is None: return "-"
    try: n = float(n)
    except Exception: return "-"
    if abs(n) >= 1_000_000_000_000: return f"{n/1_000_000_000_000:.2f}T"
    if abs(n) >= 1_000_000_000: return f"{n/1_000_000_000:.2f}B"
    if abs(n) >= 1_000_000: return f"{n/1_000_000:.2f}M"
    return f"{n:,.2f}"

def fetch_data(ticker: str):
    tk = yf.Ticker(ticker)
    hist = tk.history(period="1y", interval="1d")
    if hist is None or hist.empty: return None, None, None
    closes = [float(x) for x in hist["Close"].dropna().tolist()]
    dates = [d.strftime("%Y-%m-%d") for d in hist.index.to_pydatetime()]
    info = tk.info if tk else {}
    return closes, dates, info

def score_asset(closes, horizon):
    last = closes[-1]
    ret = total_return(closes)
    rsi14 = rsi(closes, 14)
    s20 = sma(closes, 20)
    s50 = sma(closes, 50)
    s200 = sma(closes, 200)

    tecnico = 5
    if s50 > s200: tecnico += 2
    if last > s20: tecnico += 1
    if 45 <= rsi14 <= 65: tecnico += 1
    if rsi14 > 75 or rsi14 < 25: tecnico -= 1
    tecnico = clamp(tecnico)

    if ret > 0.35: fundamental = 8
    elif ret > 0.15: fundamental = 7
    elif ret > 0: fundamental = 6
    elif ret > -0.15: fundamental = 5
    else: fundamental = 4

    z = zscore_last(closes)
    if z > 1.5: sentimiento = 4
    elif z < -1.5: sentimiento = 5
    else: sentimiento = 6

    if horizon == "corto":
        total = tecnico * 0.5 + fundamental * 0.3 + sentimiento * 0.2
    elif horizon == "largo":
        total = tecnico * 0.2 + fundamental * 0.6 + sentimiento * 0.2
    else:
        total = tecnico * 0.35 + fundamental * 0.4 + sentimiento * 0.25

    if total >= 7.5: veredicto = "COMPRAR"
    elif total <= 4.5: veredicto = "VENDER"
    else: veredicto = "MANTENER"

    return {
        "fundamental": fundamental,
        "tecnico": tecnico,
        "sentimiento": sentimiento,
        "total": round(total, 2),
        "veredicto": veredicto,
        "retorno_1y_pct": round(ret * 100, 2),
        "rsi14": round(rsi14, 2),
        "sma20": round(s20, 2),
        "sma50": round(s50, 2),
        "sma200": round(s200, 2),
        "last": round(last, 2),
        "stop_loss": round(last * 0.93, 2),
        "take_profit": round(last * 1.12, 2),
        "volatilidad_pct": round(annualized_vol(closes) * 100, 2),
        "max_drawdown_pct": round(max_drawdown(closes) * 100, 2),
    }

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/analyze/{ticker}")
def analyze(ticker: str, horizon: str = "mediano"):
    t = ticker.upper().strip()
    h = horizon.lower().strip()
    if h not in {"corto", "mediano", "largo"}: h = "mediano"

    peer = PEER_MAP.get(t, "SPY")
    c1, d1, i1 = fetch_data(t)
    c2, d2, i2 = fetch_data(peer)

    if not c1: return {"error": f"No se pudieron obtener datos para {t}"}
    if not c2: return {"error": f"No se pudieron obtener datos para competidor {peer}"}

    s1 = score_asset(c1, h)
    s2 = score_asset(c2, h)

    n = min(len(c1), len(c2), len(d1), len(d2))
    labels = d1[-n:]
    b1, b2 = c1[-n], c2[-n]
    rel1 = [round((x / b1) * 100, 2) for x in c1[-n:]]
    rel2 = [round((x / b2) * 100, 2) for x in c2[-n:]]

    return {
        "ticker": t,
        "peer": peer,
        "horizon": h,
        "company": {
            "name": i1.get("longName") or i1.get("shortName") or t,
            "sector": i1.get("sector") or "-",
            "industry": i1.get("industry") or "-",
            "country": i1.get("country") or "-",
            "market_cap": fmt_money(i1.get("marketCap")),
            "pe": i1.get("trailingPE"),
            "dividend_yield_pct": round((i1.get("dividendYield") or 0) * 100, 2) if i1.get("dividendYield") is not None else None
        },
        "scores": s1,
        "comparativa": {
            t: {"retorno_1y_pct": s1["retorno_1y_pct"], "volatilidad_pct": s1["volatilidad_pct"], "max_drawdown_pct": s1["max_drawdown_pct"], "rsi14": s1["rsi14"]},
            peer: {"retorno_1y_pct": s2["retorno_1y_pct"], "volatilidad_pct": s2["volatilidad_pct"], "max_drawdown_pct": s2["max_drawdown_pct"], "rsi14": s2["rsi14"]},
        },
        "chart": {"labels": labels, "ticker_base100": rel1, "peer_base100": rel2}
    }

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html lang="es">
<head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>InvestMAC</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body{font-family:Arial;background:#f2f4f8;margin:0;padding:20px}
.wrap{max-width:1000px;margin:0 auto}
.card{background:#fff;border-radius:12px;padding:16px;margin-bottom:12px;box-shadow:0 4px 12px rgba(0,0,0,.08)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
input,select,button{width:100%;padding:10px;border:1px solid #ccc;border-radius:8px;font-size:16px}
button{background:#111;color:#fff;border:none;cursor:pointer}
.k{font-size:12px;color:#555}.v{font-size:22px;font-weight:700}
table{width:100%;border-collapse:collapse} th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}
</style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h2>InvestMAC</h2>
    <div class="grid2">
      <div><label>Ticker</label><input id="ticker" placeholder="AAPL, GGAL, NVDA"/></div>
      <div><label>Horizonte</label><select id="horizon"><option value="corto">Corto</option><option value="mediano" selected>Mediano</option><option value="largo">Largo</option></select></div>
    </div>
    <div style="margin-top:10px"><button onclick="run()">OK</button></div>
  </div>

  <div class="card" id="main" style="display:none">
    <div class="grid3">
      <div><div class="k">Empresa</div><div class="v" id="name">-</div></div>
      <div><div class="k">Ticker</div><div class="v" id="tick">-</div></div>
      <div><div class="k">Competidor</div><div class="v" id="peer">-</div></div>
      <div><div class="k">Sector</div><div id="sector">-</div></div>
      <div><div class="k">Industria</div><div id="industry">-</div></div>
      <div><div class="k">País</div><div id="country">-</div></div>
    </div>
  </div>

  <div class="card" id="cmp" style="display:none">
    <h3>Comparación</h3>
    <table id="cmpTable"></table>
  </div>

  <div class="card" id="ch" style="display:none">
    <h3>Gráfico</h3><canvas id="chart"></canvas>
  </div>
</div>

<script>
let chartRef=null;
async function run(){
  const t=document.getElementById("ticker").value.trim();
  const h=document.getElementById("horizon").value;
  if(!t){alert("Ingresá un ticker");return;}
  const res=await fetch(`/analyze/${encodeURIComponent(t)}?horizon=${encodeURIComponent(h)}`);
  const d=await res.json();
  if(d.error){alert(d.error);return;}

  document.getElementById("main").style.display="block";
  document.getElementById("cmp").style.display="block";
  document.getElementById("ch").style.display="block";

  document.getElementById("name").textContent=d.company?.name ?? "-";
  document.getElementById("tick").textContent=d.ticker ?? "-";
  document.getElementById("peer").textContent=d.peer ?? "-";
  document.getElementById("sector").textContent=d.company?.sector ?? "-";
  document.getElementById("industry").textContent=d.company?.industry ?? "-";
  document.getElementById("country").textContent=d.company?.country ?? "-";

  const tkr=d.ticker, pr=d.peer, ct=d.comparativa[tkr], cp=d.comparativa[pr];
  document.getElementById("cmpTable").innerHTML=`
    <thead><tr><th>Métrica</th><th>${tkr}</th><th>${pr}</th></tr></thead>
    <tbody>
      <tr><td>Retorno 1Y %</td><td>${ct?.retorno_1y_pct ?? "-"}</td><td>${cp?.retorno_1y_pct ?? "-"}</td></tr>
      <tr><td>Volatilidad %</td><td>${ct?.volatilidad_pct ?? "-"}</td><td>${cp?.volatilidad_pct ?? "-"}</td></tr>
      <tr><td>Max Drawdown %</td><td>${ct?.max_drawdown_pct ?? "-"}</td><td>${cp?.max_drawdown_pct ?? "-"}</td></tr>
      <tr><td>RSI14</td><td>${ct?.rsi14 ?? "-"}</td><td>${cp?.rsi14 ?? "-"}</td></tr>
    </tbody>
  `;

  const ctx=document.getElementById("chart").getContext("2d");
  if(chartRef) chartRef.destroy();
  chartRef=new Chart(ctx,{
    type:"line",
    data:{
      labels:d.chart?.labels ?? [],
      datasets:[
        {label:`${tkr} base100`, data:d.chart?.ticker_base100 ?? [], borderWidth:2, tension:0.2},
        {label:`${pr} base100`, data:d.chart?.peer_base100 ?? [], borderWidth:2, tension:0.2}
      ]
    }
  });
}
</script>
</body>
</html>
"""