from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import random

app = FastAPI(title="InvestMAC", version="8.0.0")

PEER_MAP = {
    "AAPL": "MSFT", "MSFT": "AAPL",
    "NVDA": "AMD", "AMD": "NVDA",
    "TSLA": "RIVN", "GGAL": "BMA", "BMA": "GGAL",
    "META": "GOOGL", "GOOGL": "META", "AMZN": "WMT"
}

def seed_from_ticker(t: str) -> int:
    return sum(ord(c) for c in t.upper())

def gen_metrics(ticker: str):
    r = random.Random(seed_from_ticker(ticker))
    ret = round(r.uniform(-25, 40), 2)
    vol = round(r.uniform(15, 60), 2)
    rsi = round(r.uniform(20, 80), 2)
    dd = round(r.uniform(-45, -8), 2)
    price = round(r.uniform(20, 450), 2)
    return ret, vol, rsi, dd, price

def score_from_metrics(ret, vol, rsi, horizon):
    fundamental = 8 if ret > 20 else 7 if ret > 8 else 6 if ret > 0 else 4
    tecnico = 7 if 45 <= rsi <= 65 else 5 if (30 <= rsi < 45 or 65 < rsi <= 70) else 4
    sentimiento = 6 if vol < 35 else 5 if vol < 45 else 4

    if horizon == "corto":
        total = tecnico * 0.5 + fundamental * 0.3 + sentimiento * 0.2
    elif horizon == "largo":
        total = tecnico * 0.2 + fundamental * 0.6 + sentimiento * 0.2
    else:
        total = tecnico * 0.35 + fundamental * 0.4 + sentimiento * 0.25

    if total >= 7.5:
        veredicto = "COMPRAR"
    elif total <= 4.5:
        veredicto = "VENDER"
    else:
        veredicto = "MANTENER"

    return fundamental, tecnico, sentimiento, round(total, 2), veredicto

def status_item(item, value):
    if item == "Retorno 1Y":
        if value > 15: return "good", "Fuerte desempeño anual.", "Mantener sesgo alcista con gestión de riesgo."
        if value < -10: return "bad", "Debilidad anual marcada.", "Reducir exposición y esperar confirmación."
        return "neutral", "Desempeño moderado.", "Operar con prudencia."
    if item == "Volatilidad":
        if value > 45: return "bad", "Alta variación de precio.", "Usar posición más pequeña."
        if value < 25: return "good", "Riesgo de oscilación menor.", "Apto para perfil conservador."
        return "neutral", "Riesgo medio.", "Combinar con stop técnico."
    if item == "RSI14":
        if value > 70: return "bad", "Sobrecompra técnica.", "Esperar pullback o confirmación."
        if value < 30: return "good", "Sobreventa técnica.", "Buscar rebote confirmado."
        return "neutral", "Zona media técnica.", "Sin señal extrema."
    return "neutral", "Sin regla.", "Monitorear."

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

        ret, vol, rsi, dd, price = gen_metrics(t)
        ret_p, vol_p, rsi_p, dd_p, price_p = gen_metrics(peer)

        fundamental, tecnico, sentimiento, total, veredicto = score_from_metrics(ret, vol, rsi, h)

        s_ret, why_ret, act_ret = status_item("Retorno 1Y", ret)
        s_vol, why_vol, act_vol = status_item("Volatilidad", vol)
        s_rsi, why_rsi, act_rsi = status_item("RSI14", rsi)

        # gráfico sintético base100
        rng = random.Random(seed_from_ticker(t) + 99)
        rng2 = random.Random(seed_from_ticker(peer) + 77)
        base1, base2 = 100.0, 100.0
        ser1, ser2, labels = [], [], []
        for i in range(1, 53):
            base1 *= (1 + rng.uniform(-0.03, 0.035))
            base2 *= (1 + rng2.uniform(-0.03, 0.03))
            ser1.append(round(base1, 2))
            ser2.append(round(base2, 2))
            labels.append(f"S{i}")

        return {
            "ok": True,
            "ticker": t,
            "peer": peer,
            "horizon": h,
            "company": {
                "name": t,
                "sector": "N/D",
                "industry": "N/D",
                "country": "N/D"
            },
            "scores": {
                "fundamental": fundamental,
                "tecnico": tecnico,
                "sentimiento": sentimiento,
                "total": total,
                "veredicto": veredicto
            },
            "risk": {
                "stop_loss": round(price * 0.93, 2),
                "take_profit": round(price * 1.12, 2)
            },
            "details": [
                {"item": "Retorno 1Y", "value": ret, "status": s_ret, "why": why_ret, "action": act_ret},
                {"item": "Volatilidad", "value": vol, "status": s_vol, "why": why_vol, "action": act_vol},
                {"item": "RSI14", "value": rsi, "status": s_rsi, "why": why_rsi, "action": act_rsi},
            ],
            "comparativa": {
                t: {"retorno_1y_pct": ret, "volatilidad_pct": vol, "max_drawdown_pct": dd, "rsi14": rsi},
                peer: {"retorno_1y_pct": ret_p, "volatilidad_pct": vol_p, "max_drawdown_pct": dd_p, "rsi14": rsi_p},
            },
            "chart": {"labels": labels, "ticker_base100": ser1, "peer_base100": ser2}
        }

    except Exception as e:
        return {"ok": False, "error": "Falló el análisis en servidor.", "detail": str(e)}

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html><html lang="es"><head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>InvestMAC</title><script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body{font-family:Arial;background:#f2f4f8;margin:0;padding:20px}
.wrap{max-width:1000px;margin:0 auto}
.card{background:#fff;border-radius:12px;padding:16px;margin-bottom:12px;box-shadow:0 4px 12px rgba(0,0,0,.08)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
input,select,button{width:100%;padding:10px;border:1px solid #ccc;border-radius:8px;font-size:16px}
button{background:#111;color:#fff;border:none;cursor:pointer}
.k{font-size:12px;color:#555}.v{font-size:22px;font-weight:700}
.good{color:#15803d}.bad{color:#b91c1c}.neutral{color:#334155}
table{width:100%;border-collapse:collapse} th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}
.err{color:#b91c1c;font-weight:700}
</style></head><body>
<div class="wrap">
  <div class="card">
    <h2>InvestMAC</h2>
    <div class="grid2">
      <div><label>Ticker</label><input id="ticker" placeholder="AAPL, GGAL, NVDA"/></div>
      <div><label>Horizonte</label><select id="horizon">
        <option value="corto">Corto</option><option value="mediano" selected>Mediano</option><option value="largo">Largo</option>
      </select></div>
    </div>
    <div style="margin-top:10px"><button onclick="run()">OK</button></div>
    <div id="msg" class="err" style="margin-top:10px"></div>
  </div>

  <div class="card" id="main" style="display:none">
    <div class="grid3">
      <div><div class="k">Ticker</div><div class="v" id="tick">-</div></div>
      <div><div class="k">Competidor</div><div class="v" id="peer">-</div></div>
      <div><div class="k">Veredicto</div><div class="v" id="vered">-</div></div>
    </div>
  </div>

  <div class="card" id="detailCard" style="display:none"><h3>Detalle</h3><table id="detailTable"></table></div>
  <div class="card" id="cmp" style="display:none"><h3>Comparación</h3><table id="cmpTable"></table></div>
  <div class="card" id="ch" style="display:none"><h3>Gráfico</h3><canvas id="chart"></canvas></div>
</div>

<script>
let chartRef=null;
function cls(v){if(v==="good")return"good";if(v==="bad")return"bad";return"neutral";}
async function run(){
  const t=document.getElementById("ticker").value.trim();
  const h=document.getElementById("horizon").value;
  const msg=document.getElementById("msg");
  msg.textContent="";
  if(!t){msg.textContent="Ingresá un ticker.";return;}

  const res=await fetch(`/analyze/${encodeURIComponent(t)}?horizon=${encodeURIComponent(h)}`);
  const d=await res.json();
  if(!d.ok){msg.textContent=(d.error||"Error")+" "+(d.detail||"");return;}

  document.getElementById("main").style.display="block";
  document.getElementById("detailCard").style.display="block";
  document.getElementById("cmp").style.display="block";
  document.getElementById("ch").style.display="block";

  document.getElementById("tick").textContent=d.ticker;
  document.getElementById("peer").textContent=d.peer;
  document.getElementById("vered").textContent=d.scores?.veredicto ?? "-";

  document.getElementById("detailTable").innerHTML = `
    <thead><tr><th>Ítem</th><th>Valor</th><th>Estado</th><th>Por qué</th><th>Qué hacer</th></tr></thead>
    <tbody>${(d.details||[]).map(x=>`<tr><td>${x.item}</td><td>${x.value}</td><td class="${cls(x.status)}">${x.status}</td><td>${x.why}</td><td>${x.action}</td></tr>`).join("")}</tbody>`;

  const tkr=d.ticker, pr=d.peer, ct=d.comparativa[tkr], cp=d.comparativa[pr];
  document.getElementById("cmpTable").innerHTML=`
    <thead><tr><th>Métrica</th><th>${tkr}</th><th>${pr}</th></tr></thead>
    <tbody>
      <tr><td>Retorno 1Y %</td><td>${ct.retorno_1y_pct}</td><td>${cp.retorno_1y_pct}</td></tr>
      <tr><td>Volatilidad %</td><td>${ct.volatilidad_pct}</td><td>${cp.volatilidad_pct}</td></tr>
      <tr><td>Max Drawdown %</td><td>${ct.max_drawdown_pct}</td><td>${cp.max_drawdown_pct}</td></tr>
      <tr><td>RSI14</td><td>${ct.rsi14}</td><td>${cp.rsi14}</td></tr>
    </tbody>`;

  const ctx=document.getElementById("chart").getContext("2d");
  if(chartRef) chartRef.destroy();
  chartRef=new Chart(ctx,{type:"line",data:{labels:d.chart.labels,datasets:[
    {label:`${tkr} base100`,data:d.chart.ticker_base100,borderWidth:2,tension:0.2},
    {label:`${pr} base100`,data:d.chart.peer_base100,borderWidth:2,tension:0.2}
  ]}});
}
</script></body></html>
"""