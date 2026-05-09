from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import traceback

app = FastAPI(title="InvestMAC API", version="7.2.0")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/analyze/{ticker}")
def analyze(ticker: str, horizon: str = "mediano"):
    """
    Versión robusta:
    - Si yfinance falla en Vercel, devuelve error JSON controlado (no 500 crudo).
    """
    try:
        import yfinance as yf
        from statistics import pstdev

        t = ticker.upper().strip()
        h = horizon.lower().strip()
        if h not in {"corto", "mediano", "largo"}:
            h = "mediano"

        peer_map = {
            "AAPL": "MSFT", "MSFT": "AAPL",
            "NVDA": "AMD", "AMD": "NVDA",
            "TSLA": "RIVN", "GGAL": "BMA", "BMA": "GGAL"
        }
        peer = peer_map.get(t, "SPY")

        def fetch_close(sym):
            hist = yf.Ticker(sym).history(period="1y", interval="1d")
            if hist is None or hist.empty:
                return None
            return [float(x) for x in hist["Close"].dropna().tolist()]

        c1 = fetch_close(t)
        c2 = fetch_close(peer)

        if not c1:
            return {"ok": False, "error": f"No se pudieron obtener datos para {t}."}
        if not c2:
            return {"ok": False, "error": f"No se pudieron obtener datos para competidor {peer}."}

        def ret_1y(prices):
            if len(prices) < 2:
                return 0.0
            return (prices[-1] / prices[0] - 1) * 100

        def vol(prices):
            if len(prices) < 3:
                return 0.0
            rets = [(prices[i] / prices[i - 1] - 1) for i in range(1, len(prices))]
            return pstdev(rets) * (252 ** 0.5) * 100

        r1 = round(ret_1y(c1), 2)
        r2 = round(ret_1y(c2), 2)
        v1 = round(vol(c1), 2)
        v2 = round(vol(c2), 2)

        # Score simple
        fundamental = 7 if r1 > 10 else (6 if r1 > 0 else 4)
        tecnico = 6 if c1[-1] > c1[-20] else 4
        sentimiento = 6
        total = round((fundamental + tecnico + sentimiento) / 3, 2)

        if total >= 7.5:
            veredicto = "COMPRAR"
        elif total <= 4.5:
            veredicto = "VENDER"
        else:
            veredicto = "MANTENER"

        # gráfico base100
        n = min(len(c1), len(c2))
        b1, b2 = c1[-n], c2[-n]
        rel1 = [round((x / b1) * 100, 2) for x in c1[-n:]]
        rel2 = [round((x / b2) * 100, 2) for x in c2[-n:]]
        labels = [str(i + 1) for i in range(n)]

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
            "comparativa": {
                t: {"retorno_1y_pct": r1, "volatilidad_pct": v1},
                peer: {"retorno_1y_pct": r2, "volatilidad_pct": v2}
            },
            "chart": {
                "labels": labels,
                "ticker_base100": rel1,
                "peer_base100": rel2
            }
        }

    except Exception as e:
        return {
            "ok": False,
            "error": "Falló el análisis en servidor.",
            "detail": str(e),
            "trace_hint": "Revisar logs de Vercel / dependencias."
        }


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html lang="es">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
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
    .err{color:#b91c1c;font-weight:700}
  </style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h2>InvestMAC</h2>
    <div class="grid2">
      <div><label>Ticker</label><input id="ticker" placeholder="AAPL, GGAL, NVDA"/></div>
      <div><label>Horizonte</label>
        <select id="horizon">
          <option value="corto">Corto</option>
          <option value="mediano" selected>Mediano</option>
          <option value="largo">Largo</option>
        </select>
      </div>
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

  <div class="card" id="cmp" style="display:none">
    <h3>Comparación</h3>
    <table id="cmpTable"></table>
  </div>

  <div class="card" id="ch" style="display:none">
    <h3>Gráfico</h3><canvas id="chart"></canvas>
  </div>
</div>

<script>
let chartRef = null;

async function run(){
  const t = document.getElementById("ticker").value.trim();
  const h = document.getElementById("horizon").value;
  const msg = document.getElementById("msg");
  msg.textContent = "";

  if(!t){ msg.textContent = "Ingresá un ticker."; return; }

  try {
    const res = await fetch(`/analyze/${encodeURIComponent(t)}?horizon=${encodeURIComponent(h)}`);
    const d = await res.json(); // ahora siempre debería ser JSON

    if(!d.ok){
      msg.textContent = d.error + (d.detail ? (" | " + d.detail) : "");
      return;
    }

    document.getElementById("main").style.display="block";
    document.getElementById("cmp").style.display="block";
    document.getElementById("ch").style.display="block";

    document.getElementById("tick").textContent = d.ticker ?? "-";
    document.getElementById("peer").textContent = d.peer ?? "-";
    document.getElementById("vered").textContent = d.scores?.veredicto ?? "-";

    const tkr=d.ticker, pr=d.peer, ct=d.comparativa[tkr], cp=d.comparativa[pr];
    document.getElementById("cmpTable").innerHTML=`
      <thead><tr><th>Métrica</th><th>${tkr}</th><th>${pr}</th></tr></thead>
      <tbody>
        <tr><td>Retorno 1Y %</td><td>${ct?.retorno_1y_pct ?? "-"}</td><td>${cp?.retorno_1y_pct ?? "-"}</td></tr>
        <tr><td>Volatilidad %</td><td>${ct?.volatilidad_pct ?? "-"}</td><td>${cp?.volatilidad_pct ?? "-"}</td></tr>
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
  } catch (e) {
    msg.textContent = "Error de red o de parseo de respuesta.";
  }
}
</script>
</body>
</html>
"""