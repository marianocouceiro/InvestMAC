from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import yfinance as yf
from statistics import mean, pstdev

app = FastAPI(title="InvestMAC API", version="5.0.0")


PEER_MAP = {
    "AAPL": "MSFT",
    "MSFT": "AAPL",
    "NVDA": "AMD",
    "AMD": "NVDA",
    "TSLA": "RIVN",
    "GGAL": "BMA",
    "BMA": "GGAL",
    "META": "GOOGL",
    "GOOGL": "META",
    "AMZN": "WMT",
}


def clamp(n, lo=1, hi=10):
    return max(lo, min(hi, n))


def sma(values, w):
    if len(values) < w:
        return values[-1] if values else 0.0
    return sum(values[-w:]) / w


def rsi(values, period=14):
    if len(values) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(-period, 0):
        d = values[i] - values[i - 1]
        if d >= 0:
            gains.append(d)
        else:
            losses.append(abs(d))
    avg_gain = sum(gains) / period if gains else 0.0
    avg_loss = sum(losses) / period if losses else 0.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def total_return(prices):
    if len(prices) < 2:
        return 0.0
    return (prices[-1] / prices[0]) - 1


def annualized_vol(prices):
    if len(prices) < 2:
        return 0.0
    rets = [(prices[i] / prices[i - 1]) - 1 for i in range(1, len(prices))]
    if len(rets) < 2:
        return 0.0
    return pstdev(rets) * (252 ** 0.5)


def max_drawdown(prices):
    if not prices:
        return 0.0
    peak = prices[0]
    mdd = 0.0
    for p in prices:
        peak = max(peak, p)
        dd = (p / peak) - 1
        mdd = min(mdd, dd)
    return mdd


def zscore_last(prices):
    if len(prices) < 5:
        return 0.0
    m = mean(prices)
    s = pstdev(prices)
    if s == 0:
        return 0.0
    return (prices[-1] - m) / s


def fmt_money(n):
    if n is None:
        return None
    try:
        n = float(n)
    except Exception:
        return None
    if abs(n) >= 1_000_000_000_000:
        return f"{n/1_000_000_000_000:.2f}T"
    if abs(n) >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    if abs(n) >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    return f"{n:,.2f}"


def item_eval(label, value, status, why, action):
    return {
        "label": label,
        "value": value,
        "status": status,   # good | bad | neutral
        "why": why,
        "action": action,
    }


def eval_pe(pe):
    if pe is None:
        return item_eval("P/E", "-", "neutral", "No hay dato disponible.", "Esperar confirmación.")
    try:
        pe = float(pe)
    except Exception:
        return item_eval("P/E", pe, "neutral", "Formato no válido.", "Revisar fuente.")
    if pe < 10:
        return item_eval("P/E", round(pe, 2), "good", "Valuación baja respecto a ganancias.", "Posible oportunidad si negocio es sólido.")
    if pe > 35:
        return item_eval("P/E", round(pe, 2), "bad", "Valuación exigente, riesgo de corrección.", "Entrar con cautela o por etapas.")
    return item_eval("P/E", round(pe, 2), "neutral", "Valuación razonable intermedia.", "Mantener seguimiento trimestral.")


def eval_forward_pe(fpe, pe):
    if fpe is None:
        return item_eval("Forward P/E", "-", "neutral", "No hay estimación forward.", "No usar como señal principal.")
    try:
        fpe = float(fpe)
    except Exception:
        return item_eval("Forward P/E", fpe, "neutral", "Formato no válido.", "Revisar fuente.")
    if pe is not None:
        try:
            pef = float(pe)
            if fpe < pef:
                return item_eval("Forward P/E", round(fpe, 2), "good", "El mercado espera mejora de ganancias.", "Sesgo positivo si se confirma guidance.")
        except Exception:
            pass
    if fpe > 35:
        return item_eval("Forward P/E", round(fpe, 2), "bad", "Proyección cara para próximos 12 meses.", "Evitar sobreexposición.")
    return item_eval("Forward P/E", round(fpe, 2), "neutral", "No muestra señal extrema.", "Complementar con crecimiento y márgenes.")


def eval_dividend(dy):
    if dy is None:
        return item_eval("Dividend Yield %", "-", "neutral", "No paga dividendo o no hay dato.", "No evaluar este activo por renta.")
    try:
        dy = float(dy)
    except Exception:
        return item_eval("Dividend Yield %", dy, "neutral", "Formato no válido.", "Revisar dato.")
    if dy >= 3:
        return item_eval("Dividend Yield %", round(dy, 2), "good", "Rendimiento por dividendos atractivo.", "Útil para perfiles de renta.")
    if dy < 1:
        return item_eval("Dividend Yield %", round(dy, 2), "bad", "Aporte por dividendo muy bajo.", "Priorizar crecimiento/precio, no renta.")
    return item_eval("Dividend Yield %", round(dy, 2), "neutral", "Rendimiento moderado.", "Mantener como componente secundario.")


def eval_rsi(r):
    if r > 70:
        return item_eval("RSI14", round(r, 2), "bad", "Sobrecompra: posible pausa/corrección.", "Esperar pullback o confirmar ruptura con volumen.")
    if r < 30:
        return item_eval("RSI14", round(r, 2), "good", "Sobreventa: posible rebote técnico.", "Buscar confirmación con vela/volumen.")
    return item_eval("RSI14", round(r, 2), "neutral", "Zona media, sin extremo técnico.", "Mantener estrategia base.")


def eval_trend(last, s20, s50, s200):
    if s50 > s200 and last > s20:
        return item_eval("Tendencia (SMA20/50/200)", f"{last:.2f} / {s20:.2f} / {s50:.2f} / {s200:.2f}", "good",
                         "Tendencia alcista consistente.", "Favorece mantener/comprar en retrocesos.")
    if s50 < s200 and last < s20:
        return item_eval("Tendencia (SMA20/50/200)", f"{last:.2f} / {s20:.2f} / {s50:.2f} / {s200:.2f}", "bad",
                         "Tendencia bajista estructural.", "Reducir riesgo y esperar reversión.")
    return item_eval("Tendencia (SMA20/50/200)", f"{last:.2f} / {s20:.2f} / {s50:.2f} / {s200:.2f}", "neutral",
                     "Tendencia mixta o transición.", "Operar con tamaño moderado.")


def fetch_data(ticker: str):
    tk = yf.Ticker(ticker)
    hist = tk.history(period="1y", interval="1d")
    info = tk.info if tk else {}
    if hist is None or hist.empty:
        return None, None, None, None, None
    closes = [float(x) for x in hist["Close"].dropna().tolist()]
    dates = [d.strftime("%Y-%m-%d") for d in hist.index.to_pydatetime()]
    return tk, info, hist, closes, dates


def company_profile(info: dict):
    dy = info.get("dividendYield")
    return {
        "name": info.get("longName") or info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "country": info.get("country"),
        "currency": info.get("currency"),
        "exchange": info.get("exchange"),
        "website": info.get("website"),
        "market_cap": fmt_money(info.get("marketCap")),
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "dividend_yield_pct": round(dy * 100, 2) if dy is not None else None,
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
    }


def score_asset(closes, horizon):
    last = closes[-1]
    ret = total_return(closes)
    rsi14 = rsi(closes, 14)
    s20 = sma(closes, 20)
    s50 = sma(closes, 50)
    s200 = sma(closes, 200)

    tecnico = 5
    if s50 > s200:
        tecnico += 2
    if last > s20:
        tecnico += 1
    if 45 <= rsi14 <= 65:
        tecnico += 1
    if rsi14 > 75 or rsi14 < 25:
        tecnico -= 1
    tecnico = clamp(tecnico)

    if ret > 0.35:
        fundamental = 8
    elif ret > 0.15:
        fundamental = 7
    elif ret > 0:
        fundamental = 6
    elif ret > -0.15:
        fundamental = 5
    else:
        fundamental = 4

    z = zscore_last(closes)
    if z > 1.5:
        sentimiento = 4
    elif z < -1.5:
        sentimiento = 5
    else:
        sentimiento = 6
    sentimiento = clamp(sentimiento)

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

    return {
        "fundamental": fundamental,
        "tecnico": tecnico,
        "sentimiento": sentimiento,
        "total": round(total, 2),
        "veredicto": veredicto,
        "retorno_1y": round(ret * 100, 2),
        "rsi14": round(rsi14, 2),
        "sma20": round(s20, 2),
        "sma50": round(s50, 2),
        "sma200": round(s200, 2),
        "last": round(last, 2),
        "stop_loss": round(last * 0.93, 2),
        "take_profit": round(last * 1.12, 2),
        "volatilidad": round(annualized_vol(closes) * 100, 2),
        "max_drawdown": round(max_drawdown(closes) * 100, 2),
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/analyze/{ticker}")
def analyze(ticker: str, horizon: str = "mediano"):
    t = ticker.upper().strip()
    h = horizon.lower().strip()
    if h not in {"corto", "mediano", "largo"}:
        h = "mediano"
    peer = PEER_MAP.get(t, "SPY")

    _, info_t, _, close_t, dates_t = fetch_data(t)
    _, info_p, _, close_p, dates_p = fetch_data(peer)

    if not close_t:
        return {"error": f"No se pudieron obtener datos para {t}"}
    if not close_p:
        return {"error": f"No se pudieron obtener datos para competidor {peer}"}

    profile_t = company_profile(info_t or {})
    profile_p = company_profile(info_p or {})

    sc_t = score_asset(close_t, h)
    sc_p = score_asset(close_p, h)

    # Evaluaciones detalladas para colorear + explicación
    fundamentals_detail = [
        eval_pe(profile_t.get("pe_ratio")),
        eval_forward_pe(profile_t.get("forward_pe"), profile_t.get("pe_ratio")),
        eval_dividend(profile_t.get("dividend_yield_pct")),
        item_eval(
            "Retorno 1Y %",
            sc_t["retorno_1y"],
            "good" if sc_t["retorno_1y"] > 15 else ("bad" if sc_t["retorno_1y"] < -10 else "neutral"),
            "Mide momentum de mediano plazo.",
            "Si es negativo fuerte, reducir exposición; si es alto, proteger ganancias."
        ),
    ]

    tech_trend = eval_trend(sc_t["last"], sc_t["sma20"], sc_t["sma50"], sc_t["sma200"])
    technical_detail = [
        eval_rsi(sc_t["rsi14"]),
        tech_trend,
        item_eval(
            "Volatilidad 1Y %",
            sc_t["volatilidad"],
            "bad" if sc_t["volatilidad"] > 45 else ("good" if sc_t["volatilidad"] < 20 else "neutral"),
            "Mayor volatilidad implica mayor riesgo de oscilaciones.",
            "A mayor volatilidad, usar menor tamaño de posición."
        ),
        item_eval(
            "Max Drawdown 1Y %",
            sc_t["max_drawdown"],
            "bad" if sc_t["max_drawdown"] < -35 else ("good" if sc_t["max_drawdown"] > -15 else "neutral"),
            "Caída máxima histórica del período analizado.",
            "Si drawdown fue profundo, usar stop más estricto."
        ),
    ]

    # Base 100 para gráfico comparativo
    n = min(len(close_t), len(close_p), len(dates_t), len(dates_p))
    labels = dates_t[-n:]
    bt = close_t[-n][0] if isinstance(close_t[-n], list) else close_t[-n]
    bp = close_p[-n][0] if isinstance(close_p[-n], list) else close_p[-n]
    t_rel = [round((x / bt) * 100, 2) for x in close_t[-n:]]
    p_rel = [round((x / bp) * 100, 2) for x in close_p[-n:]]

    return {
        "ticker": t,
        "peer": peer,
        "horizon": h,
        "company": profile_t,
        "peer_company": profile_p,
        "summary": f"{t}: veredicto {sc_t['veredicto']} con score {sc_t['total']}.",
        "scores": {
            "fundamental": sc_t["fundamental"],
            "tecnico": sc_t["tecnico"],
            "sentimiento": sc_t["sentimiento"],
            "total": sc_t["total"],
            "veredicto": sc_t["veredicto"],
        },
        "risk": {
            "stop_loss": sc_t["stop_loss"],
            "take_profit": sc_t["take_profit"],
        },
        "fundamentals_detail": fundamentals_detail,
        "technical_detail": technical_detail,
        "comparativa": {
            t: {
                "retorno_1y_pct": sc_t["retorno_1y"],
                "volatilidad_pct": sc_t["volatilidad"],
                "max_drawdown_pct": sc_t["max_drawdown"],
                "rsi14": sc_t["rsi14"],
            },
            peer: {
                "retorno_1y_pct": sc_p["retorno_1y"],
                "volatilidad_pct": sc_p["volatilidad"],
                "max_drawdown_pct": sc_p["max_drawdown"],
                "rsi14": sc_p["rsi14"],
            },
        },
        "chart": {
            "labels": labels,
            "ticker_base100": t_rel,
            "peer_base100": p_rel,
        },
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>InvestMAC</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body { margin:0; padding:20px; font-family:Arial,sans-serif; background:#f3f5f9; }
    .wrap { max-width:1150px; margin:0 auto; }
    .card { background:#fff; border-radius:14px; padding:16px; margin-bottom:14px; box-shadow:0 4px 14px rgba(0,0,0,.08); }
    h1, h3 { margin:0 0 10px; }
    .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
    .grid3 { display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; }
    .full { grid-column:1/-1; }
    label { display:block; font-size:13px; font-weight:700; margin:8px 0 6px; color:#334155; }
    input, select, button { width:100%; padding:10px; border:1px solid #cbd5e1; border-radius:10px; font-size:16px; }
    button { background:#0f172a; color:#fff; border:none; cursor:pointer; }
    .k { font-size:11px; color:#64748b; text-transform:uppercase; margin-bottom:4px; }
    .v { font-size:20px; font-weight:700; }
    .buy { color:#15803d; } .hold { color:#a16207; } .sell { color:#b91c1c; }
    table { width:100%; border-collapse:collapse; }
    th, td { padding:8px; border-bottom:1px solid #e2e8f0; font-size:14px; text-align:left; vertical-align:top; }
    .status-good { color:#15803d; font-weight:700; }
    .status-bad { color:#b91c1c; font-weight:700; }
    .status-neutral { color:#334155; font-weight:700; }
    .mini { font-size:12px; color:#64748b; margin-top:4px; }
    .pill { display:inline-block; padding:4px 8px; border-radius:999px; background:#eef2ff; color:#3730a3; font-size:12px; margin-left:8px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>InvestMAC <span class="pill">Análisis Visual</span></h1>
      <div class="grid2">
        <div>
          <label>Ticker</label>
          <input id="ticker" placeholder="Ej: AAPL, GGAL, NVDA..." />
        </div>
        <div>
          <label>Horizonte</label>
          <select id="horizon">
            <option value="corto">Corto (1-3 meses)</option>
            <option value="mediano" selected>Mediano (3-12 meses)</option>
            <option value="largo">Largo (12-36 meses)</option>
          </select>
        </div>
        <div class="full"><button onclick="runAnalysis()">OK</button></div>
      </div>
    </div>

    <div class="card" id="profileCard" style="display:none;">
      <h3>Perfil Empresa</h3>
      <div class="grid3">
        <div><div class="k">Empresa</div><div class="v" id="cName">-</div></div>
        <div><div class="k">Ticker</div><div class="v" id="cTicker">-</div></div>
        <div><div class="k">Competidor</div><div class="v" id="cPeer">-</div></div>
        <div><div class="k">Sector</div><div id="cSector">-</div></div>
        <div><div class="k">Industria</div><div id="cIndustry">-</div></div>
        <div><div class="k">País</div><div id="cCountry">-</div></div>
      </div>
    </div>

    <div class="card" id="scoreCard" style="display:none;">
      <h3>Resumen Ejecutivo</h3>
      <div class="grid3">
        <div><div class="k">Veredicto</div><div class="v hold" id="sVerdict">-</div></div>
        <div><div class="k">Score Total</div><div class="v" id="sTotal">-</div></div>
        <div><div class="k">Horizonte</div><div class="v" id="sHorizon">-</div></div>
        <div><div class="k">Fundamental</div><div class="v" id="sFund">-</div></div>
        <div><div class="k">Técnico</div><div class="v" id="sTec">-</div></div>
        <div><div class="k">Sentimiento</div><div class="v" id="sSent">-</div></div>
      </div>
      <p id="sSummary"></p>
    </div>

    <div class="card" id="fundCard" style="display:none;">
      <h3>Fundamental (semáforo + recomendación breve)</h3>
      <table id="fundTable"></table>
    </div>

    <div class="card" id="techCard" style="display:none;">
      <h3>Técnico (semáforo + recomendación breve)</h3>
      <table id="techTable"></table>
      <div class="grid2" style="margin-top:10px;">
        <div><div class="k">Stop Loss</div><div class="v" id="rSL">-</div></div>
        <div><div class="k">Take Profit</div><div class="v" id="rTP">-</div></div>
      </div>
    </div>

    <div class="card" id="cmpCard" style="display:none;">
      <h3>Comparación con Competidor</h3>
      <table id="cmpTable"></table>
    </div>

    <div class="card" id="chartCard" style="display:none;">
      <h3>Performance Relativa 1Y (Base 100)</h3>
      <canvas id="perfChart" height="110"></canvas>
    </div>
  </div>

  <script>
    let chartRef = null;

    function show(ids){ ids.forEach(id => document.getElementById(id).style.display = "block"); }
    function txt(id, val){ document.getElementById(id).textContent = (val ?? "-"); }

    function verdictClass(v){
      if(v === "COMPRAR") return "buy";
      if(v === "VENDER") return "sell";
      return "hold";
    }

    function statusClass(s){
      if(s === "good") return "status-good";
      if(s === "bad") return "status-bad";
      return "status-neutral";
    }

    function statusText(s){
      if(s === "good") return "Bueno";
      if(s === "bad") return "Malo";
      return "Normal";
    }

    function buildDetailTable(items){
      return `
        <thead>
          <tr>
            <th>Ítem</th>
            <th>Valor</th>
            <th>Estado</th>
            <th>Por qué</th>
            <th>Qué hacer</th>
          </tr>
        </thead>
        <tbody>
          ${items.map(it => `
            <tr>
              <td><strong>${it.label}</strong></td>
              <td>${it.value ?? "-"}</td>
              <td class="${statusClass(it.status)}">${statusText(it.status)}</td>
              <td>${it.why ?? "-"}</td>
              <td>${it.action ?? "-"}</td>
            </tr>
          `).join("")}
        </tbody>
      `;
    }

    async function runAnalysis(){
      const ticker = document.getElementById("ticker").value.trim();
      const horizon = document.getElementById("horizon").value;
      if(!ticker){ alert("Ingresá un ticker."); return; }

      const res = await fetch(`/analyze/${encodeURIComponent(ticker)}?horizon=${encodeURIComponent(horizon)}`);
      const data = await res.json();

      if(data.error){
        alert(data.error);
        return;
      }

      show(["profileCard","scoreCard","fundCard","techCard","cmpCard","chartCard"]);

      // Perfil
      txt("cName", data.company?.name);
      txt("cTicker", data.ticker);
      txt("cPeer", data.peer);
      txt("cSector", data.company?.sector);
      txt("cIndustry", data.company?.industry);
      txt("cCountry", data.company?.country);

      // Resumen
      txt("sTotal", data.scores?.total);
      txt("sHorizon", data.horizon);
      txt("sFund", data.scores?.fundamental);
      txt("sTec", data.scores?.tecnico);
      txt("sSent", data.scores?.sentimiento);
      txt("sSummary", data.summary);

      const sv = document.getElementById("sVerdict");
      sv.textContent = data.scores?.veredicto ?? "-";
      sv.className = "v " + verdictClass(data.scores?.veredicto);

      // Tablas detalladas con color + explicación
      document.getElementById("fundTable").innerHTML = buildDetailTable(data.fundamentals_detail || []);
      document.getElementById("techTable").innerHTML = buildDetailTable(data.technical_detail || []);

      // Riesgo
      txt("rSL", data.risk?.stop_loss);
      txt("rTP", data.risk?.take_profit);

      // Comparativa
      const t = data.ticker;
      const p = data.peer;
      const ct = data.comparativa[t];
      const cp = data.comparativa[p];
      document.getElementById("cmpTable").innerHTML = `
        <thead><tr><th>Métrica</th><th>${t}</th><th>${p}</th></tr></thead>
        <tbody>
          <tr><td>Retorno 1Y (%)</td><td>${ct?.retorno_1y_pct ?? "-"}</td><td>${cp?.retorno_1y_pct ?? "-"}</td></tr>
          <tr><td>Volatilidad (%)</td><td>${ct?.volatilidad_pct ?? "-"}</td><td>${cp?.volatilidad_pct ?? "-"}</td></tr>
          <tr><td>Max Drawdown (%)</td><td>${ct?.max_drawdown_pct ?? "-"}</td><td>${cp?.max_drawdown_pct ?? "-"}</td></tr>
          <tr><td>RSI14</td><td>${ct?.rsi14 ?? "-"}</td><td>${cp?.rsi14 ?? "-"}</td></tr>
        </tbody>
      `;

      // Gráfico
      const ctx = document.getElementById("perfChart").getContext("2d");
      if(chartRef) chartRef.destroy();
      chartRef = new Chart(ctx, {
        type: "line",
        data: {
          labels: data.chart?.labels ?? [],
          datasets: [
            { label: `${t} Base100`, data: data.chart?.ticker_base100 ?? [], borderWidth: 2, tension: 0.2 },
            { label: `${p} Base100`, data: data.chart?.peer_base100 ?? [], borderWidth: 2, tension: 0.2 }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: true,
          scales: { x: { ticks: { maxTicksLimit: 8 } } }
        }
      });
    }
  </script>
</body>
</html>
"""