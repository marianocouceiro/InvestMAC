from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import yfinance as yf
from statistics import mean, pstdev

app = FastAPI(title="InvestMAC API", version="5.1.0")

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

def fetch_data(ticker: str):
    tk = yf.Ticker(ticker)
    hist = tk.history(period="1y", interval="1d")
    info = tk.info if tk else {}
    if hist is None or hist.empty:
        return None, None, None, None
    closes = [float(x) for x in hist["Close"].dropna().tolist()]
    dates = [d.strftime("%Y-%m-%d") for d in hist.index.to_pydatetime()]
    return info, closes, dates, hist

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
    if h not in {"corto", "mediano", "largo"}:
        h = "mediano"

    peer = PEER_MAP.get(t, "SPY")
    info_t, close_t, dates_t, _ = fetch_data(t)
    info_p, close_p, dates_p, _ = fetch_data(peer)

    if not close_t:
        return {"error": f"No se pudieron obtener datos para {t}"}
    if not close_p:
        return {"error": f"No se pudieron obtener datos para competidor {peer}"}

    sc_t = score_asset(close_t, h)
    sc_p = score_asset(close_p, h)

    n = min(len(close_t), len(close_p), len(dates_t), len(dates_p))
    labels = dates_t[-n:]
    bt = close_t[-n]
    bp = close_p[-n]
    t_rel = [round((x / bt) * 100, 2) for x in close_t[-n:]]
    p_rel = [round((x / bp) * 100, 2) for x in close_p[-n:]]

    return {
        "ticker": t,
        "peer": peer,
        "horizon": h,
        "summary": f"{t}: score {sc_t['total']} | veredicto {sc_t['veredicto']}",
        "scores": {
            "fundamental": sc_t["fundamental"],
            "tecnico": sc_t["tecnico"],
            "sentimiento": sc_t["sentimiento"],
            "total": sc_t["total"],
            "veredicto": sc_t["veredicto"]
        },
        "risk": {
            "stop_loss": sc_t["stop_loss"],
            "take_profit": sc_t["take_profit"]
        },
        "comparativa": {
            t: {
                "retorno_1y_pct": sc_t["retorno_1y_pct"],
                "volatilidad_pct": sc_t["volatilidad_pct"],
                "max_drawdown_pct": sc_t["max_drawdown_pct"],
                "rsi14": sc_t["rsi14"]
            },
            peer: {
                "retorno_1y_pct": sc_p["retorno_1y_pct"],
                "volatilidad_pct": sc_p["volatilidad_pct"],
                "max_drawdown_pct": sc_p["max_drawdown_pct"],
                "rsi14": sc_p["rsi14"]
            }
        },
        "chart": {
            "labels": labels,
            "ticker_base100": t_rel,
            "peer_base100": p_rel
        }
    }

@app.get("/", response_class=HTMLResponse)
def home():
    return "<h1>InvestMAC online ✅</h1><p>Probá /health y /analyze/AAPL?horizon=mediano</p>"