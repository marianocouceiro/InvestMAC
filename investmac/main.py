from fastapi import FastAPI

app = FastAPI(title="InvestMAC API", version="1.0.0")

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/")
def home():
    return {"ok": True, "app": "InvestMAC"}

@app.get("/analyze/{ticker}")
def analyze(ticker: str, horizon: str = "mediano"):
    return {
        "ticker": ticker.upper(),
        "horizon": horizon,
        "veredicto": "MANTENER",
        "nota": "Deploy base funcionando"
    }