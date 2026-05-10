import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.signal import argrelextrema
import ccxt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Analizador Pro - Dual Mode", page_icon="📈", layout="wide")

# ---- CSS compacto ----
st.markdown("""
<style>
    .main .block-container { padding-top: 0.5rem !important; padding-bottom: 0rem !important; }
    div[data-testid="stMetric"] { background-color: #1e1e1ecc; padding: 4px 6px; border-radius: 8px; text-align: center; }
    div[data-testid="stMetric"] label, div[data-testid="stMetric"] div { color: #ffffff !important; }
    section[data-testid="stSidebar"] { width: 280px !important; }
    .dataframe td, .dataframe th { font-size: 0.8rem !important; padding: 2px 6px !important; }
    .stMarkdown, .stAlert, .stSuccess { margin-bottom: 0.2rem; }
    h1, h2, h3 { margin-bottom: 0.2rem; margin-top: 0.2rem; }
    hr { margin: 0.3rem 0; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Analizador Financiero Pro")
st.caption("Modo AndyStopLoss (RSI+ASL21+Soportes) | Modo Completo (Técnico+Chartismo+Fundamental) | Cripto: Binance (~100ms)")

# Inicializar session_state
for key in ['datos', 'analisis', 'simbolo', 'precio_entrada_base', 'modo_analisis', 'info']:
    if key not in st.session_state:
        st.session_state[key] = None

# ------------------------------------------------------------
# FUNCIONES DE INDICADORES COMUNES
# ------------------------------------------------------------
def calcular_indicadores(df):
    df = df.copy()
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # WMA_22 manual
    def calc_wma(series, window):
        weights = np.arange(1, window+1)
        return series.rolling(window).apply(lambda x: np.sum(weights * x) / weights.sum(), raw=True)
    df['WMA_22'] = calc_wma(df['Close'], 22)
    
    df['ASL21'] = (df['EMA_20'] + df['WMA_22']) / 2
    df['SMA_30'] = df['Close'].rolling(30).mean()
    df['EMA_150'] = df['Close'].ewm(span=150, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Diff'] = df['MACD'] - df['MACD_Signal']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['RSI_EMA20'] = df['RSI'].ewm(span=20, adjust=False).mean()
    df['RSI_WMA22'] = calc_wma(df['RSI'], 22)
    df['RSI_ASL21'] = (df['RSI_EMA20'] + df['RSI_WMA22']) / 2
    
    # Bollinger
    df['BB_Middle'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2*bb_std
    df['BB_Lower'] = df['BB_Middle'] - 2*bb_std
    df['BB_Pos'] = 100 * (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Estocástico
    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(1)
    df['ATR'] = tr.rolling(14).mean()
    
    return df

def analisis_chartismo(df):
    precios = df['Close'].values
    orden = max(3, min(10, len(precios)//20))
    maximos = argrelextrema(precios, np.greater, order=orden)[0]
    minimos = argrelextrema(precios, np.less, order=orden)[0]
    soportes = [round(precios[i],2) for i in minimos[-3:] if i < len(precios)]
    resistencias = [round(precios[i],2) for i in maximos[-3:] if i < len(precios)]
    
    punt = 0
    figs = []
    if soportes and abs(precios[-1]-soportes[0])/precios[-1] < 0.02:
        punt += 30
        figs.append(f" Soporte ${soportes[0]:,.2f}")
    if resistencias and abs(resistencias[0]-precios[-1])/precios[-1] < 0.02:
        figs.append(f" Resistencia ${resistencias[0]:,.2f}")
    
    if len(precios) > 20:
        pend = np.polyfit(range(20), precios[-20:], 1)[0]
        if pend > 0.002*precios[-1]:
            punt += 15
            figs.append(" Tendencia alcista")
        elif pend < -0.002*precios[-1]:
            figs.append(" Tendencia bajista")
        else:
            punt += 5
            figs.append(" Tendencia lateral")
    
    return {'puntuacion': punt, 'figuras': figs, 'soportes': soportes, 'resistencias': resistencias}

def analisis_tecnico_completo(df):
    ult = df.iloc[-1]
    precio = ult['Close']
    punt = 0
    señales = []
    
    if precio > ult['EMA_9'] and precio > ult['EMA_21']:
        punt += 20
        señales.append(" Precio sobre EMA 9 y 21")
    elif precio > ult['EMA_21']:
        punt += 10
        señales.append(" Precio sobre EMA 21")
    else:
        señales.append(" Precio bajo EMA 21")
    
    if ult['MACD'] > ult['MACD_Signal']:
        punt += 25
        señales.append(" MACD alcista")
    else:
        señales.append(" MACD bajista")
    
    if ult['RSI'] < 30:
        punt += 20
        señales.append(f" RSI {ult['RSI']:.0f} - SOBREVENTA")
    elif ult['RSI'] > 70:
        señales.append(f" RSI {ult['RSI']:.0f} - SOBRECOMPRA")
    else:
        punt += 5
    
    if ult['BB_Pos'] < 20:
        punt += 15
        señales.append(" Cerca de banda inferior")
    elif ult['BB_Pos'] > 80:
        señales.append(" Cerca de banda superior")
    
    if ult['Stoch_K'] < 20 and ult['Stoch_D'] < 20:
        punt += 15
        señales.append(" Estocástico sobreventa")
    
    if punt >= 70:
        rec, acc = " FUERTEMENTE ALCISTA", "COMPRAR"
    elif punt >= 50:
        rec, acc = " LIGERAMENTE ALCISTA", "CONSIDERAR"
    elif punt >= 30:
        rec, acc = " NEUTRAL", "ESPERAR"
    else:
        rec, acc = " BAJISTA", "NO COMPRAR"
    
    return {'puntuacion': punt, 'recomendacion': rec, 'accion': acc, 'señales': señales, 'precio': precio, 'rsi': ult['RSI']}

def analisis_fundamental(simbolo, info):
    punt = 0
    expl = []
    if not info:
        return {'puntuacion': 0, 'explicaciones': ['No hay datos fundamentales (criptomoneda)'], 'accion': 'N/A'}
    
    per = info.get('trailingPE')
    if per and per != 'N/A':
        try:
            if float(per) < 15:
                punt += 30
                expl.append(f" PER: {float(per):.2f} (infravalorada)")
        except: pass
    
    peg = info.get('pegRatio')
    if peg and peg != 'N/A':
        try:
            if float(peg) < 1:
                punt += 25
                expl.append(f" PEG: {float(peg):.2f} (crecimiento barato)")
        except: pass
    
    roe = info.get('returnOnEquity')
    if roe and roe != 'N/A':
        try:
            if float(roe) > 0.15:
                punt += 25
                expl.append(f" ROE: {float(roe)*100:.1f}% (rentable)")
        except: pass
    
    return {'puntuacion': punt, 'explicaciones': expl, 'accion': 'INVERTIR' if punt>=50 else 'EVITAR'}

def analisis_andystoploss(df):
    ult = df.iloc[-1]
    precios = df['Close'].values
    orden = max(3, min(10, len(precios)//20))
    maximos = argrelextrema(precios, np.greater, order=orden)[0]
    minimos = argrelextrema(precios, np.less, order=orden)[0]
    soportes = [round(precios[i],2) for i in minimos[-3:] if i < len(precios)]
    resistencias = [round(precios[i],2) for i in maximos[-3:] if i < len(precios)]
    
    punt_ent, sen_ent = 0, []
    if soportes and abs(ult['Close'] - soportes[0]) / ult['Close'] < 0.02:
        punt_ent += 30
        sen_ent.append(f" Soporte ${soportes[0]:,.2f}")
    if ult['RSI'] < 30:
        punt_ent += 25
        sen_ent.append(f" RSI {ult['RSI']:.0f} (sobreventa)")
    if ult['Close'] > ult['ASL21']:
        punt_ent += 15
        sen_ent.append(f" Sobre ASL21 ${ult['ASL21']:,.2f}")
    if ult['Close'] > ult['EMA_150'] and ult['Close'] > ult['EMA_200']:
        punt_ent += 10
        sen_ent.append(" Sobre EMAs 150/200")
    
    punt_sal, sen_sal = 0, []
    if resistencias and abs(resistencias[0] - ult['Close']) / ult['Close'] < 0.02:
        punt_sal += 30
        sen_sal.append(f" Resistencia ${resistencias[0]:,.2f}")
    if ult['RSI'] > 70:
        punt_sal += 30
        sen_sal.append(f" RSI {ult['RSI']:.0f} (sobrecompra)")
    if ult['Close'] < ult['ASL21']:
        punt_sal += 25
        sen_sal.append(f" Bajo ASL21 ${ult['ASL21']:,.2f}")
    if ult['Close'] < ult['SMA_30']:
        punt_sal += 20
        sen_sal.append(f" Bajo SMA30 ${ult['SMA_30']:,.2f}")
    
    neta = punt_ent - punt_sal
    if neta >= 40 and punt_ent >= 50:
        decision, color = " COMPRAR", "green"
    elif punt_sal >= 40:
        decision, color = " VENDER", "red"
    elif punt_ent >= 30:
        decision, color = " DUDAR", "orange"
    else:
        decision, color = " ESPERAR", "gray"
    
    return {
        'puntuacion': neta,
        'punt_ent': punt_ent, 'punt_sal': punt_sal,
        'decision': decision, 'color': color,
        'sen_ent': sen_ent, 'sen_sal': sen_sal,
        'precio': ult['Close'], 'rsi': ult['RSI'], 'asl21': ult['ASL21'], 'sma30': ult['SMA_30'],
        'soportes': soportes, 'resistencias': resistencias
    }

# ------------------------------------------------------------
# OBTENER DATOS
# ------------------------------------------------------------
def obtener_datos_cripto(simbolo_ccxt, intervalo, limite=500):
    exchange = ccxt.binance()
    timeframe_map = {'1d': '1d', '4h': '4h', '45min': '45m', '15min': '15m', '5min': '5m'}
    timeframe = timeframe_map.get(intervalo, '1d')
    
    try:
        ohlcv = exchange.fetch_ohlcv(simbolo_ccxt, timeframe=timeframe, limit=limite)
        if not ohlcv:
            return pd.DataFrame()
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        st.error(f"Error Binance: {e}")
        return pd.DataFrame()

def obtener_datos_accion(simbolo, periodo, intervalo):
    ticker = yf.Ticker(simbolo)
    yf_interval_map = {'1d': '1d', '4h': '1h', '45min': '30m', '15min': '15m', '5min': '5m'}
    intervalo_yf = yf_interval_map.get(intervalo, '1d')
    
    if intervalo in ['4h', '45min', '15min', '5min']:
        if periodo in ['1y', '2y']:
            periodo = '1mo'
        df = ticker.history(period=periodo, interval=intervalo_yf)
    else:
        df = ticker.history(period=periodo, interval=intervalo_yf)
    return df, ticker

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    st.header(" Configuración")
    
    with st.form(key="analisis_form"):
        simbolo = st.text_input("Símbolo", value="BTC-USD")
        periodo = st.selectbox("Período (solo acciones)", ["1mo","3mo","6mo","1y","2y"], index=3)
        intervalo = st.selectbox("Intervalo", ["1d", "4h", "45min", "15min", "5min"], index=0)
        
        st.markdown("---")
        modo_analisis = st.radio(
            " Modo de análisis:",
            ["AndyStopLoss", "Completo (Técnico+Chartismo+Fundamental)"],
            index=0
        )
        
        analizar_btn = st.form_submit_button(" ANALIZAR", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.header(" Gestión de Riesgo")
    sl_pct = st.slider("Stop Loss %", 1.0, 30.0, 16.0, 0.5)
    tp1_pct = st.slider("Take Profit 1 %", 5.0, 60.0, 20.0, 1.0)
    tp2_pct = st.slider("Take Profit 2 %", 10.0, 100.0, 40.0, 1.0)
    
    st.header(" Posición")
    capital_total = st.number_input("Capital total (USD)", 10, 1000000, 10000, 500)
    riesgo_cap_pct = st.slider("Riesgo % del capital", 0.5, 10.0, 2.0, 0.5)
    
    if modo_analisis == "AndyStopLoss":
        usar_soporte = st.checkbox(" Entrar en SOPORTE (sugerido)", value=True)
        st.caption(" Reglas AndyStopLoss")
        st.caption("Entrada: SOPORTE o RSI<30 | SL: pierde ASL21/SMA30")
    else:
        usar_soporte = False
        st.caption(" Modo Completo")
        st.caption("Técnico (EMA, MACD, RSI, Bollinger, Estocástico) + Chartismo + Fundamental")

# ------------------------------------------------------------
# PROCESAR ANÁLISIS
# ------------------------------------------------------------
if analizar_btn or (st.session_state.datos is not None and simbolo != st.session_state.simbolo):
    with st.spinner(f'Analizando {simbolo} con modo {modo_analisis[:15]}...'):
        try:
            es_cripto = simbolo.endswith('-USD')
            
            if es_cripto:
                simbolo_binance = simbolo.replace('-USD', '/USDT')
                df = obtener_datos_cripto(simbolo_binance, intervalo, limite=500)
                if df.empty:
                    st.error("No se obtuvieron datos de Binance")
                    st.stop()
                fuente = f"Binance (~100ms) - {intervalo}"
                info = None
            else:
                df, ticker = obtener_datos_accion(simbolo, periodo, intervalo)
                if df.empty:
                    st.error(f"No se obtuvieron datos de {simbolo}")
                    st.stop()
                info = ticker.info
                fuente = f"Yahoo Finance - {periodo}, {intervalo}"
            
            if len(df) < 30:
                st.warning(f"Solo {len(df)} velas disponibles.")
            
            df = calcular_indicadores(df)
            
            if modo_analisis == "AndyStopLoss":
                analisis = analisis_andystoploss(df)
                chartismo = None
                fundamental = None
                
                if usar_soporte and analisis.get('soportes') and analisis['soportes']:
                    precio_entrada_sugerido = analisis['soportes'][0]
                    razon_entrada = f"Soporte detectado en ${precio_entrada_sugerido:,.2f}"
                else:
                    precio_entrada_sugerido = analisis['precio']
                    razon_entrada = "Precio actual"
                
            else:
                analisis_tecnico = analisis_tecnico_completo(df)
                chartismo = analisis_chartismo(df)
                fundamental = analisis_fundamental(simbolo, info)
                
                punt_total = analisis_tecnico['puntuacion'] * 0.5 + chartismo['puntuacion'] * 0.3
                if fundamental:
                    punt_total += fundamental['puntuacion'] * 0.2
                
                if punt_total >= 65:
                    decision, color = " COMPRAR", "green"
                elif punt_total >= 45:
                    decision, color = " DUDAR", "orange"
                else:
                    decision, color = " NO COMPRAR", "red"
                
                analisis = {
                    'puntuacion': punt_total,
                    'decision': decision,
                    'color': color,
                    'precio': analisis_tecnico['precio'],
                    'rsi': analisis_tecnico['rsi'],
                    'sen_ent': analisis_tecnico['señales'][:4],
                    'sen_sal': chartismo['figuras'][:2] if chartismo['figuras'] else [],
                    'soportes': chartismo['soportes'],
                    'resistencias': chartismo['resistencias'],
                    'asl21': df['ASL21'].iloc[-1],
                    'sma30': df['SMA_30'].iloc[-1],
                    'fundamental_punt': fundamental['puntuacion'],
                    'tecnico_punt': analisis_tecnico['puntuacion'],
                    'chartismo_punt': chartismo['puntuacion'],
                    'explicaciones': analisis_tecnico['señales'][:4],
                    'explicaciones_fund': fundamental['explicaciones']
                }
                
                precio_entrada_sugerido = analisis['precio']
                razon_entrada = "Precio actual según análisis técnico"
            
            st.session_state.datos = df
            st.session_state.analisis = analisis
            st.session_state.simbolo = simbolo
            st.session_state.modo_analisis = modo_analisis
            st.session_state.info = info
            st.session_state.precio_entrada_sugerido = precio_entrada_sugerido
            st.session_state.razon_entrada = razon_entrada
            st.session_state.sl_pct = sl_pct
            st.session_state.tp1_pct = tp1_pct
            st.session_state.tp2_pct = tp2_pct
            st.session_state.capital_total = capital_total
            st.session_state.riesgo_cap_pct = riesgo_cap_pct
            
            st.success(f" {simbolo.upper()} - {len(df)} velas - {fuente} - Modo: {modo_analisis}")
            
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

# ------------------------------------------------------------
# MOSTRAR RESULTADOS
# ------------------------------------------------------------
if st.session_state.analisis is not None:
    a = st.session_state.analisis
    df = st.session_state.datos
    modo = st.session_state.modo_analisis
    
    precio_entrada = st.session_state.precio_entrada_sugerido
    sl_pct = st.session_state.sl_pct
    tp1_pct = st.session_state.tp1_pct
    tp2_pct = st.session_state.tp2_pct
    capital_total = st.session_state.capital_total
    riesgo_cap_pct = st.session_state.riesgo_cap_pct
    
    stop_loss = precio_entrada * (1 - sl_pct/100)
    tp1 = precio_entrada * (1 + tp1_pct/100)
    tp2 = precio_entrada * (1 + tp2_pct/100)
    
    perdida_max = capital_total * riesgo_cap_pct / 100
    sl_decimal = sl_pct / 100
    capital_invertir = min(perdida_max / sl_decimal if sl_decimal > 0 else 0, capital_total)
    cantidad = capital_invertir / precio_entrada if precio_entrada > 0 else 0
    perdida_esperada = cantidad * (precio_entrada - stop_loss)
    ganancia_tp1 = cantidad * (tp1 - precio_entrada)
    ganancia_tp2 = cantidad * (tp2 - precio_entrada)
    riesgo_real = (perdida_esperada / capital_total) * 100 if capital_total > 0 else 0
    reward_risk = (tp1 - precio_entrada) / (precio_entrada - stop_loss) if (precio_entrada - stop_loss) > 0 else 0
    
    # Tarjeta de decisión
    st.markdown(f"""
    <div style='background-color:{a['color']}20; padding:6px; border-radius:12px; margin-bottom:10px; text-align:center'>
        <h2 style='margin:0; color:{a['color']}'>{a['decision']}</h2>
        <p style='margin:0; font-size:0.8rem'>
            Modo: {modo} | Velas: {len(df)}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # SECCIÓN DE ALERTA PARA TRADINGVIEW
    st.markdown("---")
    st.subheader(" CONFIGURAR ALERTA EN TRADINGVIEW")
    
    col_alerta1, col_alerta2 = st.columns([1, 1])
    with col_alerta1:
        st.markdown(f"""
        <div style='background-color:#1e3a5f; padding:10px; border-radius:10px; margin:5px 0'>
            <h4 style='margin:0'> PRECIO DE ENTRADA</h4>
            <p style='font-size:24px; margin:5px 0'><b>${precio_entrada:,.2f}</b></p>
            <p style='margin:0; font-size:12px'>{st.session_state.razon_entrada}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_alerta2:
        st.markdown(f"""
        <div style='background-color:#2d2d2d; padding:10px; border-radius:10px; margin:5px 0'>
            <h4 style='margin:0'> STOP LOSS</h4>
            <p style='font-size:20px; margin:5px 0'><b>${stop_loss:,.2f}</b></p>
            <p style='margin:0; font-size:12px'>Pérdida: -${abs(perdida_esperada):,.0f} ({sl_pct:.0f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("###  Cómo configurar la alerta en TradingView:")
    
    mensaje_alerta = f" ALERTA {st.session_state.simbolo.upper()}! Precio alcanzó ${precio_entrada:,.2f} | Stop Loss: ${stop_loss:,.2f} ({sl_pct:.0f}%) | Take Profit 1: ${tp1:,.2f} ({tp1_pct:.0f}%) | Take Profit 2: ${tp2:,.2f} ({tp2_pct:.0f}%) | Invertir: ${capital_invertir:,.0f} ({riesgo_cap_pct:.0f}% del capital)"
    
    st.code(mensaje_alerta, language="text")
    
    # Métricas
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric(" Precio Actual", f"${a['precio']:,.2f}")
    with col2: st.metric(" RSI", f"{a['rsi']:.0f}")
    with col3: st.metric(" ASL21", f"${a.get('asl21', 0):,.2f}")
    with col4: st.metric(" Entrada Objetivo", f"${precio_entrada:,.2f}")
    
    # Señales
    col_ent, col_sal = st.columns(2)
    with col_ent:
        st.caption(" **Señales Positivas**")
        for s in a.get('sen_ent', ['No hay señales'])[:3]:
            st.write(f"- {s}")
    with col_sal:
        st.caption(" **Señales Negativas**")
        for s in a.get('sen_sal', ['No hay señales'])[:2]:
            st.write(f"- {s}")
    
    # Gráficos
    st.markdown("---")
    tab1, tab2 = st.tabs([" Gráfico Principal", " RSI"])
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Velas'))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_9'], name='EMA 9', line=dict(color='orange', width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df['ASL21'], name='ASL21', line=dict(color='purple', width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_30'], name='SMA30', line=dict(color='red', dash='dash')))
        fig.add_hline(y=precio_entrada, line_dash="dash", line_color="green", annotation_text=f"ENTRADA ${precio_entrada:,.2f}")
        fig.add_hline(y=stop_loss, line_dash="dash", line_color="red", annotation_text=f"STOP ${stop_loss:,.2f}")
        fig.update_layout(height=480, margin=dict(l=0,r=0,t=30,b=0), template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple', width=2)))
        fig2.add_trace(go.Scatter(x=df.index, y=df['RSI_ASL21'], name='RSI ASL21', line=dict(color='orange', width=1, dash='dash')))
        fig2.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Sobrecompra (70)")
        fig2.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Sobreventa (30)")
        fig2.update_layout(height=400, margin=dict(l=0,r=0,t=30,b=0), template='plotly_dark')
        st.plotly_chart(fig2, use_container_width=True)