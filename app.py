import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import os

def limpiar_pantalla():
    os.system('cls' if os.name == 'nt' else 'clear')

def obtener_datos(simbolo):
    """Descarga los datos del activo"""
    try:
        # Descargar datos históricos (últimos 6 meses)
        fin = datetime.now()
        inicio = fin - timedelta(days=180)
        
        ticker = yf.Ticker(simbolo)
        datos = ticker.history(start=inicio, end=fin)
        
        if datos.empty:
            return None, None, None
        
        # Obtener información fundamental
        info = ticker.info
        
        # Datos actuales
        precio_actual = datos['Close'].iloc[-1]
        volumen_actual = datos['Volume'].iloc[-1]
        
        return datos, info, precio_actual, volumen_actual
    
    except Exception as e:
        print(f"Error al obtener datos: {e}")
        return None, None, None, None

def calcular_tecnicos(datos, volumen_actual):
    """Calcula los indicadores técnicos según criterios óptimos"""
    
    # Calcular VWAP (aproximado)
    datos['Typical_Price'] = (datos['High'] + datos['Low'] + datos['Close']) / 3
    datos['Cumulative_TPV'] = (datos['Typical_Price'] * datos['Volume']).cumsum()
    datos['Cumulative_Volume'] = datos['Volume'].cumsum()
    datos['VWAP'] = datos['Cumulative_TPV'] / datos['Cumulative_Volume']
    
    vwap_actual = datos['VWAP'].iloc[-1]
    precio_actual = datos['Close'].iloc[-1]
    
    # Calcular ROC (Rate of Change - 9 períodos)
    roc_periodos = 9
    datos['ROC'] = ((datos['Close'] - datos['Close'].shift(roc_periodos)) / datos['Close'].shift(roc_periodos)) * 100
    roc_actual = datos['ROC'].iloc[-1]
    
    # Calcular WMA (Weighted Moving Average - 20 períodos)
    periodo_wma = 20
    pesos = list(range(1, periodo_wma + 1))
    suma_pesos = sum(pesos)
    
    def calcular_wma(serie):
        if len(serie) < periodo_wma:
            return None
        return sum(serie.iloc[-periodo_wma:] * pesos) / suma_pesos
    
    datos['WMA'] = datos['Close'].rolling(window=periodo_wma).apply(calcular_wma, raw=False)
    wma_actual = datos['WMA'].iloc[-1]
    
    # Calcular volumen promedio
    volumen_promedio = datos['Volume'].tail(20).mean()
    volumen_ratio = volumen_actual / volumen_promedio if volumen_promedio > 0 else 0
    
    # Análisis de tendencia (máximos y mínimos crecientes)
    ultimos_10_maximos = datos['High'].tail(10).max()
    ultimos_20_maximos = datos['High'].tail(20).max()
    ultimos_10_minimos = datos['Low'].tail(10).min()
    ultimos_20_minimos = datos['Low'].tail(20).min()
    
    maxima_creciente = ultimos_10_maximos > ultimos_20_maximos
    minima_creciente = ultimos_10_minimos > ultimos_20_minimos
    
    # Puntuación técnica (0-100)
    puntuacion = 0
    
    # Criterio VWAP (máxima prioridad)
    if precio_actual > vwap_actual:
        puntuacion += 35
        vwap_senal = "ALCISTA ✓"
    else:
        vwap_senal = "BEARISTA ✗"
    
    # Criterio ROC
    if roc_actual > 0:
        puntuacion += 30
        roc_senal = "POSITIVO ✓"
    elif roc_actual < -5:
        roc_senal = "NEGATIVO ✗"
    else:
        puntuacion += 15
        roc_senal = "NEUTRAL ~"
    
    # Criterio WMA
    if precio_actual > wma_actual:
        puntuacion += 20
        wma_senal = "ALCISTA ✓"
    else:
        wma_senal = "BEARISTA ✗"
    
    # Volumen (confirmación)
    if volumen_ratio > 1.2:
        puntuacion += 10
        volumen_senal = "FUERTE ✓"
    elif volumen_ratio < 0.8:
        volumen_senal = "DÉBIL ✗"
    else:
        puntuacion += 5
        volumen_senal = "NORMAL ~"
    
    # Estructura de tendencia
    if maxima_creciente and minima_creciente:
        puntuacion += 5
        estructura_senal = " ALCISTA ✓"
    elif maxima_creciente or minima_creciente:
        estructura_senal = " NEUTRAL ~"
    else:
        estructura_senal = " BEARISTA ✗"
    
    return {
        'puntuacion': puntuacion,
        'vwap': {'valor': round(vwap_actual, 2), 'senal': vwap_senal},
        'roc': {'valor': round(roc_actual, 2), 'senal': roc_senal},
        'wma': {'valor': round(wma_actual, 2), 'senal': wma_senal},
        'volumen': {'ratio': round(volumen_ratio, 2), 'senal': volumen_senal},
        'estructura': estructura_senal,
        'precio_actual': round(precio_actual, 2)
    }

def calcular_fundamental(info):
    """Calcula análisis fundamental según criterios óptimos"""
    
    # Extraer datos con manejo de errores
    per = info.get('trailingPE', None)
    peg = info.get('pegRatio', None)
    roe = info.get('returnOnEquity', None)
    deuda_equity = info.get('debtToEquity', None)
    sector = info.get('sector', 'Desconocido')
    
    # Si porcentajes, convertir a número
    if roe and roe > 1:
        roe = roe * 100
    
    puntuacion = 0
    explicaciones = []
    
    # 1. Criterio PER (Precio/Ganancias)
    if per:
        if per < 15:
            puntuacion += 30
            per_senal = f"✓ BUENO ({per:.2f} es bajo)"
            explicaciones.append(f"• PER de {per:.2f}: Indicador de infravaloración (óptimo <15)")
        elif per < 25:
            puntuacion += 15
            per_senal = f"~ ACEPTABLE ({per:.2f})"
            explicaciones.append(f"• PER de {per:.2f}: Valor razonable")
        else:
            per_senal = f"✗ CARO ({per:.2f} es alto)"
            explicaciones.append(f"• PER de {per:.2f}: Podría estar sobrevalorada")
    else:
        per_senal = "? DATOS NO DISPONIBLES"
        explicaciones.append("• PER: No disponible para este activo")
    
    # 2. Criterio PEG (ajusta PER por crecimiento)
    if peg:
        if peg < 1:
            puntuacion += 30
            peg_senal = f"✓ EXCELENTE ({peg:.2f})"
            explicaciones.append(f"• PEG de {peg:.2f}: Crecimiento justifica el precio")
        elif peg < 1.5:
            puntuacion += 15
            peg_senal = f"~ OK ({peg:.2f})"
            explicaciones.append(f"• PEG de {peg:.2f}: Valor aceptable")
        else:
            peg_senal = f"✗ CARO ({peg:.2f})"
            explicaciones.append(f"• PEG de {peg:.2f}: Podría estar sobrevalorada")
    else:
        peg_senal = "? DATOS NO DISPONIBLES"
    
    # 3. Criterio ROE (Rentabilidad)
    if roe:
        if roe > 15:
            puntuacion += 25
            roe_senal = f"✓ EXCELENTE ({roe:.1f}%)"
            explicaciones.append(f"• ROE de {roe:.1f}%: Empresa muy rentable")
        elif roe > 10:
            puntuacion += 15
            roe_senal = f"~ ACEPTABLE ({roe:.1f}%)"
            explicaciones.append(f"• ROE de {roe:.1f}%: Rentabilidad decente")
        else:
            roe_senal = f"✗ BAJA ({roe:.1f}%)"
            explicaciones.append(f"• ROE de {roe:.1f}%: Rentabilidad mejorable")
    else:
        roe_senal = "? DATOS NO DISPONIBLES"
    
    # 4. Criterio Deuda/Patrimonio
    if deuda_equity:
        if deuda_equity < 80:
            puntuacion += 15
            deuda_senal = f"✓ BAJA ({deuda_equity:.1f}%)"
            explicaciones.append(f"• Deuda/Patrimonio de {deuda_equity:.1f}%: Riesgo controlado")
        elif deuda_equity < 150:
            puntuacion += 8
            deuda_senal = f"~ MODERADA ({deuda_equity:.1f}%)"
            explicaciones.append(f"• Deuda/Patrimonio de {deuda_equity:.1f}%: Riesgo manejable")
        else:
            deuda_senal = f"✗ ALTA ({deuda_equity:.1f}%)"
            explicaciones.append(f"• Deuda/Patrimonio de {deuda_equity:.1f}%: Riesgo elevado")
    else:
        deuda_senal = "? DATOS NO DISPONIBLES"
    
    return {
        'puntuacion': puntuacion,
        'per': per_senal,
        'peg': peg_senal,
        'roe': roe_senal,
        'deuda': deuda_senal,
        'explicaciones': explicaciones,
        'sector': sector
    }

def generar_recomendacion(puntuacion_tecnica, puntuacion_fundamental):
    """Genera la recomendación final"""
    
    puntuacion_total = (puntuacion_tecnica + puntuacion_fundamental) / 2
    
    if puntuacion_total >= 70:
        if puntuacion_tecnica >= 70 and puntuacion_fundamental >= 70:
            return "🟢 COMPRA FUERTE", f"Puntuación Total: {puntuacion_total:.1f} - Ambos análisis son excelentes"
        elif puntuacion_tecnica >= 70:
            return "🟡 COMPRA (Táctica)", f"Puntuación Total: {puntuacion_total:.1f} - Técnico fuerte, Fundamental OK"
        else:
            return "🟡 COMPRA (Valor)", f"Puntuación Total: {puntuacion_total:.1f} - Fundamental fuerte, Técnico aceptable"
    
    elif puntuacion_total >= 50:
        return "🟠 DUDAR / ESPERAR", f"Puntuación Total: {puntuacion_total:.1f} - Mejor esperar confirmación"
    
    else:
        return "🔴 NO COMPRAR", f"Puntuación Total: {puntuacion_total:.1f} - Señales negativas"

def mostrar_resultados(simbolo, tecnicos, fundamental):
    """Muestra los resultados en formato amigable"""
    
    limpiar_pantalla()
    print("=" * 70)
    print(f"📊 ANÁLISIS DE {simbolo.upper()} - {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print("=" * 70)
    
    print("\n📈 ANÁLISIS TÉCNICO (Trading Intradía/Semanal)")
    print("-" * 50)
    print(f"💰 Precio Actual: ${tecnicos['precio_actual']}")
    print(f"📊 Puntuación Técnica: {tecnicos['puntuacion']}/100")
    print(f"\nIndicadores:")
    print(f"  • VWAP: {tecnicos['vwap']['valor']} - {tecnicos['vwap']['senal']}")
    print(f"  • ROC: {tecnicos['roc']['valor']}% - {tecnicos['roc']['senal']}")
    print(f"  • WMA: {tecnicos['wma']['valor']} - {tecnicos['wma']['senal']}")
    print(f"  • Volumen: Ratio {tecnicos['volumen']['ratio']} - {tecnicos['volumen']['senal']}")
    print(f"  • Estructura de Tendencia: {tecnicos['estructura']}")
    
    print("\n📊 ANÁLISIS FUNDAMENTAL (Valor de la Empresa)")
    print("-" * 50)
    print(f"🏭 Sector: {fundamental['sector']}")
    print(f"📊 Puntuación Fundamental: {fundamental['puntuacion']}/100")
    print(f"\nIndicadores:")
    print(f"  • PER (Precio/Ganancias): {fundamental['per']}")
    print(f"  • PEG (PER ajustado por crecimiento): {fundamental['peg']}")
    print(f"  • ROE (Rentabilidad): {fundamental['roe']}")
    print(f"  • Deuda/Patrimonio: {fundamental['deuda']}")
    
    print("\n💡 EXPLICACIÓN BREVE:")
    print("-" * 50)
    for exp in fundamental['explicaciones'][:3]:  # Mostrar 3 más importantes
        print(exp)
    
    # Recomendación final
    print("\n" + "=" * 70)
    recomendacion, detalle = generar_recomendacion(tecnicos['puntuacion'], fundamental['puntuacion'])
    print(f"🎯 CONCLUSIÓN: {recomendacion}")
    print(f"📝 {detalle}")
    print("=" * 70)
    
    print("\n⚠️ NOTA: Esto es una herramienta de análisis, no es asesoría financiera.")
    print("   Siempre investiga por tu cuenta antes de invertir.")

def main():
    """Función principal"""
    limpiar_pantalla()
    print("=" * 70)
    print("🏦 APP DE ANÁLISIS FINANCIERO - TRADING E INVERSIÓN")
    print("=" * 70)
    print("\n📌 Ejemplos de símbolos: KO (Coca-Cola), AAPL (Apple), MSFT (Microsoft)")
    print("   TSLA (Tesla), AMZN (Amazon), GOOGL (Google)")
    print("\n" + "-" * 70)
    
    while True:
        simbolo = input("\n🔍 ¿Qué activo quieres analizar? (Ej: KO): ").upper().strip()
        
        if not simbolo:
            print("❌ Por favor ingresa un símbolo válido")
            continue
        
        print(f"\n⏳ Analizando {simbolo}... Esto puede tomar unos segundos")
        
        # Obtener datos
        datos, info, precio_actual, volumen_actual = obtener_datos(simbolo)
        
        if datos is None:
            print(f"❌ No se pudo obtener datos de {simbolo}. Verifica el símbolo.")
            print("   Consejo: Usa símbolos como KO, AAPL, MSFT, etc.")
            continue
        
        if len(datos) < 30:
            print(f"⚠️ Pocos datos históricos para {simbolo}. Intenta con otro símbolo.")
            continue
        
        # Calcular análisis
        tecnicos = calcular_tecnicos(datos, volumen_actual)
        fundamental = calcular_fundamental(info)
        
        # Mostrar resultados
        mostrar_resultados(simbolo, tecnicos, fundamental)
        
        # Preguntar si quiere analizar otro
        print("\n" + "-" * 70)
        otra = input("\n¿Analizar otro activo? (s/n): ").lower()
        if otra != 's':
            print("\n👋 ¡Gracias por usar la app! Hasta luego.")
            break

if __name__ == "__main__":
    main()