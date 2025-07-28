# Predictor Dow Jones - Modelo XGBoost + Trading en Vivo

Predicción del porcentaje de cambio del Dow Jones entre las 20:00 y 23:00 usando XGBoost + **Sistema de Trading Automatizado**.

## Descripción

Este proyecto implementa un **sistema completo de trading automatizado** que:

1. **Entrena un modelo de machine learning** optimizado para predecir cambios del Dow Jones
2. **Se conecta a brokers** (MetaTrader 5 o simulador) para obtener datos en tiempo real
3. **Ejecuta operaciones automáticamente** basadas en las predicciones del modelo
4. **Gestiona riesgo** con stop loss, take profit y control de posiciones
5. **Monitorea performance** con métricas avanzadas y drawdown intraoperación

### 🎯 **Componentes del Sistema:**

- **Backtesting**: Entrenar y validar el modelo con datos históricos
- **Trading en Vivo**: Ejecutar operaciones reales usando el modelo entrenado
- **Broker Simulado**: Para pruebas seguras sin riesgo
- **MetaTrader 5**: Para trading real (Windows solamente)

## Características del Modelo

- **Algoritmo**: Ensemble Optimizado (XGBoost + Random Forest + Gradient Boosting + ElasticNet)
- **Optimización**: GridSearchCV con TimeSeriesSplit para hiperparámetros
- **Target**: Porcentaje de cambio entre precio a las 20:00 y 23:00
- **Features**: Indicadores técnicos, contexto diario y semanal
- **Evaluación**: MAE, R², backtesting con capital real ($100,000)
- **Sistema de Trading**: Cada 0.1% de movimiento = $500 (apalancamiento 5x)

## Funcionalidades Avanzadas

### 🤖 **Optimización Automática del Modelo**

- **GridSearchCV**: Búsqueda automática de mejores hiperparámetros
- **TimeSeriesSplit**: Validación cruzada temporal para series financieras
- **Ensemble Inteligente**: Combina 4 algoritmos diferentes con pesos optimizados
- **Selección Automática**: Elige el mejor modelo (individual vs ensemble)

### 💰 **Sistema de Trading Realista**

- **Capital Inicial**: $100,000
- **Tamaño de Posición**: Cada 0.1% de movimiento = $500
- **Apalancamiento Efectivo**: 5x
- **P&L en Dólares**: Todas las métricas calculadas en capital real

### 📊 **Métricas Profesionales**

- **Sharpe Ratio**: Rendimiento ajustado por riesgo
- **Calmar Ratio**: Rentabilidad vs Drawdown máximo
- **Profit Factor**: Relación ganancia bruta/pérdida bruta
- **Expectancy**: Ganancia esperada por trade en dólares
- **Recovery Factor**: Capacidad de recuperación
- **Rachas de Wins/Losses**: Análisis de consistencia
- **Drawdown Intraoperación**: Máximo riesgo durante cada trade (20:00-23:00)

### 🔍 **Análisis de Riesgo Intraoperación**

- **Drawdown por Trade**: Máximo movimiento en contra durante cada operación
- **Timeline de Riesgo**: Hora exacta del máximo drawdown
- **Distribución de DD**: Estadísticas completas de drawdowns intraoperación
- **Porcentaje de Trades con DD**: Qué porcentaje de operaciones experimentan drawdown

## 🤖 Trading en Vivo Automatizado

### **🚀 Características del Sistema de Trading:**

- **Conexión a Brokers**: MetaTrader 5 (Windows) o Simulador (multiplataforma)
- **Trading Automático**: Operaciones ejecutadas sin intervención manual
- **Gestión de Riesgo**: Stop Loss, Take Profit, tamaño de posición dinámico
- **Horarios Específicos**: Solo opera entre 20:00-23:00 días laborables
- **Monitoreo Continuo**: Logs detallados y métricas en tiempo real

### **📊 Flujo de Trading Automatizado:**

1. **20:00 UTC+3**: El sistema analiza datos históricos del día
2. **Predicción**: Genera predicción usando el modelo entrenado
3. **Evaluación**: Verifica confianza mínima y límites de trading
4. **Ejecución**: Abre posición con SL/TP si cumple criterios
5. **Monitoreo**: Supervisa la posición durante 3 horas
6. **23:00 UTC+3**: Cierra automáticamente la posición

### **⚙️ Configuración del Sistema:**

```json
{
  "broker_type": "simulator", // o "mt5"
  "symbol": "US30",
  "volume": 0.1,
  "max_positions": 1,
  "stop_loss_pct": 2.0,
  "take_profit_pct": 3.0,
  "timezone": "Europe/Moscow", // UTC+3
  "risk_per_trade": 1.0, // % del capital por trade
  "min_confidence": 0.5 // Confianza mínima para abrir posición
}
```

### **🔧 Comandos de Trading en Vivo:**

#### **Entrenar Modelo y Iniciar Trading:**

```bash
# 1. Entrenar modelo con datos históricos
uv run python run_predictor.py tu_archivo.csv

# 2. Crear configuración (opcional)
uv run python run_live_trading.py --create-config --timezone Europe/Moscow

# 3. Iniciar trading simulado
uv run python run_live_trading.py --broker simulator

# 4. Iniciar trading real (solo Windows con MT5)
uv run python run_live_trading.py --broker mt5
```

#### **Configuración MetaTrader 5 (Windows):**

```bash
# Verificar y configurar MT5
python setup_mt5.py

# Trading en vivo con MT5
uv run python run_live_trading.py --broker mt5 --config trading_config_mt5.json
```

#### **Modo de Entrenamiento Automático:**

```bash
# Entrenar modelo y empezar trading en un comando
uv run python run_live_trading.py --broker simulator --train-first
```

### **📱 Monitoreo del Sistema:**

```bash
# Ver estado del sistema
python run_live_trading.py status

# Logs en tiempo real
tail -f live_trading.log

# Verificar archivos generados
ls -la *.pkl *.json *.log *.csv
```

### **🛡️ Gestión de Riesgo Implementada:**

- **Stop Loss**: 2% por defecto (configurable)
- **Take Profit**: 3% por defecto (configurable)
- **Tamaño dinámico**: Calculado según % de riesgo del capital
- **Límites diarios**: Máximo 5 trades por día
- **Máximo posiciones**: 1 posición simultánea
- **Horarios restringidos**: Solo días laborables 20:00-23:00 UTC+3

### **📈 Métricas en Tiempo Real:**

- Balance y equity actual
- P&L diario y acumulado
- Número de trades ejecutados
- Win rate del día
- Drawdown máximo de posiciones abiertas

## Instalación

El proyecto utiliza `uv` como administrador de dependencias:

```bash
# El proyecto ya está configurado con uv
# Las dependencias están instaladas: xgboost, pandas, scikit-learn, ta, matplotlib, numpy
```

## Uso

### Formato de Datos Requerido

El CSV debe tener el siguiente formato:

```
Date,Time,Open,High,Low,Close,Volume
20130930,17:34:00,15110.760,15110.760,15110.760,15110.760,2000
20130930,17:35:00,15104.688,15104.688,15104.688,15104.688,2000
...
```

### Ejecución

```bash
# Método 1: Script optimizado con opciones
uv run python run_predictor.py datos_dow_jones.csv

# Método 2: Modo rápido (automáticamente limita a 50k muestras)
uv run python run_predictor.py datos_dow_jones.csv --fast

# Método 3: Configuración personalizada
uv run python run_predictor.py datos_dow_jones.csv --max-samples 100000 --chunk-size 5000

# Método 4: Uso programático optimizado
uv run python -c "
from src.predictor_dj.main import main_optimized
predictor, results = main_optimized('datos_dow_jones.csv', max_samples=50000)
"
```

## Optimizaciones de Rendimiento

### 🚀 Optimizaciones Implementadas

- **Tipos de datos optimizados**: Uso de `float32` e `int8` para reducir memoria en ~50%
- **Carga eficiente**: Especificación de dtypes y formato de fecha al leer CSV
- **Procesamiento en chunks**: Para datasets grandes (>500MB) procesamiento automático en chunks
- **Cálculos vectorizados**: Operaciones pandas optimizadas y agregaciones eficientes
- **Filtrado temprano**: Eliminación de datos fuera de horarios de trading desde el inicio
- **Indicadores técnicos optimizados**: SMA con pandas rolling, MACD simplificado
- **Limitación automática**: Control de memoria para datasets masivos

### 📊 Rendimiento Esperado

| Tamaño Dataset | Tiempo Estimado | Memoria Pico | Optimización      |
| -------------- | --------------- | ------------ | ----------------- |
| < 100MB        | 2-5 minutos     | < 1GB        | GridSearch 3-fold |
| 100-500MB      | 5-15 minutos    | < 2GB        | GridSearch 3-fold |
| > 500MB        | 15-30 minutos   | < 3GB        | GridSearch 2-fold |

**Nota**: Los tiempos incluyen optimización de hiperparámetros y entrenamiento de ensemble

### ⚙️ Configuración para Datasets Grandes

```python
# Para datasets muy grandes (>1M registros)
predictor = DowJonesPredictor(
    max_samples=100000,  # Limitar muestras
    chunk_size=5000      # Reducir chunk size
)

# Para máximo rendimiento en datasets pequeños
predictor = DowJonesPredictor()  # Sin límites
```

## Funcionalidades

### Procesamiento de Datos

- Carga y limpieza de datos de 1 minuto
- Combinación de fecha y hora
- Filtrado por horarios específicos (20:00-23:00)

### Características (Features)

- **Contexto Diario**: Precios OHLC, volumen, variación porcentual
- **Indicadores Técnicos**: SMA(15,30), RSI(14), MACD
- **Contexto Semanal**: Promedio de precios, volatilidad, tendencia
- **Temporal**: Día de la semana

### Modelo

- **XGBoost Regressor** optimizado para series temporales financieras
- División: 70% entrenamiento, 15% validación, 15% prueba
- Validación cruzada temporal

### Evaluación

- **MAE**: Error absoluto medio
- **R²**: Coeficiente de determinación
- **Backtesting**: Rentabilidad simulada
- **Precisión direccional**: Porcentaje de predicciones correctas de dirección

### Visualización

- Gráfico de rentabilidad acumulada
- Scatter plot de predicciones vs valores reales
- Guardado automático como 'resultados_predictor_dj.png'

## Salida del Programa

El programa genera:

1. **Optimización del modelo** (GridSearch de hiperparámetros)
2. **Ensemble de 4 algoritmos** (XGBoost, Random Forest, Gradient Boosting, ElasticNet)
3. **Métricas avanzadas de evaluación** (MAE, R²)
4. **Backtesting con capital real** ($100,000 inicial)
5. **CSV detallado** con todas las operaciones (`backtest_detallado.csv`)
6. **Gráficos profesionales** (4 paneles de análisis)
7. **Métricas de performance** (Sharpe, Calmar, Profit Factor, etc.)

### 📄 **Contenido del CSV Detallado:**

- Fecha y precio de cada operación
- Predicción vs resultado real
- Dirección predicha vs actual
- P&L por trade en dólares
- Equity curve completa
- Drawdown del portafolio en tiempo real
- **Drawdown intraoperación**: Máximo DD durante cada trade
- **Hora del máximo DD**: Timestamp exacto del peor momento
- Análisis de wins/losses

### 📈 **Gráficos Generados:**

1. **Equity Curve + Drawdown**: Evolución del capital y drawdown del portafolio
2. **Predicciones vs Reales**: Precisión del modelo con colores por acierto
3. **Distribución P&L**: Histogram de ganancias/pérdidas por trade
4. **Drawdown Intraoperación**: Análisis del riesgo durante cada operación (20:00-23:00)

## Estructura del Proyecto

```
predictor_dj/
├── src/
│   └── predictor_dj/
│       └── main.py          # Código principal del predictor
├── run_predictor.py         # Script de ejecución
├── README.md               # Este archivo
└── pyproject.toml          # Configuración de dependencias
```

## Ejemplo de Uso Completo

```python
from src.predictor_dj.main import DowJonesPredictor

# Crear predictor
predictor = DowJonesPredictor()

# Ejecutar análisis completo
results = predictor.run_complete_analysis('tu_archivo.csv')

# Acceder a resultados
print(f"MAE: {results['evaluation']['mae']}")
print(f"R²: {results['evaluation']['r2']}")
print(f"Rentabilidad: {results['backtest']['total_return']:.2f}%")
```

## Notas Importantes

- Los datos deben tener resolución de 1 minuto
- Se requiere un mínimo de datos históricos para crear características válidas
- El modelo está optimizado para predecir el período específico 20:00-23:00
- Los resultados incluyen tanto predicción de dirección como magnitud del cambio

## 🎯 Ejemplos Prácticos de Uso

### **Escenario 1: Backtesting y Análisis**

```bash
# Entrenar modelo y generar backtesting completo
uv run python run_predictor.py data/datos_dow_jones.csv

# Archivos generados:
# - backtest_detallado.csv (todas las operaciones)
# - resultados_predictor_dj_completo.png (gráficos)
# - trained_model.pkl (modelo para trading en vivo)
```

### **Escenario 2: Trading Simulado**

```bash
# Iniciar trading simulado con entrenamiento automático
uv run python run_live_trading.py --broker simulator --train-first

# El sistema:
# 1. Entrena el modelo con tus datos
# 2. Inicia trading simulado con $100,000
# 3. Opera automáticamente entre 20:00-23:00
# 4. Genera logs detallados en live_trading.log
```

### **Escenario 3: Trading Real con MT5 (Windows)**

```bash
# 1. Configurar MT5
python setup_mt5.py

# 2. Entrenar modelo
uv run python run_predictor.py data/datos_dow_jones.csv

# 3. Iniciar trading real
uv run python run_live_trading.py --broker mt5 --config trading_config_mt5.json
```

### **Escenario 4: Desarrollo y Testing**

```bash
# Crear configuración personalizada
uv run python run_live_trading.py --create-config --broker simulator --balance 50000

# Ejecutar modo rápido para testing
uv run python run_predictor.py datos.csv --fast

# Monitorear sistema en tiempo real
python run_live_trading.py status
tail -f live_trading.log
```

## 📋 Flujo de Trabajo Completo

### **Para Trading en Vivo:**

1. **Preparación de Datos**: Obtener datos históricos del Dow Jones con resolución de 1 minuto
2. **Backtesting**: Ejecutar `run_predictor.py` para entrenar y validar el modelo
3. **Configuración**: Crear configuración de trading con `--create-config`
4. **Testing**: Probar con el simulador primero
5. **Producción**: Usar MT5 real solo después de validar con simulador

### **Archivos Importantes:**

- `trained_model.pkl`: Modelo entrenado para trading en vivo
- `trading_config.json`: Configuración del sistema de trading
- `backtest_detallado.csv`: Historial completo de operaciones
- `live_trading.log`: Logs del sistema en tiempo real

### **Métricas a Monitorear:**

- **Rentabilidad anualizada** > 20%
- **Sharpe Ratio** > 1.0
- **Máximo Drawdown** < 15%
- **Win Rate** > 60%
- **Profit Factor** > 1.5

## ⚠️ Advertencias Importantes

- **Trading Real**: MT5 ejecuta operaciones con dinero real. Use el simulador para pruebas
- **Horarios**: El sistema solo opera entre 20:00-23:00 UTC+3 (días laborables)
- **Riesgo**: Cada trade arriesga máximo 1% del capital (configurable)
- **Supervisión**: Aunque es automatizado, requiere supervisión periódica
- **Internet**: Requiere conexión estable para funcionamiento correcto

## 🆘 Solución de Problemas

### **Error: "Modelo no encontrado"**

```bash
# Solución: Entrenar modelo primero
uv run python run_predictor.py tu_archivo.csv
```

### **Error: "No se pudo conectar al broker"**

```bash
# Para MT5: Verificar que esté abierto y logueado
python setup_mt5.py

# Para simulador: Verificar configuración
python run_live_trading.py status
```

### **Trading no ejecuta operaciones**

- Verificar horario de trading (20:00-23:00 UTC+3)
- Revisar confianza mínima en configuración
- Verificar límites diarios de trades
- Consultar logs: `tail -f live_trading.log`

### **Configurar zona horaria diferente**

```bash
# Opciones de zona horaria disponibles:
uv run python run_live_trading.py --create-config --timezone Europe/Moscow  # UTC+3
uv run python run_live_trading.py --create-config --timezone US/Eastern     # UTC-5/-4
uv run python run_live_trading.py --create-config --timezone Europe/London  # UTC+0/+1
uv run python run_live_trading.py --create-config --timezone Asia/Tokyo     # UTC+9
```

---

¡El sistema está listo para usar! 🚀 Comience con el simulador para familiarizarse antes de pasar a trading real.
