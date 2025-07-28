# Predicción del Movimiento y Porcentaje de Cambio en el Dow Jones (20:00 - 23:00) con XGBoost

Este proyecto tiene como objetivo predecir tanto la **dirección** (sube o baja) como el **porcentaje exacto de cambio** en el **Dow Jones** durante las horas específicas de **20:00 a 23:00**, basándose en el comportamiento histórico del día actual y de la semana hasta ese momento. Se utilizará un modelo de **XGBoost** para realizar una **regresión** y predecir el **porcentaje de cambio** en lugar de solo predecir si el mercado subirá o bajará.

## Descripción del Proyecto

### Objetivo

Predecir el **porcentaje de cambio** entre el precio de **cierre a las 23:00** y el precio de **apertura a las 20:00**, basándose en las características extraídas del comportamiento del mercado durante el día actual y la semana hasta ese momento.

- **Target (Objetivo)**:

  - El **target** será el **porcentaje de cambio** entre el precio de cierre a las 23:00 y el precio de apertura a las 20:00.

  Fórmula del target:

  \[
  \text{Target} = \frac{\text{Precio de cierre a las 23:00} - \text{Precio de apertura a las 20:00}}{\text{Precio de apertura a las 20:00}} \times 100
  \]

  Este valor representará la **subida o bajada** porcentual que se espera entre las 20:00 y las 23:00.

### Datos de Entrada

Los datos utilizados son históricos del **Dow Jones** con una resolución de **1 minuto** desde 2013 hasta la fecha actual. Estos datos incluyen las siguientes columnas:

- `datetime`: Fecha y hora de la observación.
- `open`: Precio de apertura en cada minuto.
- `high`: Precio máximo alcanzado en ese minuto.
- `low`: Precio mínimo alcanzado en ese minuto.
- `close`: Precio de cierre en ese minuto.
- `volume`: Volumen de transacciones en ese minuto.

### Características (Features)

El modelo se alimenta de las siguientes características, calculadas a partir de los datos históricos:

1. **Contexto Diario**:

   - **Precios**: Apertura, cierre, máximo y mínimo del día hasta el momento de la predicción.
   - **Volumen de operaciones** del día.
   - **Indicadores Técnicos**:
     - **Media Móvil (SMA)** de 15 y 30 minutos.
     - **RSI (Relative Strength Index)** de 14 períodos.
     - **MACD (Moving Average Convergence Divergence)**.
   - **Variación porcentual diaria**: Cambio desde la apertura hasta el momento actual.

2. **Contexto Semanal**:

   - **Promedio de precios** de cierre de los días previos de la semana hasta el momento actual.
   - **Tendencia de la semana**: Se determina si la semana ha sido mayormente alcista o bajista.
   - **Volatilidad semanal**: Se calcula la desviación estándar de los precios de los días previos.

3. **Otros**:
   - **Día de la semana**: Se puede usar como una característica categórica que indique si es lunes, martes, etc.

### Modelo Utilizado

Se emplea el algoritmo **XGBoost** para la regresión, que predice el **porcentaje de cambio**. **XGBoost** es un modelo basado en árboles de decisión que ha demostrado ser eficiente para problemas de regresión y clasificación, especialmente con grandes cantidades de datos como las series temporales de mercados financieros.

#### Entrenamiento y Evaluación

1. **División de los datos**:

   - Los datos se dividen en tres conjuntos: **Entrenamiento**, **Validación** y **Prueba**.
   - **Entrenamiento**: 70% de los datos.
   - **Validación**: 15% de los datos.
   - **Prueba**: 15% de los datos.

   El modelo se entrena utilizando el conjunto de entrenamiento y se valida con el conjunto de validación para ajustar los parámetros del modelo.

2. **Métricas de evaluación**:

   - **Error Absoluto Medio (MAE)**: Mide la magnitud promedio de los errores sin tener en cuenta su dirección (si el mercado sube o baja).
   - **Coeficiente de Determinación (R²)**: Mide cuán bien las predicciones se ajustan a los datos reales. Un R² cercano a 1 indica un buen ajuste del modelo.

3. **Ajuste de parámetros**:
   - Se realiza una búsqueda de hiperparámetros para optimizar el rendimiento del modelo, utilizando técnicas como **Grid Search** o **Random Search**.

### Backtesting

**Backtesting** es el proceso de evaluar el rendimiento del modelo utilizando datos históricos. El modelo hace predicciones basadas en los datos del pasado y se calcula la rentabilidad de seguir esas predicciones.

#### Procedimiento de Backtesting:

1. El modelo utiliza los datos previos a las 20:00 para predecir el **porcentaje de cambio** entre las 20:00 y las 23:00.
2. Se calcula la **rentabilidad acumulada** de las predicciones.
3. **Error de predicción**: Comparando la predicción del porcentaje con el cambio real.

   La rentabilidad se calcula considerando el **porcentaje de cambio** predicho por el modelo y su comparación con el cambio real.

### Resultados Esperados

- **Precisión del modelo**: El modelo debe ser capaz de predecir con alta precisión el **porcentaje de cambio** en el mercado entre las 20:00 y las 23:00.
- **Rentabilidad en Backtesting**: Se espera que el modelo ofrezca un rendimiento positivo en el **backtest**, demostrando que las predicciones de porcentaje de cambio son acertadas a lo largo del tiempo.

### Conclusiones

Este proyecto proporciona un modelo de **regresión** para predecir el **porcentaje de cambio** del mercado en el **Dow Jones** durante un intervalo específico (20:00 - 23:00), utilizando **XGBoost** y los datos históricos disponibles. A través de un enfoque basado en **características diarias** y **semanales**, el modelo tiene la capacidad de capturar patrones de comportamiento del mercado y predecir tanto la **dirección** como la **magnitud** del movimiento del mercado.

Este tipo de enfoque es útil para **day trading** y puede ser ampliado o ajustado para predecir otros intervalos de tiempo o activos financieros.

---

## Requisitos del Proyecto

1. **Python 3.x** con uv como administrador de dependencias
2. **Bibliotecas**:
   - **XGBoost**
   - **Pandas**
   - **Scikit-learn**
   - **TA** (para indicadores técnicos)
   - **Matplotlib** (opcional, para visualización)
3. **Datos**: Datos históricos de **Dow Jones** con resolución de **1 minuto** desde 2013.

---

Este es el diseño completo del proyecto. Si deseas más detalles sobre el proceso o ajustes específicos, no dudes en preguntar. ¡Buena suerte con el desarrollo!
