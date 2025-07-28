import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import ElasticNet
import ta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class DowJonesPredictor:
    def __init__(self, max_samples=None, chunk_size=10000):
        self.model = None
        self.feature_columns = []
        self.data = None
        self.max_samples = None  # Siempre procesar todos los datos
        self.chunk_size = chunk_size    # Procesar en chunks
        
    def load_data(self, csv_path):
        """Cargar datos del CSV con el formato especificado - OPTIMIZADO"""
        print("Cargando datos del CSV...")
        
        # Leer con optimizaciones
        df = pd.read_csv(csv_path, 
                        dtype={'Open': 'float32', 'High': 'float32', 'Low': 'float32', 
                               'Close': 'float32', 'Volume': 'int32'})
        
        print(f"Datos cargados inicialmente: {len(df)} registros")
        
        # Procesar todos los datos sin limitaciones
        # if self.max_samples and len(df) > self.max_samples:
        #     print(f"Dataset grande detectado. Tomando muestra de {self.max_samples} registros...")
        #     df = df.tail(self.max_samples).copy()
        
        # Combinar Date y Time en datetime de forma optimizada
        df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'], 
                                       format='%Y%m%d %H:%M:%S', errors='coerce')
        
        # Eliminar registros con datetime inválido
        df = df.dropna(subset=['datetime'])
        
        # Renombrar columnas para consistencia
        df = df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume'
        })
        
        # Seleccionar columnas necesarias y optimizar tipos de datos
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
        df['open'] = df['open'].astype('float32')
        df['high'] = df['high'].astype('float32') 
        df['low'] = df['low'].astype('float32')
        df['close'] = df['close'].astype('float32')
        df['volume'] = df['volume'].astype('int32')
        
        # Ordenar por datetime y resetear índice
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Filtrar solo horarios de trading para optimizar desde el inicio
        df = df[(df['datetime'].dt.hour >= 9) & (df['datetime'].dt.hour <= 23)].copy()
        
        self.data = df
        print(f"Datos procesados: {len(df)} registros desde {df['datetime'].min()} hasta {df['datetime'].max()}")
        return df
    
    def create_features(self, df):
        """Crear características (features) según especificaciones - SÚPER OPTIMIZADO"""
        print("Creando características...")
        
        # Usar copia del dataframe con tipos optimizados
        data = df.copy()
        data_len = len(data)
        
        # Extraer información temporal de una vez
        print("Extrayendo información temporal...")
        data['hour'] = data['datetime'].dt.hour.astype('int8')
        data['minute'] = data['datetime'].dt.minute.astype('int8')
        data['day_of_week'] = data['datetime'].dt.dayofweek.astype('int8')
        data['date'] = data['datetime'].dt.date
        
        # Calcular indicadores técnicos con optimizaciones
        print("Calculando indicadores técnicos optimizados...")
        
        # Usar numpy para cálculos más rápidos cuando sea posible
        close_values = data['close'].values
        
        # SMA optimizado usando pandas rolling (más rápido que ta para SMAs simples)
        data['sma_15'] = data['close'].rolling(window=15, min_periods=1).mean().astype('float32')
        data['sma_30'] = data['close'].rolling(window=30, min_periods=1).mean().astype('float32')
        
        # RSI y MACD (usar ta pero con optimizaciones)
        data['rsi'] = ta.momentum.rsi(data['close'], window=14).astype('float32')
        
        # MACD - calcular solo lo necesario
        exp1 = data['close'].ewm(span=12).mean()
        exp2 = data['close'].ewm(span=26).mean()
        data['macd'] = (exp1 - exp2).astype('float32')
        
        # Filtrar datos de trading (17:00 a 23:59) para reducir procesamiento
        print("Filtrando horarios de trading...")
        trading_mask = (data['hour'] >= 17) & (data['hour'] <= 23)
        trading_data = data[trading_mask].copy()
        
        if len(trading_data) == 0:
            print("No hay datos en horarios de trading")
            return pd.DataFrame()
        
        # Pre-calcular estadísticas diarias con agregaciones optimizadas
        print("Pre-calculando estadísticas diarias...")
        daily_agg = {
            'open': 'first',
            'high': 'max',
            'low': 'min', 
            'close': ['first', 'last'],
            'volume': 'sum'
        }
        
        daily_stats = trading_data.groupby('date', sort=False).agg(daily_agg)
        daily_stats.columns = ['day_open', 'day_high', 'day_low', 'day_close_first', 'day_close_last', 'day_volume']
        
        # Calcular variación diaria de forma vectorizada
        daily_stats['daily_variation'] = (
            (daily_stats['day_close_last'] - daily_stats['day_open']) / 
            daily_stats['day_open'] * 100
        ).astype('float32')
        
        # Optimización: encontrar puntos de tiempo más eficientemente
        print("Encontrando puntos de tiempo objetivo...")
        
        # Crear máscaras para horarios específicos
        mask_20h = (trading_data['hour'] >= 19) & (trading_data['hour'] <= 21)
        mask_23h = (trading_data['hour'] >= 22) & (trading_data['hour'] <= 23)
        
        data_20h = trading_data[mask_20h].copy()
        data_23h = trading_data[mask_23h].copy()
        
        if len(data_20h) == 0 or len(data_23h) == 0:
            print("Insuficientes datos para horarios objetivo")
            return pd.DataFrame()
        
        # Encontrar el registro más cercano a las 20:00 por día
        target_time_20h = pd.to_datetime(data_20h['date'].astype(str) + ' 20:00:00')
        data_20h['time_diff'] = abs((data_20h['datetime'] - target_time_20h).dt.total_seconds())
        closest_20h = data_20h.loc[data_20h.groupby('date')['time_diff'].idxmin()]
        
        # Encontrar el registro más cercano a las 23:00 por día  
        target_time_23h = pd.to_datetime(data_23h['date'].astype(str) + ' 23:00:00')
        data_23h['time_diff'] = abs((data_23h['datetime'] - target_time_23h).dt.total_seconds())
        closest_23h = data_23h.loc[data_23h.groupby('date')['time_diff'].idxmin()]
        
        # Merge optimizado para combinar 20h y 23h
        merged = closest_20h.merge(
            closest_23h[['date', 'close']],
            on='date',
            suffixes=('', '_23h'),
            how='inner'
        )
        
        if len(merged) == 0:
            print("No hay coincidencias entre datos de 20h y 23h")
            return pd.DataFrame()
        
        # Calcular target de forma vectorizada
        merged['target'] = (
            (merged['close_23h'] - merged['close']) / merged['close'] * 100
        ).astype('float32')
        
        # Merge con estadísticas diarias
        merged = merged.merge(daily_stats, left_on='date', right_index=True, how='left')
        
        # Características semanales optimizadas
        print("Calculando características semanales...")
        merged = merged.sort_values('date').reset_index(drop=True)
        
        # Rolling windows optimizados
        merged['week_avg_close'] = merged['close'].rolling(window=7, min_periods=1).mean().astype('float32')
        merged['week_volatility'] = merged['close'].rolling(window=7, min_periods=1).std().fillna(0).astype('float32')
        merged['week_trend'] = (merged['close'] > merged['close'].shift(7)).astype('int8')
        
        # Seleccionar columnas finales de forma eficiente
        feature_cols = [
            'datetime', 'day_of_week', 'day_open', 'close', 'day_high', 'day_low',
            'day_volume', 'daily_variation', 'sma_15', 'sma_30', 'rsi', 'macd',
            'week_avg_close', 'week_volatility', 'week_trend', 'target'
        ]
        
        features_df = merged[feature_cols].copy()
        
        # Renombrar columnas
        features_df = features_df.rename(columns={
            'close': 'day_close_current',
            'sma_15': 'current_sma_15', 
            'sma_30': 'current_sma_30',
            'rsi': 'current_rsi',
            'macd': 'current_macd'
        })
        
        # Limpieza final optimizada
        features_df['current_sma_15'] = features_df['current_sma_15'].fillna(features_df['day_close_current'])
        features_df['current_sma_30'] = features_df['current_sma_30'].fillna(features_df['day_close_current']) 
        features_df['current_rsi'] = features_df['current_rsi'].fillna(50)
        features_df['current_macd'] = features_df['current_macd'].fillna(0)
        features_df['week_trend'] = features_df['week_trend'].fillna(0)
        
        # Eliminar NaN restantes
        features_df = features_df.dropna()
        
        print(f"Características creadas: {len(features_df)} registros")
        print(f"Reducción de datos: {data_len} -> {len(features_df)} ({len(features_df)/data_len*100:.1f}%)")
        
        return features_df
    
    def create_features_chunked(self, df):
        """Procesar datasets grandes en chunks para optimizar memoria"""
        print(f"Procesando dataset grande ({len(df)} registros) en chunks...")
        
        chunk_results = []
        total_chunks = len(df) // self.chunk_size + (1 if len(df) % self.chunk_size > 0 else 1)
        
        for i in range(0, len(df), self.chunk_size):
            chunk = df.iloc[i:i+self.chunk_size].copy()
            print(f"Procesando chunk {i//self.chunk_size + 1}/{total_chunks} ({len(chunk)} registros)")
            
            chunk_features = self.create_features(chunk)
            if len(chunk_features) > 0:
                chunk_results.append(chunk_features)
        
        if chunk_results:
            final_features = pd.concat(chunk_results, ignore_index=True)
            print(f"Chunks procesados: {len(chunk_results)}, registros finales: {len(final_features)}")
            return final_features
        else:
            print("No se generaron características válidas")
            return pd.DataFrame()
    
    def auto_optimize_processing(self, df):
        """Decidir automáticamente el método de procesamiento óptimo"""
        data_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        print(f"Tamaño del dataset: {data_size_mb:.1f} MB")
        
        if data_size_mb > 500:  # > 500MB
            print("Dataset grande detectado - usando procesamiento en chunks")
            return self.create_features_chunked(df)
        elif data_size_mb > 100:  # > 100MB
            print("Dataset mediano detectado - usando optimizaciones estándar")
            return self.create_features(df)
        else:
            print("Dataset pequeño - procesamiento normal")
            return self.create_features(df)
    
    def prepare_data(self, features_df):
        """Preparar datos para entrenamiento"""
        print("Preparando datos para entrenamiento...")
        
        # Definir columnas de características
        self.feature_columns = [col for col in features_df.columns if col not in ['datetime', 'target']]
        
        X = features_df[self.feature_columns]
        y = features_df['target']
        
        # División de datos: 70% entrenamiento, 15% validación, 15% prueba
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, shuffle=False)  # 0.176 ≈ 0.15/0.85
        
        print(f"Datos de entrenamiento: {len(X_train)}")
        print(f"Datos de validación: {len(X_val)}")
        print(f"Datos de prueba: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, features_df
    
    def train_model(self, X_train, X_val, y_train, y_val):
        """Entrenar modelo optimizado con GridSearch y Ensemble"""
        print("Entrenando modelo optimizado con GridSearch y Ensemble...")
        
        # 1. Optimización de XGBoost con GridSearch
        print("🔍 Paso 1: Optimizando hiperparámetros de XGBoost...")
        
        xgb_param_grid = {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.15],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
        
        # TimeSeriesSplit para validación temporal
        tscv = TimeSeriesSplit(n_splits=3)
        
        xgb_base = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        
        xgb_grid = GridSearchCV(
            xgb_base,
            xgb_param_grid,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        
        xgb_grid.fit(X_train, y_train)
        best_xgb = xgb_grid.best_estimator_
        
        print(f"✅ Mejores parámetros XGBoost: {xgb_grid.best_params_}")
        print(f"✅ Mejor score XGBoost: {-xgb_grid.best_score_:.4f}")
        
        # 2. Entrenar modelos adicionales para ensemble
        print("🎯 Paso 2: Entrenando modelos adicionales...")
        
        # Random Forest optimizado
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting
        gb_model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # ElasticNet para regularización
        en_model = ElasticNet(
            alpha=0.1,
            l1_ratio=0.5,
            random_state=42
        )
        
        # Entrenar modelos individuales
        rf_model.fit(X_train, y_train)
        gb_model.fit(X_train, y_train)
        en_model.fit(X_train, y_train)
        
        # 3. Crear ensemble
        print("🚀 Paso 3: Creando ensemble de modelos...")
        
        self.ensemble_model = VotingRegressor([
            ('xgb', best_xgb),
            ('rf', rf_model),
            ('gb', gb_model),
            ('en', en_model)
        ], weights=[0.4, 0.25, 0.25, 0.1])  # XGBoost tiene más peso
        
        self.ensemble_model.fit(X_train, y_train)
        
        # 4. Evaluar modelos en validación
        xgb_val_pred = best_xgb.predict(X_val)
        ensemble_val_pred = self.ensemble_model.predict(X_val)
        
        xgb_val_mae = mean_absolute_error(y_val, xgb_val_pred)
        ensemble_val_mae = mean_absolute_error(y_val, ensemble_val_pred)
        
        print(f"📊 MAE Validación - XGBoost: {xgb_val_mae:.4f}")
        print(f"📊 MAE Validación - Ensemble: {ensemble_val_mae:.4f}")
        
        # Seleccionar el mejor modelo
        if ensemble_val_mae < xgb_val_mae:
            self.model = self.ensemble_model
            print("✅ Ensemble seleccionado como modelo final")
        else:
            self.model = best_xgb
            print("✅ XGBoost seleccionado como modelo final")
        
        # Guardar modelos individuales para análisis
        self.xgb_model = best_xgb
        self.rf_model = rf_model
        self.gb_model = gb_model
        self.en_model = en_model
        
        print("🎉 Entrenamiento completado exitosamente")
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluar modelo con métricas"""
        print("Evaluando modelo...")
        
        # Realizar predicciones
        y_pred = self.model.predict(X_test)
        
        # Calcular métricas
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Error Absoluto Medio (MAE): {mae:.4f}")
        print(f"Coeficiente de Determinación (R²): {r2:.4f}")
        
        return {
            'mae': mae,
            'r2': r2,
            'predictions': y_pred,
            'actual': y_test
        }
    
    def backtest(self, features_df):
        """Realizar backtesting del modelo - COMPLETO con métricas avanzadas, CSV y drawdown intraoperación"""
        print("Realizando backtesting...")
        
        # Usar datos ordenados cronológicamente
        features_df = features_df.sort_values('datetime').reset_index(drop=True)
        
        # CORRECCIÓN: Usar solo los datos más recientes que no se usaron para entrenar
        # El modelo usa 85% para entrenamiento+validación, 15% para prueba
        # Para backtesting riguroso, usar solo datos posteriores al 85%
        train_val_size = int(len(features_df) * 0.85)
        
        # Backtesting con datos completamente independientes (después del 85%)
        backtest_data = features_df.iloc[train_val_size:].copy()
        
        print(f"Período de backtesting: {backtest_data['datetime'].min()} a {backtest_data['datetime'].max()}")
        print(f"Días de backtesting: {len(backtest_data)}")
        
        if len(backtest_data) == 0:
            print("No hay datos suficientes para backtesting independiente")
            return {'total_return': 0, 'accuracy': 0, 'backtest_data': pd.DataFrame()}
        
        # Realizar predicciones
        X_backtest = backtest_data[self.feature_columns]
        predictions = self.model.predict(X_backtest)
        
        # Crear DataFrame detallado para análisis
        backtest_data['predictions'] = predictions
        backtest_data['actual_change'] = backtest_data['target']
        
        # Estrategia de trading detallada
        backtest_data['predicted_direction'] = np.where(backtest_data['predictions'] > 0, 'BUY', 'SELL')
        backtest_data['actual_direction'] = np.where(backtest_data['actual_change'] > 0, 'UP', 'DOWN')
        backtest_data['correct_direction'] = (
            ((backtest_data['predictions'] > 0) & (backtest_data['actual_change'] > 0)) |
            ((backtest_data['predictions'] < 0) & (backtest_data['actual_change'] < 0))
        )
        
        # Posiciones y returns
        backtest_data['position'] = np.where(backtest_data['predictions'] > 0, 1, -1)  # 1: Long, -1: Short
        backtest_data['returns'] = backtest_data['position'] * backtest_data['actual_change']
        backtest_data['cumulative_returns'] = backtest_data['returns'].cumsum()
        
        # === NUEVO: CALCULAR DRAWDOWN INTRAOPERACIÓN ===
        print("📊 Calculando drawdown intraoperación (20:00-23:00)...")
        
        # Inicializar columnas para drawdown intraoperación
        backtest_data['max_intra_drawdown_pct'] = 0.0
        backtest_data['max_intra_drawdown_dollars'] = 0.0
        backtest_data['min_pnl_during_trade'] = 0.0
        backtest_data['time_of_max_drawdown'] = None
        
        # Para cada día de backtesting, calcular drawdown intraoperación
        for idx, row in backtest_data.iterrows():
            trade_date = row['datetime'].date()
            position = row['position']  # 1 para LONG, -1 para SHORT
            
            # Obtener datos minuto a minuto para ese día entre 20:00 y 23:00
            day_mask = (self.data['datetime'].dt.date == trade_date) & \
                      (self.data['datetime'].dt.hour >= 20) & \
                      (self.data['datetime'].dt.hour <= 23)
            
            day_minute_data = self.data[day_mask].copy()
            
            if len(day_minute_data) == 0:
                continue
                
            # Precio de entrada (más cercano a las 20:00)
            entry_time = datetime.combine(trade_date, time(20, 0))
            day_minute_data['time_diff'] = abs((day_minute_data['datetime'] - entry_time).dt.total_seconds())
            entry_idx = day_minute_data['time_diff'].idxmin()
            entry_price = day_minute_data.loc[entry_idx, 'close']
            
            # Calcular P&L flotante minuto a minuto
            day_minute_data['pnl_pct'] = ((day_minute_data['close'] - entry_price) / entry_price * 100) * position
            
            # Encontrar el mínimo P&L (máximo drawdown)
            min_pnl_idx = day_minute_data['pnl_pct'].idxmin()
            max_drawdown_pct = day_minute_data.loc[min_pnl_idx, 'pnl_pct']
            time_of_max_dd = day_minute_data.loc[min_pnl_idx, 'datetime']
            
            # Solo considerar como drawdown si es negativo
            if max_drawdown_pct < 0:
                max_drawdown_dollars = max_drawdown_pct * 5000  # $5000 por cada 1%
                
                backtest_data.at[idx, 'max_intra_drawdown_pct'] = max_drawdown_pct
                backtest_data.at[idx, 'max_intra_drawdown_dollars'] = max_drawdown_dollars
                backtest_data.at[idx, 'min_pnl_during_trade'] = max_drawdown_pct
                backtest_data.at[idx, 'time_of_max_drawdown'] = time_of_max_dd
        
        print(f"✅ Drawdown intraoperación calculado para {len(backtest_data)} operaciones")
        
        # Calcular equity curve (valor del portafolio) - SISTEMA REAL
        initial_capital = 100000  # $100,000 capital inicial
        
        # Sistema de trading: cada 0.1% de movimiento = $500
        # Esto significa que el "notional value" efectivo es $500,000 (5x apalancamiento)
        notional_multiplier = 5000  # $500 por cada 0.1% -> $5000 por cada 1%
        
        backtest_data['pnl_dollars'] = backtest_data['returns'] * notional_multiplier
        backtest_data['cumulative_pnl'] = backtest_data['pnl_dollars'].cumsum()
        backtest_data['equity'] = initial_capital + backtest_data['cumulative_pnl']
        backtest_data['daily_pnl'] = backtest_data['equity'].diff().fillna(0)
        
        # Trades ganadores y perdedores
        backtest_data['win_trade'] = backtest_data['returns'] > 0
        backtest_data['loss_trade'] = backtest_data['returns'] < 0
        backtest_data['break_even_trade'] = backtest_data['returns'] == 0
        
        # Cálculo de rachas
        backtest_data['win_streak'] = (backtest_data['win_trade'].groupby(
            (backtest_data['win_trade'] != backtest_data['win_trade'].shift()).cumsum()
        ).cumsum() * backtest_data['win_trade']).astype(int)
        
        backtest_data['loss_streak'] = (backtest_data['loss_trade'].groupby(
            (backtest_data['loss_trade'] != backtest_data['loss_trade'].shift()).cumsum()
        ).cumsum() * backtest_data['loss_trade']).astype(int)
        
        # Drawdown del portafolio (diferente del drawdown intraoperación)
        backtest_data['running_max'] = backtest_data['equity'].expanding().max()
        backtest_data['portfolio_drawdown'] = (backtest_data['equity'] / backtest_data['running_max'] - 1) * 100
        
        # Guardar CSV detallado
        csv_columns = [
            'datetime', 'day_close_current', 'predictions', 'actual_change', 
            'predicted_direction', 'actual_direction', 'correct_direction',
            'position', 'returns', 'pnl_dollars', 'cumulative_pnl', 'equity', 'daily_pnl',
            'max_intra_drawdown_pct', 'max_intra_drawdown_dollars', 'time_of_max_drawdown',
            'win_trade', 'loss_trade', 'portfolio_drawdown', 'day_of_week'
        ]
        
        csv_data = backtest_data[csv_columns].copy()
        csv_data.to_csv('backtest_detallado.csv', index=False)
        print("📊 Archivo 'backtest_detallado.csv' generado con todas las operaciones y drawdown intraoperación")
        
        # === MÉTRICAS AVANZADAS DE PERFORMANCE ===
        
        # Métricas básicas
        total_trades = len(backtest_data)
        winning_trades = len(backtest_data[backtest_data['win_trade']])
        losing_trades = len(backtest_data[backtest_data['loss_trade']])
        break_even_trades = len(backtest_data[backtest_data['break_even_trade']])
        
        # Returns y rentabilidad en DÓLARES
        total_pnl = backtest_data['cumulative_pnl'].iloc[-1]
        total_return_pct = (total_pnl / initial_capital) * 100
        returns_series = backtest_data['returns']  # En porcentaje
        pnl_series = backtest_data['pnl_dollars']  # En dólares
        
        daily_returns = backtest_data['daily_pnl'] / backtest_data['equity'].shift(1) * 100
        daily_returns = daily_returns.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Accuracy y win rate
        accuracy = np.mean(backtest_data['correct_direction']) * 100
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Profit factor (en dólares)
        gross_profit = pnl_series[pnl_series > 0].sum()
        gross_loss = abs(pnl_series[pnl_series < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Average wins/losses (en dólares y porcentaje)
        avg_win_pct = returns_series[returns_series > 0].mean() if winning_trades > 0 else 0
        avg_loss_pct = returns_series[returns_series < 0].mean() if losing_trades > 0 else 0
        avg_win_dollars = pnl_series[pnl_series > 0].mean() if winning_trades > 0 else 0
        avg_loss_dollars = pnl_series[pnl_series < 0].mean() if losing_trades > 0 else 0
        
        # Drawdown máximo del portafolio
        max_portfolio_drawdown = backtest_data['portfolio_drawdown'].min()
        max_drawdown_dollars = (initial_capital + backtest_data['cumulative_pnl']).min() - initial_capital
        
        # === NUEVAS MÉTRICAS DE DRAWDOWN INTRAOPERACIÓN ===
        intra_drawdowns = backtest_data[backtest_data['max_intra_drawdown_pct'] < 0]
        
        if len(intra_drawdowns) > 0:
            # Estadísticas de drawdown intraoperación
            max_intra_dd_pct = intra_drawdowns['max_intra_drawdown_pct'].min()  # El más negativo
            max_intra_dd_dollars = intra_drawdowns['max_intra_drawdown_dollars'].min()
            avg_intra_dd_pct = intra_drawdowns['max_intra_drawdown_pct'].mean()
            avg_intra_dd_dollars = intra_drawdowns['max_intra_drawdown_dollars'].mean()
            trades_with_dd = len(intra_drawdowns)
            pct_trades_with_dd = (trades_with_dd / total_trades) * 100
            
            # Distribución de drawdowns
            dd_std = intra_drawdowns['max_intra_drawdown_pct'].std()
        else:
            max_intra_dd_pct = 0
            max_intra_dd_dollars = 0
            avg_intra_dd_pct = 0
            avg_intra_dd_dollars = 0
            trades_with_dd = 0
            pct_trades_with_dd = 0
            dd_std = 0
        
        # Volatilidad anualizada (asumiendo ~250 días de trading por año)
        volatility = daily_returns.std() * np.sqrt(250) if len(daily_returns) > 1 else 0
        
        # Sharpe Ratio (asumiendo risk-free rate = 0)
        sharpe_ratio = (daily_returns.mean() * 250) / volatility if volatility > 0 else 0
        
        # Calmar Ratio
        annual_return = (total_return_pct / len(backtest_data)) * 250 if len(backtest_data) > 0 else 0
        calmar_ratio = annual_return / abs(max_portfolio_drawdown) if max_portfolio_drawdown != 0 else 0
        
        # Rachas máximas
        max_win_streak = backtest_data['win_streak'].max()
        max_loss_streak = backtest_data['loss_streak'].max()
        
        # Recovery Factor
        recovery_factor = total_return_pct / abs(max_portfolio_drawdown) if max_portfolio_drawdown != 0 else 0
        
        # Expectancy (en dólares)
        expectancy_dollars = (win_rate/100 * avg_win_dollars) + ((1-win_rate/100) * avg_loss_dollars)
        expectancy_pct = (win_rate/100 * avg_win_pct) + ((1-win_rate/100) * avg_loss_pct)
        
        # === MOSTRAR MÉTRICAS ===
        print("\n" + "="*80)
        print("📈 MÉTRICAS AVANZADAS DE PERFORMANCE - CAPITAL REAL")
        print("="*80)
        
        print(f"\n💰 CAPITAL Y RENTABILIDAD:")
        print(f"  • Capital Inicial: ${initial_capital:,.2f}")
        print(f"  • Capital Final: ${backtest_data['equity'].iloc[-1]:,.2f}")
        print(f"  • P&L Total: ${total_pnl:,.2f}")
        print(f"  • Rentabilidad Total: {total_return_pct:.2f}%")
        print(f"  • Rentabilidad Anualizada: {annual_return:.2f}%")
        
        print(f"\n📊 PRECISIÓN:")
        print(f"  • Precisión Direccional: {accuracy:.1f}%")
        print(f"  • Win Rate: {win_rate:.1f}%")
        print(f"  • Expectancy por Trade: ${expectancy_dollars:.2f} ({expectancy_pct:.4f}%)")
        
        print(f"\n💰 TRADES:")
        print(f"  • Total de Trades: {total_trades}")
        print(f"  • Trades Ganadores: {winning_trades} ({win_rate:.1f}%)")
        print(f"  • Trades Perdedores: {losing_trades} ({(losing_trades/total_trades)*100:.1f}%)")
        print(f"  • Trades Break-even: {break_even_trades}")
        
        print(f"\n🎲 RENDIMIENTO PROMEDIO:")
        print(f"  • Ganancia Promedio: ${avg_win_dollars:.2f} ({avg_win_pct:.4f}%)")
        print(f"  • Pérdida Promedio: ${avg_loss_dollars:.2f} ({avg_loss_pct:.4f}%)")
        print(f"  • Profit Factor: {profit_factor:.2f}")
        
        print(f"\n⚠️ RIESGO - PORTAFOLIO:")
        print(f"  • Máximo Drawdown Portafolio: {max_portfolio_drawdown:.2f}% (${max_drawdown_dollars:,.2f})")
        print(f"  • Volatilidad Anualizada: {volatility:.2f}%")
        print(f"  • Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"  • Calmar Ratio: {calmar_ratio:.2f}")
        print(f"  • Recovery Factor: {recovery_factor:.2f}")
        
        print(f"\n📉 RIESGO - DRAWDOWN INTRAOPERACIÓN:")
        print(f"  • Trades con Drawdown: {trades_with_dd}/{total_trades} ({pct_trades_with_dd:.1f}%)")
        print(f"  • Máximo Drawdown Intraoperación: {max_intra_dd_pct:.4f}% (${max_intra_dd_dollars:.2f})")
        print(f"  • Drawdown Intraoperación Promedio: {avg_intra_dd_pct:.4f}% (${avg_intra_dd_dollars:.2f})")
        print(f"  • Desviación Std Drawdown Intra: {dd_std:.4f}%")
        
        print(f"\n🔄 RACHAS:")
        print(f"  • Máxima Racha Ganadora: {max_win_streak} trades")
        print(f"  • Máxima Racha Perdedora: {max_loss_streak} trades")
        
        print(f"\n💡 SISTEMA DE TRADING:")
        print(f"  • Cada 0.1% de movimiento = $500")
        print(f"  • Notional Value Efectivo: ${notional_multiplier * 100:,.0f}")
        print(f"  • Apalancamiento Efectivo: {(notional_multiplier * 100) / initial_capital:.1f}x")
        
        print("="*80)
        
        # Crear resumen de métricas
        performance_metrics = {
            'total_return': total_return_pct,
            'total_pnl': total_pnl,
            'annual_return': annual_return,
            'accuracy': accuracy,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_portfolio_drawdown,
            'max_drawdown_dollars': max_drawdown_dollars,
            'max_intra_dd_pct': max_intra_dd_pct,
            'max_intra_dd_dollars': max_intra_dd_dollars,
            'avg_intra_dd_pct': avg_intra_dd_pct,
            'avg_intra_dd_dollars': avg_intra_dd_dollars,
            'trades_with_dd': trades_with_dd,
            'pct_trades_with_dd': pct_trades_with_dd,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'volatility': volatility,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'avg_win_pct': avg_win_pct,
            'avg_loss_pct': avg_loss_pct,
            'avg_win_dollars': avg_win_dollars,
            'avg_loss_dollars': avg_loss_dollars,
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            'expectancy_pct': expectancy_pct,
            'expectancy_dollars': expectancy_dollars,
            'recovery_factor': recovery_factor,
            'final_equity': backtest_data['equity'].iloc[-1],
            'initial_capital': initial_capital,
            'backtest_data': backtest_data
        }
        
        return performance_metrics
    
    def plot_results(self, backtest_results):
        """Visualizar resultados - MEJORADO con métricas avanzadas y drawdown intraoperación"""
        print("Generando gráficos...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        backtest_data = backtest_results['backtest_data']
        
        # Gráfico 1: Equity curve y drawdown del portafolio
        ax1_twin = ax1.twinx()
        ax1.plot(range(len(backtest_data)), backtest_data['equity'], 'b-', linewidth=2, label='Equity ($)')
        ax1_twin.fill_between(range(len(backtest_data)), backtest_data['portfolio_drawdown'], 0, 
                             color='red', alpha=0.3, label='Portfolio Drawdown (%)')
        
        # Añadir línea de capital inicial
        ax1.axhline(y=backtest_results['initial_capital'], color='gray', linestyle='--', alpha=0.7, label='Capital Inicial')
        
        ax1.set_title(f'Equity Curve: ${backtest_results["initial_capital"]:,.0f} → ${backtest_results["final_equity"]:,.0f}', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Período')
        ax1.set_ylabel('Valor del Portafolio ($)', color='blue')
        ax1_twin.set_ylabel('Portfolio Drawdown (%)', color='red')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # Gráfico 2: Predicciones vs Valores Reales (mejorado)
        correct_preds = backtest_data[backtest_data['correct_direction']]
        wrong_preds = backtest_data[~backtest_data['correct_direction']]
        
        ax2.scatter(wrong_preds['actual_change'], wrong_preds['predictions'], 
                   alpha=0.6, color='red', s=30, label=f'Incorrectas ({len(wrong_preds)})')
        ax2.scatter(correct_preds['actual_change'], correct_preds['predictions'], 
                   alpha=0.6, color='green', s=30, label=f'Correctas ({len(correct_preds)})')
        ax2.plot([-10, 10], [-10, 10], 'k--', label='Línea perfecta', alpha=0.7)
        ax2.set_title(f'Predicciones vs Valores Reales\nPrecisión: {backtest_results["accuracy"]:.1f}%', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Cambio Real (%)')
        ax2.set_ylabel('Predicción (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: Distribución de P&L en dólares
        ax3.hist(backtest_data['pnl_dollars'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax3.axvline(backtest_data['pnl_dollars'].mean(), color='red', linestyle='--', 
                   label=f'Media: ${backtest_data["pnl_dollars"].mean():.2f}')
        ax3.set_title(f'Distribución de P&L por Trade\nExpectancy: ${backtest_results["expectancy_dollars"]:.2f}', 
                     fontsize=14, fontweight='bold')
        ax3.set_xlabel('P&L por Trade ($)')
        ax3.set_ylabel('Frecuencia')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: Análisis de Drawdown Intraoperación
        intra_drawdowns = backtest_data[backtest_data['max_intra_drawdown_pct'] < 0]
        
        if len(intra_drawdowns) > 5:
            # Histograma de drawdowns intraoperación
            ax4.hist(intra_drawdowns['max_intra_drawdown_pct'], bins=20, alpha=0.7, 
                    color='darkred', edgecolor='black')
            ax4.axvline(intra_drawdowns['max_intra_drawdown_pct'].mean(), color='orange', 
                       linestyle='--', linewidth=2, label=f'Media: {intra_drawdowns["max_intra_drawdown_pct"].mean():.3f}%')
            ax4.axvline(intra_drawdowns['max_intra_drawdown_pct'].min(), color='red', 
                       linestyle='-', linewidth=2, label=f'Máximo: {intra_drawdowns["max_intra_drawdown_pct"].min():.3f}%')
            
            ax4.set_title(f'Distribución Drawdown Intraoperación\n{len(intra_drawdowns)}/{len(backtest_data)} trades ({backtest_results["pct_trades_with_dd"]:.1f}%)', 
                         fontsize=14, fontweight='bold')
            ax4.set_xlabel('Drawdown Intraoperación (%)')
            ax4.set_ylabel('Frecuencia')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            # Si hay pocos drawdowns, mostrar timeline
            ax4.scatter(range(len(backtest_data)), backtest_data['max_intra_drawdown_pct'], 
                       c=backtest_data['max_intra_drawdown_pct'], cmap='RdYlGn', s=50, alpha=0.7)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax4.set_title(f'Timeline Drawdown Intraoperación\nMáximo: {backtest_results["max_intra_dd_pct"]:.3f}%', 
                         fontsize=14, fontweight='bold')
            ax4.set_xlabel('Período')
            ax4.set_ylabel('Drawdown Intraoperación (%)')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=3.0)
        plt.savefig('resultados_predictor_dj_completo.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📈 Gráficos guardados como 'resultados_predictor_dj_completo.png'")
        print(f"📊 Drawdown intraoperación analizado: {backtest_results['trades_with_dd']} de {backtest_results['total_trades']} trades")
    
    def predict_next_period(self, current_data):
        """Predecir el próximo período basado en datos actuales"""
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado aún")
        
        # Procesar datos actuales para crear características
        # Este método sería utilizado para predicciones en tiempo real
        
        prediction = self.model.predict(current_data[self.feature_columns])
        return prediction[0]
    
    def run_complete_analysis(self, csv_path):
        """Ejecutar análisis completo"""
        print("=== INICIANDO ANÁLISIS COMPLETO DEL PREDICTOR DOW JONES ===\n")
        
        # 1. Cargar datos
        df = self.load_data(csv_path)
        
        # 2. Crear características
        features_df = self.auto_optimize_processing(df)
        
        if len(features_df) == 0:
            print("Error: No se pudieron crear características válidas")
            return None
        
        # 3. Preparar datos
        X_train, X_val, X_test, y_train, y_val, y_test, features_df = self.prepare_data(features_df)
        
        # 4. Entrenar modelo
        self.train_model(X_train, X_val, y_train, y_val)
        
        # 5. Evaluar modelo
        evaluation_results = self.evaluate_model(X_test, y_test)
        
        # 6. Realizar backtesting
        backtest_results = self.backtest(features_df)
        
        # 7. Visualizar resultados
        self.plot_results(backtest_results)
        
        print("\n" + "="*60)
        print("🏆 RESUMEN FINAL DE RESULTADOS - MODELO OPTIMIZADO")
        print("="*60)
        print(f"📊 MODELO:")
        print(f"  • Tipo: {type(self.model).__name__}")
        print(f"  • MAE: {evaluation_results['mae']:.4f}")
        print(f"  • R²: {evaluation_results['r2']:.4f}")
        print(f"\n💰 CAPITAL Y TRADING:")
        print(f"  • Capital Inicial: ${backtest_results['initial_capital']:,.2f}")
        print(f"  • Capital Final: ${backtest_results['final_equity']:,.2f}")
        print(f"  • P&L Total: ${backtest_results['total_pnl']:,.2f}")
        print(f"  • Rentabilidad Total: {backtest_results['total_return']:.2f}%")
        print(f"  • Rentabilidad Anualizada: {backtest_results['annual_return']:.2f}%")
        print(f"\n📈 PERFORMANCE:")
        print(f"  • Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
        print(f"  • Máximo Drawdown Portafolio: {backtest_results['max_drawdown']:.2f}% (${backtest_results['max_drawdown_dollars']:,.2f})")
        print(f"  • Profit Factor: {backtest_results['profit_factor']:.2f}")
        print(f"\n🎯 PRECISIÓN:")
        print(f"  • Precisión Direccional: {backtest_results['accuracy']:.1f}%")
        print(f"  • Win Rate: {backtest_results['win_rate']:.1f}%")
        print(f"  • Expectancy: ${backtest_results['expectancy_dollars']:.2f} por trade")
        print(f"\n📉 RIESGO INTRAOPERACIÓN:")
        print(f"  • Trades con Drawdown: {backtest_results['trades_with_dd']}/{backtest_results['total_trades']} ({backtest_results['pct_trades_with_dd']:.1f}%)")
        print(f"  • Máximo Drawdown Intraoperación: {backtest_results['max_intra_dd_pct']:.3f}% (${backtest_results['max_intra_dd_dollars']:.2f})")
        print(f"\n💡 SISTEMA:")
        print(f"  • 0.1% movimiento = $500")
        print(f"  • Apalancamiento Efectivo: 5.0x")
        print(f"\n📁 ARCHIVOS GENERADOS:")
        print(f"  • backtest_detallado.csv - Todas las operaciones + drawdown intraoperación")
        print(f"  • resultados_predictor_dj_completo.png - Gráficos avanzados")
        
        # Guardar modelo entrenado automáticamente
        model_saved = self.save_model('trained_model.pkl')
        if model_saved:
            print(f"  • trained_model.pkl - Modelo entrenado para trading en vivo")
        
        print("="*60)
        
        return {
            'model': self.model,
            'evaluation': evaluation_results,
            'backtest': backtest_results,
            'features_df': features_df
        }
    
    def save_model(self, filepath: str = 'trained_model.pkl') -> bool:
        """Guardar el modelo entrenado para uso en trading en vivo"""
        try:
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            print(f"✅ Modelo guardado en {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error guardando modelo: {e}")
            return False
    
    @classmethod
    def load_model(cls, filepath: str = 'trained_model.pkl'):
        """Cargar modelo entrenado desde archivo"""
        try:
            import pickle
            with open(filepath, 'rb') as f:
                predictor = pickle.load(f)
            print(f"✅ Modelo cargado desde {filepath}")
            return predictor
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return None

# Función principal para uso directo
def main(csv_path):
    """Función principal para ejecutar el predictor"""
    predictor = DowJonesPredictor()
    results = predictor.run_complete_analysis(csv_path)
    return predictor, results

def main_optimized(csv_path, max_samples=50000):
    """Función principal optimizada para datasets grandes"""
    predictor = DowJonesPredictor(max_samples=max_samples, chunk_size=10000)
    results = predictor.run_complete_analysis(csv_path)
    return predictor, results

if __name__ == "__main__":
    # El usuario proporcionará el path del CSV
    print("Predictor Dow Jones - Listo para recibir datos")
    print("Uso normal: predictor.run_complete_analysis('path_to_csv')")
    print("Uso optimizado: main_optimized('path_to_csv', max_samples=50000)")
    print("\nOptimizaciones implementadas:")
    print("- Tipos de datos optimizados (float32, int8)")
    print("- Procesamiento en chunks para datasets grandes")
    print("- Cálculos vectorizados y agregaciones eficientes")
    print("- Filtrado temprano de horarios no relevantes")
    print("- Limitación automática de muestras para datasets masivos") 