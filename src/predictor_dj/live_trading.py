"""
Sistema de Trading en Vivo para el Predictor Dow Jones
Utiliza el modelo entrenado para hacer predicciones y ejecutar operaciones
"""

import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime, time, timedelta
import pytz
import schedule
import time as time_module
import json
import os
from typing import Dict, List, Optional
import ta

from .broker_interface import BrokerInterface, create_broker
from .main import DowJonesPredictor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LiveTradingSystem:
    """Sistema de trading en vivo usando el modelo entrenado"""
    
    def __init__(self, config_path: str = "trading_config.json"):
        self.config = self.load_config(config_path)
        self.broker = None
        self.predictor = None
        self.is_running = False
        self.positions = {}
        self.daily_stats = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'pnl': 0.0,
            'start_balance': 0.0
        }
        
        # Configurar zona horaria
        self.timezone = pytz.timezone(self.config.get('timezone', 'Europe/Moscow'))  # UTC+3
        
    def load_config(self, config_path: str) -> Dict:
        """Cargar configuración del sistema"""
        default_config = {
            'broker_type': 'simulator',
            'symbol': 'US30',  # Dow Jones en MT5
            'volume': 0.1,
            'max_positions': 1,
            'max_daily_trades': 5,
            'stop_loss_pct': 2.0,
            'take_profit_pct': 3.0,
            'trading_hours': {
                'start': '20:00',
                'end': '23:00'
            },
            'timezone': 'Europe/Moscow',  # UTC+3
            'model_path': 'trained_model.pkl',
            'min_confidence': 0.5,
            'risk_per_trade': 1.0,  # % del capital por trade
            'data_lookback': 1440  # minutos de datos históricos
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"✅ Configuración cargada desde {config_path}")
            except Exception as e:
                logger.warning(f"⚠️ Error cargando config, usando defaults: {e}")
        else:
            # Crear archivo de config por defecto
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            logger.info(f"📁 Archivo de configuración creado: {config_path}")
        
        return default_config
    
    def initialize(self) -> bool:
        """Inicializar el sistema de trading"""
        logger.info("🚀 Inicializando sistema de trading en vivo...")
        
        # 1. Crear broker
        try:
            self.broker = create_broker(
                broker_type=self.config['broker_type'],
                initial_balance=self.config.get('initial_balance', 100000)
            )
            
            if not self.broker.connect():
                logger.error("❌ No se pudo conectar al broker")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error creando broker: {e}")
            return False
        
        # 2. Cargar modelo entrenado
        try:
            model_path = self.config['model_path']
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.predictor = pickle.load(f)
                logger.info(f"✅ Modelo cargado desde {model_path}")
            else:
                logger.error(f"❌ Archivo de modelo no encontrado: {model_path}")
                logger.info("💡 Entrene el modelo primero ejecutando el backtesting")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            return False
        
        # 3. Obtener balance inicial
        account_info = self.broker.get_account_info()
        self.daily_stats['start_balance'] = account_info.get('balance', 0)
        
        logger.info("✅ Sistema de trading inicializado correctamente")
        logger.info(f"📊 Balance inicial: ${self.daily_stats['start_balance']:,.2f}")
        
        return True
    
    def is_trading_time(self) -> bool:
        """Verificar si está en horario de trading"""
        now = datetime.now(self.timezone)
        current_time = now.time()
        
        start_time = time.fromisoformat(self.config['trading_hours']['start'])
        end_time = time.fromisoformat(self.config['trading_hours']['end'])
        
        # Verificar día de la semana (Lunes=0, Domingo=6)
        if now.weekday() >= 5:  # Sábado o Domingo
            return False
        
        return start_time <= current_time <= end_time
    
    def get_market_data(self) -> pd.DataFrame:
        """Obtener datos de mercado para predicción"""
        try:
            symbol = self.config['symbol']
            lookback = self.config['data_lookback']
            
            # Obtener datos históricos
            data = self.broker.get_historical_data(symbol, 'M1', lookback)
            
            if len(data) < 100:
                logger.warning("⚠️ Datos insuficientes para predicción")
                return None
            
            logger.info(f"📊 Datos obtenidos: {len(data)} velas")
            return data
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo datos: {e}")
            return None
    
    def make_prediction(self, data: pd.DataFrame) -> Dict:
        """Hacer predicción usando el modelo entrenado"""
        try:
            # Usar la función de crear características del predictor original
            features_df = self.predictor.auto_optimize_processing(data)
            
            if len(features_df) == 0:
                logger.warning("⚠️ No se pudieron crear características")
                return None
            
            # Tomar la última fila (más reciente)
            latest_features = features_df.tail(1)
            X = latest_features[self.predictor.feature_columns]
            
            # Hacer predicción
            prediction = self.predictor.model.predict(X)[0]
            
            # Calcular confianza (usando probabilidad si está disponible)
            confidence = min(abs(prediction) / 0.5, 1.0)  # Normalizar a 0-1
            
            direction = 'BUY' if prediction > 0 else 'SELL'
            
            prediction_result = {
                'prediction_pct': prediction,
                'direction': direction,
                'confidence': confidence,
                'timestamp': datetime.now(),
                'features_count': len(X.columns)
            }
            
            logger.info(f"🔮 Predicción: {direction} {prediction:.4f}% (Confianza: {confidence:.2f})")
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"❌ Error en predicción: {e}")
            return None
    
    def calculate_position_size(self, prediction: Dict) -> float:
        """Calcular tamaño de posición basado en riesgo"""
        account_info = self.broker.get_account_info()
        balance = account_info.get('balance', 0)
        
        # Riesgo por trade como % del balance
        risk_amount = balance * (self.config['risk_per_trade'] / 100)
        
        # Calcular volumen basado en stop loss
        stop_loss_pct = self.config['stop_loss_pct']
        
        # Para Dow Jones: 1 punto = $5 aproximadamente
        point_value = 5.0
        current_price = self.broker.get_current_price(self.config['symbol'])['ask']
        
        # Volume = Risk Amount / (Stop Loss Points * Point Value)
        stop_loss_points = current_price * (stop_loss_pct / 100)
        volume = risk_amount / (stop_loss_points * point_value)
        
        # Redondear a volumen mínimo con máximo 1 decimal para MT5
        volume = round(max(volume, 0.1), 1)
        
        # Limitar volumen máximo
        max_volume = self.config.get('max_volume', 1.0)
        volume = min(volume, max_volume)
        
        logger.info(f"💰 Tamaño calculado: {volume} lotes (Riesgo: ${risk_amount:.2f})")
        
        return volume
    
    def open_position(self, prediction: Dict) -> bool:
        """Abrir nueva posición basada en predicción"""
        try:
            # Verificar límites
            if len(self.positions) >= self.config['max_positions']:
                logger.info("⏸️ Máximo de posiciones alcanzado")
                return False
            
            if self.daily_stats['trades'] >= self.config['max_daily_trades']:
                logger.info("⏸️ Máximo de trades diarios alcanzado")
                return False
            
            # Verificar confianza mínima
            if prediction['confidence'] < self.config['min_confidence']:
                logger.info(f"⏸️ Confianza insuficiente: {prediction['confidence']:.2f}")
                return False
            
            # Calcular parámetros de la orden
            symbol = self.config['symbol']
            direction = prediction['direction']
            volume = self.calculate_position_size(prediction)
            
            current_price = self.broker.get_current_price(symbol)
            entry_price = current_price['ask'] if direction == 'BUY' else current_price['bid']
            
            # Calcular stop loss y take profit
            sl_pct = self.config['stop_loss_pct']
            tp_pct = self.config['take_profit_pct']
            
            if direction == 'BUY':
                stop_loss = entry_price * (1 - sl_pct/100)
                take_profit = entry_price * (1 + tp_pct/100)
            else:
                stop_loss = entry_price * (1 + sl_pct/100)
                take_profit = entry_price * (1 - tp_pct/100)
            
            # Ejecutar orden
            result = self.broker.place_order(
                symbol=symbol,
                order_type=direction,
                volume=volume,
                sl=stop_loss,
                tp=take_profit
            )
            
            if result['success']:
                position_data = {
                    'id': result.get('position_id', result.get('order_id')),
                    'symbol': symbol,
                    'direction': direction,
                    'volume': volume,
                    'entry_price': result['execution_price'],
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'open_time': result['time'],
                    'prediction': prediction,
                    'target_close_time': datetime.now() + timedelta(hours=3)  # Cerrar a las 23:00
                }
                
                self.positions[position_data['id']] = position_data
                self.daily_stats['trades'] += 1
                
                logger.info(f"🎯 Posición abierta: {direction} {volume} @ {result['execution_price']}")
                logger.info(f"🛡️ SL: {stop_loss:.2f} | 🎯 TP: {take_profit:.2f}")
                
                return True
            else:
                logger.error(f"❌ Error abriendo posición: {result.get('error')}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error en open_position: {e}")
            return False
    
    def monitor_positions(self):
        """Monitorear y gestionar posiciones abiertas"""
        try:
            broker_positions = self.broker.get_positions()
            current_time = datetime.now()
            
            for pos_id, position_data in list(self.positions.items()):
                # Buscar posición en el broker
                broker_pos = next((p for p in broker_positions if p['id'] == pos_id), None)
                
                if not broker_pos:
                    # Posición ya cerrada
                    logger.info(f"🔒 Posición {pos_id} cerrada automáticamente")
                    
                    # Actualizar estadísticas (asumiendo que se cerró en break-even si no tenemos datos)
                    self.daily_stats['losses'] += 1  # Conservador
                    del self.positions[pos_id]
                    continue
                
                # Verificar si debe cerrarse por horario
                if current_time >= position_data['target_close_time']:
                    logger.info(f"⏰ Cerrando posición {pos_id} por horario (23:00)")
                    
                    if self.broker.close_position(pos_id):
                        final_profit = broker_pos.get('profit', 0)
                        self.daily_stats['pnl'] += final_profit
                        
                        if final_profit > 0:
                            self.daily_stats['wins'] += 1
                        else:
                            self.daily_stats['losses'] += 1
                        
                        del self.positions[pos_id]
                
                # Log del estado actual
                current_profit = broker_pos.get('profit', 0)
                logger.info(f"📊 Posición {pos_id}: P&L = ${current_profit:.2f}")
                
        except Exception as e:
            logger.error(f"❌ Error monitoreando posiciones: {e}")
    
    def trading_cycle(self):
        """Ciclo principal de trading"""
        try:
            logger.info("🔄 Iniciando ciclo de trading...")
            
            # Verificar horario de trading
            if not self.is_trading_time():
                logger.info("⏸️ Fuera de horario de trading")
                return
            
            # Verificar conexión
            if not self.broker.is_connected():
                logger.error("❌ Broker desconectado")
                return
            
            # Monitorear posiciones existentes
            self.monitor_positions()
            
            # Solo abrir nuevas posiciones al inicio del horario de trading
            now = datetime.now(self.timezone)
            start_time = time.fromisoformat(self.config['trading_hours']['start'])
            if now.hour == start_time.hour and now.minute == start_time.minute:
                
                # Obtener datos de mercado
                market_data = self.get_market_data()
                if market_data is None:
                    return
                
                # Hacer predicción
                prediction = self.make_prediction(market_data)
                if prediction is None:
                    return
                
                # Abrir posición si es apropiado
                self.open_position(prediction)
            
            # Mostrar resumen diario
            self.log_daily_summary()
            
        except Exception as e:
            logger.error(f"❌ Error en ciclo de trading: {e}")
    
    def log_daily_summary(self):
        """Mostrar resumen del día"""
        account_info = self.broker.get_account_info()
        current_balance = account_info.get('balance', 0)
        equity = account_info.get('equity', 0)
        
        daily_pnl = current_balance - self.daily_stats['start_balance']
        win_rate = (self.daily_stats['wins'] / max(self.daily_stats['trades'], 1)) * 100
        
        logger.info("📈 RESUMEN DIARIO:")
        logger.info(f"  Balance: ${current_balance:,.2f} | Equity: ${equity:,.2f}")
        logger.info(f"  P&L Diario: ${daily_pnl:,.2f}")
        logger.info(f"  Trades: {self.daily_stats['trades']} | Wins: {self.daily_stats['wins']} | Win Rate: {win_rate:.1f}%")
        logger.info(f"  Posiciones Abiertas: {len(self.positions)}")
    
    def reset_daily_stats(self):
        """Resetear estadísticas diarias"""
        account_info = self.broker.get_account_info()
        self.daily_stats = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'pnl': 0.0,
            'start_balance': account_info.get('balance', 0)
        }
        logger.info("🔄 Estadísticas diarias reseteadas")
    
    def run(self):
        """Ejecutar sistema de trading"""
        if not self.initialize():
            logger.error("❌ No se pudo inicializar el sistema")
            return
        
        logger.info("🚀 Sistema de trading iniciado")
        logger.info(f"📊 Símbolo: {self.config['symbol']}")
        logger.info(f"⏰ Horario: {self.config['trading_hours']['start']} - {self.config['trading_hours']['end']}")
        
        # Programar tareas
        schedule.every().minute.do(self.trading_cycle)
        schedule.every().day.at("00:01").do(self.reset_daily_stats)
        
        self.is_running = True
        
        try:
            while self.is_running:
                schedule.run_pending()
                time_module.sleep(30)  # Verificar cada 30 segundos
                
        except KeyboardInterrupt:
            logger.info("🛑 Sistema detenido por usuario")
        except Exception as e:
            logger.error(f"❌ Error en sistema: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Detener sistema de trading"""
        logger.info("🛑 Deteniendo sistema de trading...")
        
        self.is_running = False
        
        # Cerrar posiciones abiertas
        for pos_id in list(self.positions.keys()):
            logger.info(f"🔒 Cerrando posición {pos_id}...")
            self.broker.close_position(pos_id)
        
        # Desconectar broker
        if self.broker:
            self.broker.disconnect()
        
        logger.info("✅ Sistema de trading detenido")

# Funciones de utilidad
def save_trained_model(predictor: DowJonesPredictor, path: str = "trained_model.pkl"):
    """Guardar modelo entrenado para uso en vivo"""
    try:
        with open(path, 'wb') as f:
            pickle.dump(predictor, f)
        logger.info(f"✅ Modelo guardado en {path}")
        return True
    except Exception as e:
        logger.error(f"❌ Error guardando modelo: {e}")
        return False

def create_default_config(path: str = "trading_config.json"):
    """Crear archivo de configuración por defecto"""
    config = {
        "broker_type": "simulator",
        "symbol": "US30",
        "volume": 0.1,
        "max_positions": 1,
        "max_daily_trades": 5,
        "stop_loss_pct": 2.0,
        "take_profit_pct": 3.0,
        "trading_hours": {
            "start": "20:00",
            "end": "23:00"
        },
        "timezone": "Europe/Moscow",  # UTC+3
        "model_path": "trained_model.pkl",
        "min_confidence": 0.5,
        "risk_per_trade": 1.0,
        "data_lookback": 1440,
        "initial_balance": 100000
    }
    
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"✅ Configuración creada en {path}")
    return config

def configure_timezone(timezone_str: str = "Europe/Moscow"):
    """Configurar zona horaria para el trading"""
    timezone_map = {
        'utc+3': 'Europe/Moscow',
        'moscow': 'Europe/Moscow', 
        'eastern': 'US/Eastern',
        'utc': 'UTC',
        'london': 'Europe/London',
        'tokyo': 'Asia/Tokyo',
        'sydney': 'Australia/Sydney'
    }
    
    # Permitir usar nombres comunes
    tz = timezone_map.get(timezone_str.lower(), timezone_str)
    
    try:
        pytz.timezone(tz)
        print(f"✅ Zona horaria configurada: {tz}")
        return tz
    except pytz.exceptions.UnknownTimeZoneError:
        print(f"❌ Zona horaria desconocida: {timezone_str}")
        print(f"💡 Usando por defecto: Europe/Moscow (UTC+3)")
        return 'Europe/Moscow'

if __name__ == "__main__":
    # Crear sistema de trading
    system = LiveTradingSystem()
    system.run() 