"""
Interfaces y implementaciones para conectarse a diferentes brokers
Incluye simulador para Linux y MT5 real para Windows
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Tuple
import json
import os

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BrokerInterface(ABC):
    """Interfaz abstracta para brokers"""
    
    @abstractmethod
    def connect(self) -> bool:
        """Conectar al broker"""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Desconectar del broker"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Verificar si está conectado"""
        pass
    
    @abstractmethod
    def get_historical_data(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        """Obtener datos históricos"""
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> Dict[str, float]:
        """Obtener precio actual (bid/ask)"""
        pass
    
    @abstractmethod
    def place_order(self, symbol: str, order_type: str, volume: float, 
                   price: float = None, sl: float = None, tp: float = None) -> Dict:
        """Ejecutar orden"""
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Dict]:
        """Obtener posiciones abiertas"""
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict:
        """Obtener información de la cuenta"""
        pass
    
    @abstractmethod
    def close_position(self, position_id: int) -> bool:
        """Cerrar posición específica"""
        pass

class SimulatedBroker(BrokerInterface):
    """Broker simulado para pruebas y desarrollo en Linux"""
    
    def __init__(self, initial_balance: float = 100000, spread: float = 2.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.spread = spread  # spread en puntos
        self.connected = False
        self.positions = []
        self.position_counter = 1
        self.current_prices = {}
        
        # Simular datos históricos (normalmente vendrían de un feed real)
        self.historical_data_cache = {}
        
    def connect(self) -> bool:
        """Simular conexión"""
        logger.info("🔄 Conectando al broker simulado...")
        time.sleep(1)  # Simular latencia
        self.connected = True
        logger.info("✅ Conectado exitosamente al broker simulado")
        return True
    
    def disconnect(self) -> bool:
        """Simular desconexión"""
        logger.info("🔄 Desconectando del broker simulado...")
        self.connected = False
        logger.info("✅ Desconectado del broker simulado")
        return True
    
    def is_connected(self) -> bool:
        return self.connected
    
    def get_historical_data(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        """Simular datos históricos del Dow Jones"""
        logger.info(f"📊 Obteniendo {count} velas {timeframe} para {symbol}")
        
        # Generar datos simulados realistas
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=count)
        
        dates = pd.date_range(start=start_time, end=end_time, freq='1min')[:count]
        
        # Simular precios del Dow Jones (alrededor de 34000-35000)
        base_price = 34500.0
        np.random.seed(42)  # Para resultados reproducibles
        
        # Generar walk aleatorio con tendencia
        returns = np.random.normal(0, 0.01, count)  # 1% de volatilidad
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Crear OHLC
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            high = close * (1 + abs(np.random.normal(0, 0.002)))
            low = close * (1 - abs(np.random.normal(0, 0.002)))
            open_price = prices[i-1] if i > 0 else close
            volume = np.random.randint(1000, 5000)
            
            data.append({
                'datetime': date,
                'open': open_price,
                'high': max(open_price, high, close),
                'low': min(open_price, low, close),
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        logger.info(f"✅ Datos históricos obtenidos: {len(df)} registros")
        return df
    
    def get_current_price(self, symbol: str) -> Dict[str, float]:
        """Simular precio actual"""
        # Generar precio simulado
        base_price = 34500.0 + np.random.normal(0, 50)  # Variación alrededor de 34500
        spread_points = self.spread
        
        bid = base_price - spread_points/2
        ask = base_price + spread_points/2
        
        price_data = {
            'symbol': symbol,
            'bid': round(bid, 2),
            'ask': round(ask, 2),
            'spread': spread_points,
            'time': datetime.now()
        }
        
        self.current_prices[symbol] = price_data
        return price_data
    
    def place_order(self, symbol: str, order_type: str, volume: float, 
                   price: float = None, sl: float = None, tp: float = None) -> Dict:
        """Simular ejecución de orden"""
        if not self.connected:
            return {'success': False, 'error': 'No conectado'}
        
        current_price = self.get_current_price(symbol)
        
        # Determinar precio de ejecución
        if order_type.upper() == 'BUY':
            execution_price = current_price['ask']
            position_type = 'LONG'
        elif order_type.upper() == 'SELL':
            execution_price = current_price['bid']
            position_type = 'SHORT'
        else:
            return {'success': False, 'error': 'Tipo de orden inválido'}
        
        # Calcular valor notional
        notional_value = volume * execution_price
        
        # Verificar margen suficiente
        required_margin = notional_value * 0.2  # 20% de margen (5:1 leverage)
        if required_margin > self.balance:
            return {'success': False, 'error': 'Margen insuficiente'}
        
        # Crear posición
        position = {
            'id': self.position_counter,
            'symbol': symbol,
            'type': position_type,
            'volume': volume,
            'open_price': execution_price,
            'open_time': datetime.now(),
            'sl': sl,
            'tp': tp,
            'profit': 0.0
        }
        
        self.positions.append(position)
        self.position_counter += 1
        
        logger.info(f"📈 Orden ejecutada: {order_type} {volume} {symbol} @ {execution_price}")
        
        return {
            'success': True,
            'position_id': position['id'],
            'execution_price': execution_price,
            'volume': volume,
            'time': position['open_time']
        }
    
    def get_positions(self) -> List[Dict]:
        """Obtener posiciones abiertas con P&L actualizado"""
        updated_positions = []
        
        for pos in self.positions:
            current_price = self.get_current_price(pos['symbol'])
            
            # Calcular P&L
            if pos['type'] == 'LONG':
                current_market_price = current_price['bid']  # Para cerrar LONG usamos bid
                price_diff = current_market_price - pos['open_price']
            else:  # SHORT
                current_market_price = current_price['ask']  # Para cerrar SHORT usamos ask
                price_diff = pos['open_price'] - current_market_price
            
            # P&L en dólares (asumiendo 1 punto = $5 como en el sistema original)
            profit_points = price_diff
            profit_dollars = profit_points * 5.0 * pos['volume']
            
            pos['current_price'] = current_market_price
            pos['profit'] = profit_dollars
            pos['profit_points'] = profit_points
            
            updated_positions.append(pos.copy())
        
        return updated_positions
    
    def get_account_info(self) -> Dict:
        """Información de la cuenta simulada"""
        positions = self.get_positions()
        floating_pnl = sum(pos['profit'] for pos in positions)
        
        return {
            'balance': self.balance,
            'equity': self.balance + floating_pnl,
            'margin': sum(pos['volume'] * pos['open_price'] * 0.2 for pos in positions),
            'free_margin': self.balance + floating_pnl - sum(pos['volume'] * pos['open_price'] * 0.2 for pos in positions),
            'profit': floating_pnl,
            'currency': 'USD',
            'leverage': 5
        }
    
    def close_position(self, position_id: int) -> bool:
        """Cerrar posición específica"""
        for i, pos in enumerate(self.positions):
            if pos['id'] == position_id:
                # Calcular P&L final
                positions = self.get_positions()
                position_data = next((p for p in positions if p['id'] == position_id), None)
                
                if position_data:
                    final_profit = position_data['profit']
                    self.balance += final_profit
                    
                    logger.info(f"🔒 Posición {position_id} cerrada. P&L: ${final_profit:.2f}")
                    
                    # Remover de posiciones abiertas
                    self.positions.pop(i)
                    return True
        
        return False

class MT5Broker(BrokerInterface):
    """Broker real para MetaTrader 5 (Windows solamente)"""
    
    def __init__(self, account: int = None, password: str = None, server: str = None, **kwargs):
        self.account = account
        self.password = password
        self.server = server
        self.mt5 = None
        
        # Ignorar parámetros adicionales que no aplican a MT5 (como initial_balance)
        
        try:
            import MetaTrader5 as mt5
            self.mt5 = mt5
        except ImportError:
            logger.error("❌ MetaTrader5 no disponible. Use en Windows con MT5 instalado.")
            raise ImportError("MetaTrader5 solo funciona en Windows")
    
    def connect(self) -> bool:
        """Conectar a MetaTrader 5"""
        if not self.mt5:
            return False
        
        logger.info("🔄 Conectando a MetaTrader 5...")
        
        if self.account and self.password and self.server:
            success = self.mt5.initialize(login=self.account, password=self.password, server=self.server)
        else:
            success = self.mt5.initialize()
        
        if success:
            logger.info("✅ Conectado exitosamente a MetaTrader 5")
            account_info = self.mt5.account_info()
            if account_info:
                logger.info(f"Cuenta: {account_info.login}, Balance: ${account_info.balance}")
        else:
            logger.error("❌ Error al conectar a MetaTrader 5")
        
        return success
    
    def disconnect(self) -> bool:
        """Desconectar de MetaTrader 5"""
        if self.mt5:
            self.mt5.shutdown()
            logger.info("✅ Desconectado de MetaTrader 5")
            return True
        return False
    
    def is_connected(self) -> bool:
        """Verificar conexión"""
        if not self.mt5:
            return False
        return self.mt5.terminal_info() is not None
    
    def get_historical_data(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        """Obtener datos históricos de MT5"""
        if not self.is_connected():
            raise ConnectionError("No conectado a MT5")
        
        # Mapear timeframes
        timeframe_map = {
            'M1': self.mt5.TIMEFRAME_M1,
            '1min': self.mt5.TIMEFRAME_M1,
            'M5': self.mt5.TIMEFRAME_M5,
            'M15': self.mt5.TIMEFRAME_M15,
            'H1': self.mt5.TIMEFRAME_H1,
            'D1': self.mt5.TIMEFRAME_D1
        }
        
        mt5_timeframe = timeframe_map.get(timeframe, self.mt5.TIMEFRAME_M1)
        
        # Obtener datos
        rates = self.mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
        
        if rates is None or len(rates) == 0:
            raise ValueError(f"No se pudieron obtener datos para {symbol}")
        
        # Convertir a DataFrame
        df = pd.DataFrame(rates)
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df = df[['datetime', 'open', 'high', 'low', 'close', 'tick_volume']]
        df = df.rename(columns={'tick_volume': 'volume'})
        
        logger.info(f"✅ Datos históricos MT5 obtenidos: {len(df)} registros para {symbol}")
        return df
    
    def get_current_price(self, symbol: str) -> Dict[str, float]:
        """Obtener precio actual de MT5"""
        if not self.is_connected():
            raise ConnectionError("No conectado a MT5")
        
        tick = self.mt5.symbol_info_tick(symbol)
        if tick is None:
            raise ValueError(f"No se pudo obtener precio para {symbol}")
        
        return {
            'symbol': symbol,
            'bid': tick.bid,
            'ask': tick.ask,
            'spread': tick.ask - tick.bid,
            'time': datetime.fromtimestamp(tick.time)
        }
    
    def place_order(self, symbol: str, order_type: str, volume: float, 
                   price: float = None, sl: float = None, tp: float = None) -> Dict:
        """Ejecutar orden en MT5"""
        if not self.is_connected():
            return {'success': False, 'error': 'No conectado a MT5'}
        
        # Preparar request
        if order_type.upper() == 'BUY':
            trade_type = self.mt5.ORDER_TYPE_BUY
            price = self.mt5.symbol_info_tick(symbol).ask if price is None else price
        elif order_type.upper() == 'SELL':
            trade_type = self.mt5.ORDER_TYPE_SELL
            price = self.mt5.symbol_info_tick(symbol).bid if price is None else price
        else:
            return {'success': False, 'error': 'Tipo de orden inválido'}
        
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": trade_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 234000,
            "comment": "PredictorDJ",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }
        
        # Enviar orden
        result = self.mt5.order_send(request)
        
        if result.retcode != self.mt5.TRADE_RETCODE_DONE:
            return {
                'success': False, 
                'error': f'Error en orden: {result.retcode}',
                'comment': result.comment
            }
        
        logger.info(f"📈 Orden MT5 ejecutada: {order_type} {volume} {symbol} @ {result.price}")
        
        return {
            'success': True,
            'order_id': result.order,
            'position_id': result.deal,
            'execution_price': result.price,
            'volume': result.volume,
            'time': datetime.now()
        }
    
    def get_positions(self) -> List[Dict]:
        """Obtener posiciones abiertas de MT5"""
        if not self.is_connected():
            return []
        
        positions = self.mt5.positions_get()
        if positions is None:
            return []
        
        position_list = []
        for pos in positions:
            position_list.append({
                'id': pos.ticket,
                'symbol': pos.symbol,
                'type': 'LONG' if pos.type == 0 else 'SHORT',
                'volume': pos.volume,
                'open_price': pos.price_open,
                'current_price': pos.price_current,
                'sl': pos.sl,
                'tp': pos.tp,
                'profit': pos.profit,
                'open_time': datetime.fromtimestamp(pos.time)
            })
        
        return position_list
    
    def get_account_info(self) -> Dict:
        """Información de cuenta MT5"""
        if not self.is_connected():
            return {}
        
        account = self.mt5.account_info()
        if account is None:
            return {}
        
        return {
            'balance': account.balance,
            'equity': account.equity,
            'margin': account.margin,
            'free_margin': account.margin_free,
            'profit': account.profit,
            'currency': account.currency,
            'leverage': account.leverage
        }
    
    def close_position(self, position_id: int) -> bool:
        """Cerrar posición en MT5"""
        if not self.is_connected():
            return False
        
        # Obtener información de la posición
        positions = self.mt5.positions_get(ticket=position_id)
        if not positions:
            return False
        
        position = positions[0]
        
        # Preparar orden de cierre
        if position.type == 0:  # LONG
            trade_type = self.mt5.ORDER_TYPE_SELL
            price = self.mt5.symbol_info_tick(position.symbol).bid
        else:  # SHORT
            trade_type = self.mt5.ORDER_TYPE_BUY
            price = self.mt5.symbol_info_tick(position.symbol).ask
        
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": trade_type,
            "position": position_id,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "PredictorDJ Close",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }
        
        result = self.mt5.order_send(request)
        
        if result.retcode == self.mt5.TRADE_RETCODE_DONE:
            logger.info(f"🔒 Posición MT5 {position_id} cerrada exitosamente")
            return True
        else:
            logger.error(f"❌ Error cerrando posición MT5: {result.retcode}")
            return False

def create_broker(broker_type: str = "simulator", **kwargs) -> BrokerInterface:
    """Factory para crear brokers"""
    if broker_type.lower() == "simulator":
        return SimulatedBroker(**kwargs)
    elif broker_type.lower() == "mt5":
        return MT5Broker(**kwargs)
    else:
        raise ValueError(f"Tipo de broker no soportado: {broker_type}") 