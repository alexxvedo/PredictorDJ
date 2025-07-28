#!/usr/bin/env python3
"""
Script auxiliar para configurar MetaTrader 5 (Solo Windows)
"""

import os
import sys
import platform

def check_mt5_installation():
    """Verificar instalación de MetaTrader 5"""
    print("🔍 Verificando instalación de MetaTrader 5...")
    
    if platform.system() != "Windows":
        print("❌ MetaTrader 5 solo funciona en Windows")
        print("💡 Use el broker simulador en Linux/Mac")
        return False
    
    # Rutas comunes de MT5
    mt5_paths = [
        r"C:\Program Files\MetaTrader 5\terminal64.exe",
        r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
        r"C:\Users\{}\AppData\Roaming\MetaQuotes\Terminal\terminal64.exe".format(os.getenv('USERNAME', ''))
    ]
    
    mt5_found = False
    for path in mt5_paths:
        if os.path.exists(path):
            print(f"✅ MetaTrader 5 encontrado en: {path}")
            mt5_found = True
            break
    
    if not mt5_found:
        print("❌ MetaTrader 5 no encontrado")
        print("💡 Descarga MT5 desde: https://www.metatrader5.com/")
        return False
    
    return True

def test_mt5_connection():
    """Probar conexión a MT5"""
    if platform.system() != "Windows":
        print("⚠️ Test de MT5 solo disponible en Windows")
        return False
    
    try:
        import MetaTrader5 as mt5
        print("✅ Librería MetaTrader5 disponible")
        
        # Intentar conexión
        if mt5.initialize():
            print("✅ Conexión a MT5 exitosa")
            
            # Obtener información de la cuenta
            account_info = mt5.account_info()
            if account_info:
                print(f"📊 Cuenta: {account_info.login}")
                print(f"💰 Balance: ${account_info.balance:,.2f}")
                print(f"🏢 Servidor: {account_info.server}")
                print(f"🔧 Apalancamiento: 1:{account_info.leverage}")
            
            # Verificar símbolo del Dow Jones
            symbols = ['US30', 'US30Cash', 'DJIA', 'DJI30']
            dj_symbol = None
            
            for symbol in symbols:
                info = mt5.symbol_info(symbol)
                if info:
                    dj_symbol = symbol
                    print(f"📈 Símbolo Dow Jones encontrado: {symbol}")
                    
                    # Obtener precio actual
                    tick = mt5.symbol_info_tick(symbol)
                    if tick:
                        print(f"💹 Precio actual: Bid={tick.bid}, Ask={tick.ask}")
                    break
            
            if not dj_symbol:
                print("⚠️ Símbolo del Dow Jones no encontrado")
                print("💡 Símbolos comunes: US30, US30Cash, DJIA")
                print("💡 Consulte con su broker el símbolo correcto")
            
            mt5.shutdown()
            return True
            
        else:
            print("❌ No se pudo conectar a MT5")
            print("💡 Asegúrese de que MT5 esté abierto")
            print("💡 Verifique que esté logueado en una cuenta")
            return False
            
    except ImportError:
        print("❌ Librería MetaTrader5 no disponible")
        print("💡 Instale con: pip install MetaTrader5")
        return False
    except Exception as e:
        print(f"❌ Error probando MT5: {e}")
        return False

def create_mt5_config():
    """Crear configuración para MT5"""
    print("\n⚙️ CONFIGURACIÓN DE MT5")
    print("=" * 40)
    
    config = {
        "broker_type": "mt5",
        "symbol": "US30",
        "volume": 1.0,
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
        "data_lookback": 1440
    }
    
    try:
        symbol = input("Símbolo del Dow Jones en su broker (default: US30): ").strip()
        if symbol:
            config["symbol"] = symbol
        
        volume = input("Volumen por trade (default: 0.01): ").strip()
        if volume:
            try:
                config["volume"] = float(volume)
            except ValueError:
                print("⚠️ Volumen inválido, usando default")
        
        risk = input("Riesgo por trade en % (default: 1.0): ").strip()
        if risk:
            try:
                config["risk_per_trade"] = float(risk)
            except ValueError:
                print("⚠️ Riesgo inválido, usando default")
        
        # Guardar configuración
        import json
        with open('trading_config_mt5.json', 'w') as f:
            json.dump(config, f, indent=4)
        
        print("✅ Configuración MT5 guardada en trading_config_mt5.json")
        return True
        
    except KeyboardInterrupt:
        print("\n❌ Configuración cancelada")
        return False

def main():
    print("🚀 CONFIGURACIÓN METATRADER 5")
    print("=" * 50)
    
    if not check_mt5_installation():
        return
    
    print("\n🔗 Probando conexión...")
    if not test_mt5_connection():
        print("\n💡 PASOS PARA CONFIGURAR MT5:")
        print("1. Abra MetaTrader 5")
        print("2. Conecte a su cuenta de trading")
        print("3. Asegúrese de que el símbolo del Dow Jones esté disponible")
        print("4. Ejecute este script nuevamente")
        return
    
    print("\n✅ MT5 configurado correctamente")
    
    create_config = input("\n¿Crear configuración para trading en vivo? (s/N): ").lower().strip()
    if create_config in ['s', 'si', 'y', 'yes']:
        create_mt5_config()
    
    print("\n🎯 PRÓXIMOS PASOS:")
    print("1. Entrenar el modelo: python run_predictor.py tu_archivo.csv")
    print("2. Iniciar trading: python run_live_trading.py --broker mt5 --config trading_config_mt5.json")
    print("\n⚠️ IMPORTANTE: Trading en vivo utiliza dinero real")
    print("💡 Pruebe primero con el simulador para familiarizarse")

if __name__ == "__main__":
    main() 