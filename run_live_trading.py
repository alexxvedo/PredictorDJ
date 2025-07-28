#!/usr/bin/env python3
"""
Script principal para ejecutar el Trading en Vivo con Predictor Dow Jones
"""

import sys
import os
import argparse
import json
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Sistema de Trading en Vivo - Predictor Dow Jones')
    parser.add_argument('--config', default='trading_config.json', 
                       help='Archivo de configuración (default: trading_config.json)')
    parser.add_argument('--broker', choices=['simulator', 'mt5'], default='simulator',
                       help='Tipo de broker a usar')
    parser.add_argument('--train-first', action='store_true',
                       help='Entrenar modelo antes de iniciar trading')
    parser.add_argument('--create-config', action='store_true',
                       help='Crear archivo de configuración por defecto')
    parser.add_argument('--symbol', default='US30',
                       help='Símbolo a tradear (default: US30)')
    parser.add_argument('--balance', type=float, default=100000,
                       help='Balance inicial para simulador (default: 100000)')
    parser.add_argument('--timezone', default='Europe/Moscow',
                       help='Zona horaria para trading (default: Europe/Moscow UTC+3)')
    
    args = parser.parse_args()
    
    # Cambiar al directorio del proyecto para imports
    project_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_dir)
    
    try:
        from src.predictor_dj.live_trading import LiveTradingSystem, create_default_config, save_trained_model
        from src.predictor_dj.main import DowJonesPredictor
    except ImportError as e:
        print(f"❌ Error importando módulos: {e}")
        print("💡 Asegúrese de estar en el directorio del proyecto")
        sys.exit(1)
    
    print("🚀 PREDICTOR DOW JONES - SISTEMA DE TRADING EN VIVO")
    print("=" * 60)
    
    # Crear configuración si se solicita
    if args.create_config:
        config = create_default_config(args.config)
        config['broker_type'] = args.broker
        config['symbol'] = args.symbol
        config['initial_balance'] = args.balance
        config['timezone'] = args.timezone
        
        with open(args.config, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"✅ Configuración creada en {args.config}")
        print(f"🕐 Zona horaria: {args.timezone}")
        print("💡 Edite el archivo para personalizar la configuración")
        return
    
    # Entrenar modelo si se solicita
    if args.train_first:
        print("🔄 Entrenando modelo primero...")
        
        # Buscar archivo de datos
        data_files = [
            'data/2025.7.8DJ_M1_dukas_M1_UTCPlus03_No Session.csv',
            'datos_dow_jones.csv',
            'dow_jones_data.csv'
        ]
        
        data_file = None
        for file in data_files:
            if os.path.exists(file):
                data_file = file
                break
        
        if not data_file:
            print("❌ No se encontró archivo de datos históricos")
            print("💡 Coloque el archivo CSV en el directorio del proyecto")
            print("💡 O ejecute el backtesting primero para entrenar el modelo")
            return
        
        try:
            print(f"📊 Usando datos de: {data_file}")
            predictor = DowJonesPredictor()
            results = predictor.run_complete_analysis(data_file)
            
            if results:
                # Guardar modelo entrenado
                model_path = args.config.replace('.json', '_model.pkl') if args.config != 'trading_config.json' else 'trained_model.pkl'
                if save_trained_model(predictor, model_path):
                    print(f"✅ Modelo guardado en {model_path}")
                else:
                    print("❌ Error guardando modelo")
                    return
            else:
                print("❌ Error entrenando modelo")
                return
                
        except Exception as e:
            print(f"❌ Error durante entrenamiento: {e}")
            return
    
    # Verificar que existe el modelo
    config_path = args.config
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        model_path = config.get('model_path', 'trained_model.pkl')
    else:
        model_path = 'trained_model.pkl'
    
    if not os.path.exists(model_path):
        print(f"❌ Modelo no encontrado: {model_path}")
        print("💡 Use --train-first para entrenar el modelo primero")
        print("💡 O ejecute el backtesting para generar el modelo")
        return
    
    # Mostrar información del sistema
    print(f"📈 Tipo de Broker: {args.broker.upper()}")
    print(f"📊 Símbolo: {args.symbol}")
    print(f"🕐 Zona Horaria: {args.timezone}")
    print(f"⚙️ Configuración: {config_path}")
    print(f"🤖 Modelo: {model_path}")
    
    if args.broker == 'simulator':
        print(f"💰 Balance inicial: ${args.balance:,.2f}")
        print("⚠️ MODO SIMULADOR - No se ejecutarán operaciones reales")
    else:
        print("⚠️ MODO REAL - Se ejecutarán operaciones reales en MT5")
        print("⚠️ Asegúrese de tener MT5 abierto y configurado")
    
    print("\n🕒 Horario de trading: 20:00 - 23:00 UTC+3 (días laborables)")
    print("🎯 El sistema hará 1 predicción por día a las 20:00 UTC+3")
    print("🔄 Las posiciones se cerrarán automáticamente a las 23:00 UTC+3")
    
    # Confirmar inicio
    try:
        confirm = input("\n¿Iniciar sistema de trading? (s/N): ").lower().strip()
        if confirm not in ['s', 'si', 'y', 'yes']:
            print("❌ Operación cancelada")
            return
    except KeyboardInterrupt:
        print("\n❌ Operación cancelada")
        return
    
    # Iniciar sistema de trading
    try:
        print("\n🚀 Iniciando sistema de trading...")
        print("💡 Presione Ctrl+C para detener el sistema")
        print("-" * 60)
        
        system = LiveTradingSystem(config_path)
        system.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Sistema detenido por usuario")
    except Exception as e:
        print(f"\n❌ Error en sistema: {e}")
        import traceback
        traceback.print_exc()

def show_status():
    """Mostrar estado del sistema"""
    try:
        from src.predictor_dj.live_trading import LiveTradingSystem
        
        print("📊 ESTADO DEL SISTEMA DE TRADING")
        print("=" * 40)
        
        # Verificar archivos necesarios
        files_to_check = [
            ('trading_config.json', 'Configuración'),
            ('trained_model.pkl', 'Modelo entrenado'),
            ('live_trading.log', 'Log de trading')
        ]
        
        for file, desc in files_to_check:
            if os.path.exists(file):
                size = os.path.getsize(file)
                mod_time = datetime.fromtimestamp(os.path.getmtime(file))
                print(f"✅ {desc}: {file} ({size} bytes, {mod_time.strftime('%Y-%m-%d %H:%M')})")
            else:
                print(f"❌ {desc}: {file} (no encontrado)")
        
        # Verificar configuración
        if os.path.exists('trading_config.json'):
            with open('trading_config.json', 'r') as f:
                config = json.load(f)
            
            print(f"\n⚙️ CONFIGURACIÓN ACTUAL:")
            print(f"  Broker: {config.get('broker_type', 'N/A')}")
            print(f"  Símbolo: {config.get('symbol', 'N/A')}")
            print(f"  Volumen: {config.get('volume', 'N/A')}")
            print(f"  Stop Loss: {config.get('stop_loss_pct', 'N/A')}%")
            print(f"  Take Profit: {config.get('take_profit_pct', 'N/A')}%")
            print(f"  Riesgo por trade: {config.get('risk_per_trade', 'N/A')}%")
        
    except Exception as e:
        print(f"❌ Error verificando estado: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'status':
        show_status()
    else:
        main() 