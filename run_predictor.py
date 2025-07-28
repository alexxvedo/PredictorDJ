#!/usr/bin/env python3
"""
Script principal para ejecutar el Predictor Dow Jones - OPTIMIZADO
"""

import sys
import os
import argparse
from src.predictor_dj.main import DowJonesPredictor, main_optimized

def main():
    parser = argparse.ArgumentParser(description='Predictor Dow Jones con XGBoost')
    parser.add_argument('csv_path', help='Ruta al archivo CSV con datos del Dow Jones')
    parser.add_argument('--max-samples', type=int, default=None, 
                       help='Número máximo de muestras a procesar (para datasets grandes)')
    parser.add_argument('--chunk-size', type=int, default=10000,
                       help='Tamaño de chunk para procesamiento (default: 10000)')
    parser.add_argument('--fast', action='store_true',
                       help='Modo rápido: limita a 50k muestras automáticamente')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        print(f"Error: El archivo {args.csv_path} no existe")
        sys.exit(1)
    
    # Configurar parámetros según el modo
    if args.fast:
        max_samples = 50000
        chunk_size = 5000
        print("🚀 MODO RÁPIDO activado (máx. 50k muestras)")
    else:
        max_samples = args.max_samples  # Respetar configuración del usuario (None por defecto)
        chunk_size = args.chunk_size
    
    print(f"Ejecutando predictor con datos de: {args.csv_path}")
    print(f"Configuración: max_samples={max_samples}, chunk_size={chunk_size}")
    print("-" * 60)
    
    # Crear y ejecutar predictor con optimizaciones
    try:
        predictor = DowJonesPredictor(max_samples=max_samples, chunk_size=chunk_size)
        results = predictor.run_complete_analysis(args.csv_path)
        
        if results:
            print("\n" + "="*60)
            print("✅ ¡Análisis completado exitosamente!")
            print("📊 Los resultados han sido guardados y visualizados.")
            print("📁 Archivos generados:")
            print("  • backtest_detallado.csv - CSV con todas las operaciones")
            print("  • resultados_predictor_dj_completo.png - Gráficos avanzados")
            print("="*60)
        else:
            print("\n❌ Error durante el análisis.")
            
    except KeyboardInterrupt:
        print("\n⏹️  Proceso interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 