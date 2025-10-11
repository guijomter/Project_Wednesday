# main.py
import logging
from datetime import datetime
import os
import pandas as pd
from src.grafico_test import generar_reporte_visual_completo

STUDY_NAME = "lgb_optimization_cv_competencia01_i"

### Configuración de logging ###
os.makedirs("logs", exist_ok=True)
fecha = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
nombre_log = f"log_{STUDY_NAME}_graficos_{fecha}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler("logs/" + nombre_log),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Iniciando programa de graficado de resultados en test.")






### Main ###
def main():
    """Pipeline principal con optimización usando configuración YAML."""
    logger.info("=== INICIANDO OPTIMIZACIÓN CON CONFIGURACIÓN YAML ===")
  

    # # Grafico de test
    # logger.info("=== GRAFICO DE TEST ===")
    # ruta_grafico = generar_grafico_test_completo(df_fe)  ## pasarle el archivo prediciones_test_ordenadas
    # logger.info(f"✅ Gráfico generado: {ruta_grafico}")





def generar_reporte_visual_completo(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                   titulo_estudio: str = None)





























    # 1. Cargar datos
    df = cargar_datos(DATA_PATH)
  
    # 2. Feature Engineering
    atributos = ["mcuentas_saldo", "mtarjeta_visa_consumo", "cproductos"]
    cant_lag = 2
    df_fe = feature_engineering_lag(df, atributos, cant_lag)
    logger.info(f"Feature Engineering completado: {df_fe.shape}")
  
    # 3. Convertir clase_ternaria a binario
    df_fe = convertir_clase_ternaria_a_target(df_fe)
  
    # 4. Ejecutar optimización (función simple)
    study = optimizar(df_fe, n_trials=100)
  
    # 5. Análisis adicional
    logger.info("=== ANÁLISIS DE RESULTADOS ===")
    trials_df = study.trials_dataframe()
    if len(trials_df) > 0:
        top_5 = trials_df.nlargest(5, 'value')
        logger.info("Top 5 mejores trials:")
        for idx, trial in top_5.iterrows():
            logger.info(f"  Trial {trial['number']}: {trial['value']:,.0f}")
  
    logger.info("=== OPTIMIZACIÓN COMPLETADA ===")

if __name__ == "__graficar__":
    main()