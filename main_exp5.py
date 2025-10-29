# main_final.py
import pandas as pd
import os
import datetime
import logging

from src.loader import cargar_datos, convertir_clase_ternaria_a_target, convertir_clase_ternaria_a_target_peso
from src.features import feature_engineering_lag, feature_engineering_percentil, feature_engineering_min_ultimos_n_meses, feature_engineering_max_ultimos_n_meses, feature_engineering
from src.optimization import optimizar, evaluar_en_test, guardar_resultados_test, evaluar_en_test_pesos, optimizar_con_seed_pesos
from src.optimization_cv import optimizar_con_cv, optimizar_con_cv_pesos
from src.best_params import cargar_mejores_hiperparametros
from src.final_training import preparar_datos_entrenamiento_final, generar_predicciones_finales, entrenar_modelo_final, entrenar_modelo_final_pesos, preparar_datos_entrenamiento_final_pesos, entrenar_modelo_final_p_seeds, generar_predicciones_finales_seeds
from src.output_manager import guardar_predicciones_finales
from src.best_params import obtener_estadisticas_optuna
from src.config import *

## config basico logging
os.makedirs("logs", exist_ok=True)

fecha = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
monbre_log = f"log_{conf.STUDY_NAME}_{fecha}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/{monbre_log}", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


## Funcion principal
def main():
    
    logger.info("Inicio de ejecucion.")
   
    logger.info(f"Número de trials por estudio: {conf.parametros_lgb.n_trial}")

    #00 Cargar datos
    os.makedirs("data", exist_ok=True)
    df = cargar_datos(DATA_PATH)   

    #01 Feature Engineering

    fe_path = f"data/df_fe_{conf.STUDY_NAME}.csv"
    if os.path.exists(fe_path):
        logger.info(f"Archivo de features encontrado: {fe_path}. Cargando desde disco.")
        df_fe = pd.read_csv(fe_path)
    else:
        logger.info("Archivo de features no encontrado. Ejecutando feature engineering.")
        df_fe = feature_engineering(df, competencia="competencia01")
        df_fe.to_csv(fe_path, index=False)
   
    logger.info(f"Feature Engineering completado: {df_fe.shape}")

    #02 Convertir clase_ternaria a target binario
    
    df_fe = convertir_clase_ternaria_a_target_peso(df_fe)  
  
    #03 Ejecutar optimizacion de hiperparametros
    
    study = optimizar_con_seed_pesos(df_fe, n_trials=conf.parametros_lgb.n_trial)

    # #04 Análisis adicional
    logger.info("=== ANÁLISIS DE RESULTADOS ===")
    trials_df = study.trials_dataframe()
    if len(trials_df) > 0:
        top_5 = trials_df.nlargest(5, 'value')
        logger.info("Top 5 mejores trials:")
        for idx, trial in top_5.iterrows():
            logger.info(f"  Trial {trial['number']}: {trial['value']:,.0f}")
  
    logger.info("=== OPTIMIZACIÓN COMPLETADA ===")
  
    #05 Test en mes desconocido
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
  
    # Cargar mejores hiperparámetros
    mejores_params = cargar_mejores_hiperparametros()
  
    # Evaluar en test
    resultados_test = evaluar_en_test_pesos(df_fe, mejores_params, SEMILLA[0])
  
    # Guardar resultados de test
    guardar_resultados_test(resultados_test)
  
    # Resumen de evaluación en test
    logger.info("=== RESUMEN DE EVALUACIÓN EN TEST ===")
    logger.info(f"✅ Ganancia en test: {resultados_test['ganancia_test']:,.0f}")
    logger.info(f"🔍 Umbral óptimo encontrado: {resultados_test['umbral_optimo']:.4f}")
    logger.info(f"🎯 Predicciones positivas: {resultados_test['predicciones_positivas']:,} ({resultados_test['porcentaje_positivas']:.2f}%)")


    #06 Entrenar modelo final
    logger.info("=== ENTRENAMIENTO FINAL ===")
    logger.info("Preparar datos para entrenamiento final")
 
    X_train, y_train, pesos_train, X_predict, clientes_predict = preparar_datos_entrenamiento_final_pesos(df_fe)

    # Entrenar modelo final
    logger.info("Entrenar modelo final")
    modelos_finales = entrenar_modelo_final_p_seeds(X_train, y_train, pesos_train, mejores_params)

    # Generar predicciones finales
    logger.info("Generar predicciones finales")
    resultados = generar_predicciones_finales_seeds(modelos_finales, X_predict, clientes_predict, 0.08093)
  
    # Guardar predicciones
    logger.info("Guardar predicciones")
    archivo_salida = guardar_predicciones_finales(resultados)
  
    # Resumen final
    logger.info("=== RESUMEN FINAL ===")
    logger.info(f"✅ Entrenamiento final completado exitosamente")
    logger.info(f"📊 Mejores hiperparámetros utilizados: {mejores_params}")
    logger.info(f"🎯 Períodos de entrenamiento: {FINAL_TRAIN}")
    logger.info(f"🔮 Período de predicción: {FINAL_PREDIC}")

    ## Sumar cantidad de features utilizadas, feature importance y cantidad de clientes predichos
    logger.info(f"📁 Archivo de salida: {archivo_salida}")
    logger.info(f"📝 Log detallado: logs/{monbre_log}")


    logger.info(f">>> Ejecución finalizada. Revisar logs para mas detalles.")

if __name__ == "__main__":
    main()