# main_final.py
import pandas as pd
import polars as pl
import os
import datetime
import logging

#from src.loader import cargar_datos, convertir_clase_ternaria_a_target, convertir_clase_ternaria_a_target_peso
from src.loader_p import cargar_datos, convertir_clase_ternaria_a_target, convertir_clase_ternaria_a_target_peso

#from src.features import feature_engineering_lag, feature_engineering_percentil, feature_engineering_min_ultimos_n_meses, feature_engineering_max_ultimos_n_meses, feature_engineering
from src.features_p import feature_engineering_lag, feature_engineering_percentil, feature_engineering_min_ultimos_n_meses, feature_engineering_max_ultimos_n_meses, feature_engineering
#from src.optimization import optimizar, evaluar_en_test, guardar_resultados_test, evaluar_en_test_pesos, optimizar_con_seed_pesos, optimizar
from src.optimization_p import optimizar, evaluar_en_test, guardar_resultados_test, evaluar_en_test_pesos, optimizar_con_seed_pesos, optimizar
from src.optimization_cv import optimizar_con_cv, optimizar_con_cv_pesos
from src.best_params import cargar_mejores_hiperparametros
#from src.final_training import preparar_datos_entrenamiento_final, generar_predicciones_finales, entrenar_modelo_final, entrenar_modelo_final_pesos, preparar_datos_entrenamiento_final_pesos, entrenar_modelo_final_p_seeds, generar_predicciones_finales_seeds
from src.final_training_p import preparar_datos_entrenamiento_final, generar_predicciones_finales, entrenar_modelo_final, entrenar_modelo_final_pesos, preparar_datos_entrenamiento_final_pesos, entrenar_modelo_final_p_seeds, generar_predicciones_finales_seeds
#from src.output_manager import guardar_predicciones_finales
from src.output_manager_p import guardar_predicciones_finales
from src.best_params import obtener_estadisticas_optuna
from src.config import *
#from src.bucket_utils import guardar_en_buckets, cargar_de_buckets, archivo_existe_en_bucket
from src.bucket_utils_p import guardar_en_buckets, cargar_de_buckets, archivo_existe_en_bucket
from src.target import crear_clase_ternaria_gcs

## config basico logging
os.makedirs(f"{conf.BUCKET_NAME}/logs", exist_ok=True)

fecha = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
monbre_log = f"log_{conf.STUDY_NAME}_{fecha}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(f"{conf.BUCKET_NAME}/logs/{monbre_log}", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


## Funcion principal
def main():
    
    logger.info("Inicio de ejecucion.")
   
    logger.info(f"Número de trials por estudio: {conf.parametros_lgb.n_trial}")
    
    data_path_raw_gcs = f"{conf.GCS_BUCKET_URI}/{DATA_PATH_RAW}" # type: ignore
    data_path_gcs = f"{conf.GCS_BUCKET_URI}/{DATA_PATH}"
    
    # -1 Crear clase_ternaria en GCS si no existe
     
    crear_clase_ternaria_gcs(data_path_raw_gcs, data_path_gcs)

    #print(f"Ruta GCS del dataset: {data_path_gcs}")

    # 0B. Carga de datos desde GCS usando Polars
    logger.info(f"Cargando datos desde GCS: {data_path_gcs}")
    df = cargar_datos(data_path_gcs)

    #01 Feature Engineering

    # 1. Definimos la ruta de GCS y usamos .parquet
    gcs_fe_path = f"{conf.GCS_BUCKET_URI}/data/df_fe_{conf.STUDY_NAME}.parquet"

    # 2. Usamos la nueva función para verificar si existe
    if archivo_existe_en_bucket(gcs_fe_path):
        logger.info(f"Archivo de features encontrado: {gcs_fe_path}. Cargando desde GCS.")
        # 3. Usamos la nueva función para cargar
        df_fe = cargar_de_buckets(gcs_fe_path)
    
    else:
        logger.info("Archivo de features no encontrado. Ejecutando feature engineering.")
        # (Esto asume que 'df' y 'feature_engineering' existen)
        df_fe = feature_engineering(df, competencia="competencia01")
        
        # 4. Usamos la nueva función para guardar
        logger.info(f"Guardando features en: {gcs_fe_path}")
        guardar_en_buckets(df_fe, gcs_fe_path)

    logger.info(f"Feature Engineering completado: {df_fe.height, df_fe.width}")  

    #02 Convertir clase_ternaria a target binario
    
    df_fe = convertir_clase_ternaria_a_target_peso(df_fe)  
  
    #03 Ejecutar optimizacion de hiperparametros
    
    # study = optimizar_con_cv_pesos(df_fe, n_trials=conf.parametros_lgb.n_trial)
    
    study = optimizar(df_fe, n_trials=conf.parametros_lgb.n_trial, undersampling=0.05)

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
    logger.info(f"📝 Log detallado: {conf.BUCKET_NAME}/logs/{monbre_log}")


    logger.info(f">>> Ejecución finalizada. Revisar logs para mas detalles.")

if __name__ == "__main__":
    main()