import pandas as pd
import polars as pl
import os
import datetime
import logging
import json  # Importar json

# from src.loader_p import ...
from src.loader_p import cargar_datos, convertir_clase_ternaria_a_target_peso

# from src.features_p import ...
from src.features_p import run_feature_pipeline

# MODIFICADO: Eliminamos 'optimizar' y otras funciones de BO que no se usan
from src.optimization_p import guardar_resultados_test, evaluar_en_test_pesos

from src.optimization_cv import optimizar_con_cv, optimizar_con_cv_pesos
from src.optimization_ZS_p import optimizar_zero_shot  # Importamos el especialista ZS
from src.best_params import cargar_mejores_hiperparametros

# from src.final_training_p import ...
from src.final_training_p import preparar_datos_entrenamiento_final, generar_predicciones_finales, entrenar_modelo_final, entrenar_modelo_final_pesos, preparar_datos_entrenamiento_final_pesos, entrenar_modelo_final_p_seeds, generar_predicciones_finales_seeds
# from src.output_manager_p import ...
from src.output_manager_p import guardar_predicciones_finales
from src.best_params import obtener_estadisticas_optuna
from src.config import *
# from src.bucket_utils_p import ...
from src.bucket_utils_p import guardar_en_buckets, cargar_de_buckets, archivo_existe_en_bucket
from src.target import crear_clase_ternaria_gcs
from src.data_quality import data_quality_gcs

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

# --- FUNCIÓN WRAPPER DE ZERO-SHOT ---
# (Se mantiene aquí como orquestador de pipeline)
def _optimizacion_zs(df_fe: pl.DataFrame):
    """
    Ejecuta la optimización ZeroShot (FLAML) y evalúa en test.
    Adaptado para el pipeline de Polars (main_guille.py) SIN MLflow.
    """
    logger.info("=== INICIO OPTIMIZACIÓN ZERO-SHOT (FLAML) ===")
    
    zs_iter_path = os.path.join("resultados", f"{conf.STUDY_NAME}_zs_iteraciones.json")
    zs_best_path = os.path.join("resultados", f"{conf.STUDY_NAME}_zs_best_params.json")

    if os.path.exists(zs_iter_path) and os.path.exists(zs_best_path):
        logger.info("✅ Archivos ZeroShot encontrados. Cargando hiperparámetros...")
        # Usa la función genérica para cargar desde el JSON de iteraciones de ZS
        params_lightgbm = cargar_mejores_hiperparametros(archivo_json=zs_iter_path)

        with open(zs_iter_path, "r") as f:
            iteraciones = json.load(f)
        
        mejor_iteracion = max(iteraciones, key=lambda x: x["value"])
        ganancia_val = mejor_iteracion["value"]
        umbral_sugerido = mejor_iteracion.get("umbral_sugerido", 0.5)

        logger.info(f"✅ Ganancia en validación (cargada): {ganancia_val:,.0f}")
        logger.info(f"✅ Umbral sugerido (cargado): {umbral_sugerido:.4f}")

    else:
        logger.info("❌ Archivos ZeroShot no encontrados. Ejecutando búsqueda...")
        # Llama al especialista de ZS
        resultado_zs = optimizar_zero_shot(df_fe, archivo_base=conf.STUDY_NAME)

        ganancia_val = resultado_zs["ganancia_validacion"]
        umbral_sugerido = resultado_zs["umbral_sugerido"]
        params_lightgbm = resultado_zs["best_params_lightgbm"]

        logger.info(f"✅ Ganancia en validación (calculada): {ganancia_val:,.0f}")
        logger.info(f"✅ Umbral sugerido (calculado): {umbral_sugerido:.4f}")

    logger.info("=== EVALUACIÓN EN TEST (CON PARÁMETROS ZS) ===")
    
    # Reutilizamos tu función de evaluación con pesos
    resultados_test_zs = evaluar_en_test_pesos(
        df_fe, 
        params_lightgbm, 
        n_semillas=N_SEMILLERO, 
        semilla_base=SEMILLA[0], 
        undersampling=conf.parametros_lgb.undersampling
    )
    
    # Guardamos los resultados de test de ZS
    guardar_resultados_test(resultados_test_zs, sufijo_archivo="_ZS") 
    
    ganancia_test_zs = resultados_test_zs['ganancia_suavizada_test']
    
    logger.info(f"✅ Ganancia suavizada en test (ZS): {ganancia_test_zs:,.0f}")
    logger.info("=== FIN OPTIMIZACIÓN ZERO-SHOT ===")

    # Devuelve solo los parámetros, que es lo que usará el paso final
    return params_lightgbm
# --- FIN DE LA FUNCIÓN WRAPPER ---


## Funcion principal
def main():
    
    logger.info("Inicio de ejecucion.")
   
    # logger.info(f"Número de trials por estudio: {conf.parametros_lgb.n_trial}") # Comentado, n_trial era de BO

    data_path_raw_gcs = f"{conf.GCS_BUCKET_URI}/{DATA_PATH_RAW}" # type: ignore
    data_path_gcs = f"{conf.GCS_BUCKET_URI}/{DATA_PATH}"
    data_path_q_gcs = f"{conf.GCS_BUCKET_URI}/{DATA_PATH_Q}"

    #00 Crear clase_ternaria en GCS si no existe
    logger.info("=== CREACION DE CLASE TERNARIA EN GCS ===")
    crear_clase_ternaria_gcs(data_path_raw_gcs, data_path_gcs)

    ##01 Data Quality - Interpolacion de datos faltantes (meses rotos)
    logger.info("=== INICIO DE DATA QUALITY - INTERPOLACION DE DATOS FALTANTES ===")
    data_quality_gcs(
        input_bucket_path=data_path_gcs,
        output_bucket_path=data_path_q_gcs,
        yaml_config_path="data_quality.yaml"
    )

    #01-02 Feature Engineering + Target binario
    gcs_fe_path = f"{conf.GCS_BUCKET_URI}/data/df_fe_{conf.STUDY_NAME}.parquet"

    if archivo_existe_en_bucket(gcs_fe_path):
        logger.info(f"Archivo de features encontrado: {gcs_fe_path}. Cargando desde GCS.")
        df_fe = cargar_de_buckets(gcs_fe_path)
    else:
        logger.info("Archivo de features no encontrado")
        
        logger.info(f"Cargando datos desde GCS: {data_path_q_gcs}")
        df = cargar_datos(data_path_q_gcs)

        logger.info("Ejecutando feature engineering.")
        df_fe = run_feature_pipeline(df, conf.PIPELINE_STAGES)
        
        df_fe = convertir_clase_ternaria_a_target_peso(df_fe) 

        logger.info(f"Guardando features en: {gcs_fe_path}")
        guardar_en_buckets(df_fe, gcs_fe_path)

    logger.info(f"Feature Engineering completado: {df_fe.height, df_fe.width}")  
    
    # -----------------------------------------------------------------
    # BLOQUE DE OPTIMIZACIÓN BAYESIANA (03, 04, 05) ELIMINADO
    # -----------------------------------------------------------------

    # 03 Ejecutar optimizacion de hiperparametros (SOLO ZERO-SHOT)
    # La función wrapper _optimizacion_zs ahora se encarga de todo:
    # 1. Ejecuta ZS (o carga resultados)
    # 2. Evalúa en Test
    # 3. Devuelve los parámetros para el siguiente paso
    
    mejores_params = _optimizacion_zs(df_fe)
  
    # 04 Entrenar modelo final (era 06)
    logger.info("=== ENTRENAMIENTO FINAL ===")
    logger.info("Preparar datos para entrenamiento final")
 
    X_train, y_train, pesos_train, X_predict, clientes_predict = preparar_datos_entrenamiento_final_pesos(df_fe, undersampling=conf.parametros_lgb.undersampling)

    # Entrenar modelo final
    logger.info("Entrenar modelo final usando parámetros de Zero-Shot")
    modelos_finales = entrenar_modelo_final_p_seeds(X_train, y_train, pesos_train, mejores_params, n_semillas=N_SEMILLERO, semilla_base=SEMILLA[0])

    # Generar predicciones finales
    logger.info("Generar predicciones finales")
    # NOTA: Debes decidir qué umbral usar. El de ZS se loggeó en el paso anterior.
    # Aquí estoy hardcodeando el que tenías (0.08093), pero podrías 
    # querer modificar _optimizacion_zs para que también devuelva el umbral_sugerido.
    umbral_final = 0.08093 
    logger.warning(f"Usando umbral hardcodeado: {umbral_final}. Considerar usar el umbral sugerido por ZS.")
    resultados = generar_predicciones_finales_seeds(modelos_finales, X_predict, clientes_predict, umbral_final)
  
    # Guardar predicciones
    logger.info("Guardar predicciones")
    archivo_salida = guardar_predicciones_finales(resultados)
  
    # Resumen final
    logger.info("=== RESUMEN FINAL ===")
    logger.info(f"✅ Entrenamiento final completado exitosamente")
    logger.info(f"📊 Mejores hiperparámetros (ZS) utilizados: {mejores_params}")
    logger.info(f"🎯 Períodos de entrenamiento: {FINAL_TRAIN}")
    logger.info(f"🔮 Período de predicción: {FINAL_PREDIC}")

    ## Sumar cantidad de features utilizadas, feature importance y cantidad de clientes predichos
    logger.info(f"📁 Archivo de salida: {archivo_salida}")
    logger.info(f"📝 Log detallado: {conf.BUCKET_NAME}/logs/{monbre_log}")


    logger.info(f">>> Ejecución finalizada. Revisar logs para mas detalles.")

if __name__ == "__main__":
    logger.info("Script iniciado directamente.")
    try:
        main()
    except Exception as e:
        logger.critical("Ejecución fallida con una excepción no controlada.", exc_info=True)
    finally:
        logger.info("Ejecución finalizada.")