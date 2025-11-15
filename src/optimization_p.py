#src/optimization_p.py  # Versión usando Polars

import optuna
import lightgbm as lgb
import polars as pl
import numpy as np
import pandas as pd
import logging
import json
import os
import copy
from .config import *
from .gain_function import calcular_ganancia, ganancia_lgb_binary, ganancia_evaluator, lgb_gan_eval, calcular_ganancias
from datetime import timezone, timedelta, datetime

logger = logging.getLogger(__name__)

#######################################################################################################
def objetivo_ganancia(trial, df: pl.DataFrame) -> float:
    """
    Parameters:
    trial: trial de optuna
    df: pl.DataFrame con datos
    """
    # Hiperparámetros a optimizar en el modelo LightGBM
    params = {
        'objective': 'binary',
        'metric': 'None',
        'num_iterations' : trial.suggest_int('num_iterations', conf.parametros_lgb.num_iterations[0], conf.parametros_lgb.num_iterations[1]),
        'num_leaves': trial.suggest_int('num_leaves', conf.parametros_lgb.num_leaves[0], conf.parametros_lgb.num_leaves[1]),
        'learning_rate': trial.suggest_float('learn_rate', conf.parametros_lgb.learn_rate[0], conf.parametros_lgb.learn_rate[1], log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', conf.parametros_lgb.feature_fraction[0], conf.parametros_lgb.feature_fraction[1]),
        'bagging_fraction': trial.suggest_float('bagging_fraction', conf.parametros_lgb.bagging_fraction[0], conf.parametros_lgb.bagging_fraction[1]),
        'min_child_samples': trial.suggest_int('min_child_samples', conf.parametros_lgb.min_child_samples[0], conf.parametros_lgb.min_child_samples[1]),
        'max_depth': trial.suggest_int('max_depth', conf.parametros_lgb.max_depth[0], conf.parametros_lgb.max_depth[1]),
        'reg_lambda': trial.suggest_float('reg_lambda', conf.parametros_lgb.reg_lambda[0], conf.parametros_lgb.reg_lambda[1]),
        'reg_alpha': trial.suggest_float('reg_alpha', conf.parametros_lgb.reg_alpha[0], conf.parametros_lgb.reg_alpha[1]),
        'min_gain_to_split': 0.0,
        'verbosity': -1,
        'silent': True,
        'bin': 31,
        'random_state': SEMILLA[0]
    }
  
    # Preparar dataset para entrenamiento y validación con Polars
    if isinstance(MES_TRAIN, list):
        # Aseguramos que la comparación de tipos sea correcta. Si foto_mes es int y MES_TRAIN strings, hacemos cast.
        df_train = df.filter(pl.col('foto_mes').cast(pl.Utf8).is_in(MES_TRAIN))
    else:
        df_train = df.filter(pl.col('foto_mes').cast(pl.Utf8) == str(MES_TRAIN))
    
    df_val = df.filter(pl.col('foto_mes').cast(pl.Utf8) == str(MES_VALIDACION))

    # Targets a numpy
    y_train = df_train['clase_ternaria'].to_numpy()
    y_val = df_val['clase_ternaria'].to_numpy()

    # Features (excluir target) - Convertimos a pandas para máxima compatibilidad con lgb si es necesario,
    # o mantenemos en Polars si tu versión de lgb lo soporta bien.
    # Para seguridad, usamos .to_pandas() aquí, pero podrías probar sin él.
    X_train = df_train.drop(['clase_ternaria']).to_pandas()
    X_val = df_val.drop(['clase_ternaria']).to_pandas()

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params, 
        train_data,
        valid_sets=[val_data],
        feval=ganancia_lgb_binary, 
        callbacks=[lgb.early_stopping(15), lgb.log_evaluation(0)]
    )

    y_pred_proba = model.predict(X_val)
    y_pred_binary = (y_pred_proba >= UMBRAL).astype(int)

    ganancia_total = calcular_ganancia(y_val, y_pred_binary)

    guardar_iteracion(trial, ganancia_total)
    logger.info(f"Trial {trial.number}: Ganancia = {ganancia_total:,.0f}")
    return ganancia_total
   
#######################################################################################################

def guardar_iteracion(trial, ganancia, archivo_base=None):
    """
    Guarda cada iteración de la optimización en un único archivo JSON.
  
    Args:
        trial: Trial de Optuna
        ganancia: Valor de ganancia obtenido
        archivo_base: Nombre base del archivo (si es None, usa el de config.yaml)
    """
    if archivo_base is None:
        archivo_base = conf.STUDY_NAME
  
    # Nombre del archivo único para todas las iteraciones
    archivo = f"resultados/{archivo_base}_iteraciones.json"
  
    # Datos de esta iteración
    iteracion_data = {
        'trial_number': trial.number,
        'params': trial.params,
        'value': float(ganancia),
        'datetime': datetime.now().isoformat(),
        'state': 'COMPLETE',  # Si llegamos aquí, el trial se completó exitosamente
        'configuracion': {
         #   'semilla': SEMILLA,
            'mes_train': MES_TRAIN,
            'mes_validacion': MES_VALIDACION
        }
    }
  
    # Cargar datos existentes si el archivo ya existe
    if os.path.exists(archivo):
        with open(archivo, 'r') as f:
            try:
                datos_existentes = json.load(f)
                if not isinstance(datos_existentes, list):
                    datos_existentes = []
            except json.JSONDecodeError:
                datos_existentes = []
    else:
        datos_existentes = []
  
    # Agregar nueva iteración
    datos_existentes.append(iteracion_data)
  
    # Guardar todas las iteraciones en el archivo
    with open(archivo, 'w') as f:
        json.dump(datos_existentes, f, indent=2)
  
    logger.info(f"Iteración {trial.number} guardada en {archivo}")
    logger.info(f"Ganancia: {ganancia:,.0f}" + "---" + "Parámetros: {params}")

#######################################################################################################

def optimizar(df: pl.DataFrame, n_trials: int, study_name: str = None, undersampling: float = 0.01) -> optuna.Study:
    """
    Args:
        df: DataFrame de Polars con datos
        n_trials: Número de trials a ejecutar
        study_name: Nombre del estudio (si es None, usa el de config.yaml)
  
    Description:
       Ejecuta optimización bayesiana de hiperparámetros usando configuración YAML.
       Guarda cada iteración en un archivo JSON separado. 
       Pasos:
        1. Crear estudio de Optuna
        2. Ejecutar optimización
        3. Retornar estudio

    Returns:
        optuna.Study: Estudio de Optuna con resultados
    """
    study_name = conf.STUDY_NAME
    logger.info(f"Iniciando optimización con {n_trials} trials")
    logger.info(f"Configuración: TRAIN={MES_TRAIN}, VALID={MES_VALIDACION}, SEMILLA={SEMILLA}")
  
    study = crear_o_cargar_estudio(study_name, SEMILLA[0])
    trials_previos = len(study.trials)
    trials_a_ejecutar = max(0, n_trials - trials_previos)
  
    if trials_previos > 0:
        logger.info(f"🔄 Retomando desde trial {trials_previos}. A ejecutar: {trials_a_ejecutar}")
    else:
        logger.info(f"🆕 Nueva optimización: {n_trials} trials")
  
    if trials_a_ejecutar > 0:
        study.optimize(lambda trial: objetivo_ganancia_seeds(trial, df, undersampling), n_trials=trials_a_ejecutar)
        logger.info(f"🏆 Mejor ganancia: {study.best_value:,.0f}")
        logger.info(f"Mejores parámetros: {study.best_params}")
    else:
        logger.info(f"✅ Ya se completaron {n_trials} trials")
    return study


#######################################################################################################
   
def evaluar_en_test(df: pl.DataFrame, mejores_params, semilla=SEMILLA[0]) -> dict:
    """
    Evalúa el modelo con los mejores hiperparámetros en el conjunto de test.
    Solo calcula la ganancia, sin usar sklearn.
  
    Args:
        df: DataFrame de Polars con todos los datos
        mejores_params: Mejores hiperparámetros encontrados por Optuna
  
    Returns:
        dict: Resultados de la evaluación en test (ganancia + estadísticas básicas)
    """

    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    logger.info(f"Período de test: {MES_TEST}")
  
    if isinstance(MES_TRAIN, list):
        periodos_entrenamiento = MES_TRAIN + [MES_VALIDACION]
    else:
        periodos_entrenamiento = [MES_TRAIN, MES_VALIDACION]
  
    # Filtrado con Polars
    # Aseguramos casting a string si es necesario para la comparación con listas de strings
    df_train_completo = df.filter(pl.col('foto_mes').cast(pl.Utf8).is_in([str(p) for p in periodos_entrenamiento]))
    df_test = df.filter(pl.col('foto_mes').cast(pl.Utf8) == str(MES_TEST))
  
    logger.info("Entrenando modelo con mejores hiperparámetros...")
    logger.info(f'Dimensiones df_train_completo: {(df_train_completo.height, df_train_completo.width)}, Dimensiones df_test: {(df_test.height, df_test.width)}')

    # Preparar datasets (convirtiendo a pandas/numpy para lgb)
    X_train = df_train_completo.drop(['clase_ternaria']).to_pandas()
    y_train = df_train_completo['clase_ternaria'].to_numpy()
    train_data = lgb.Dataset(X_train, label=y_train)
  
    model = lgb.train(
        mejores_params,
        train_data,
        feval=ganancia_evaluator
    )

    # Predecir en test
    X_test = df_test.drop(['clase_ternaria']).to_pandas()
    y_test = df_test['clase_ternaria'].to_numpy()
    y_pred_proba = model.predict(X_test)

    # Buscar el umbral óptimo (lógica numpy, igual)
    mejor_ganancia = -np.inf
    mejor_umbral = 0.5
    y_pred_binary = np.zeros_like(y_pred_proba, dtype=int) # Inicializar

    for umbral in np.linspace(0, 1, 201):
        y_pred_bin = (y_pred_proba >= umbral).astype(int)
        ganancia = calcular_ganancia(y_test, y_pred_bin)
        if ganancia > mejor_ganancia:
            mejor_ganancia = ganancia
            mejor_umbral = umbral
            y_pred_binary = y_pred_bin

    resultados = {
        'ganancia_test': float(mejor_ganancia),
        'umbral_optimo': float(mejor_umbral),
        'total_predicciones': int(len(y_pred_binary)),
        'predicciones_positivas': int(np.sum(y_pred_binary == 1)),
        'porcentaje_positivas': float((np.sum(y_pred_binary == 1) / len(y_pred_binary)) * 100),
        'semilla': semilla
    }
    return resultados
#######################################################################################################

def guardar_resultados_test(resultados_test: dict, archivo_base=None):
    """
    Guarda los resultados de la evaluación en test en un archivo JSON.
    
    Maneja 'resultados_test' como un diccionario (clave=mes_test) y
    guarda una entrada separada en la lista del JSON por cada mes evaluado.
    
    Args:
        resultados_test: dict, 
            Diccionario donde cada clave es un mes de test (str)
            y cada valor es un dict con los resultados de ese mes.
        archivo_base: Nombre base del archivo (si es None, usa el de config.yaml)
    """
    
    if archivo_base is None:
        archivo_base = conf.STUDY_NAME
    
    # Nombre del archivo único para todas las iteraciones
    archivo = f"{conf.BUCKET_NAME}/resultados/{archivo_base}_resultado_test.json"
    
    # Cargar datos existentes si el archivo ya existe
    if os.path.exists(archivo):
        with open(archivo, 'r') as f:
            try:
                datos_existentes = json.load(f)
                if not isinstance(datos_existentes, list):
                    datos_existentes = []
            except json.JSONDecodeError:
                datos_existentes = []
    else:
        datos_existentes = []
    
    # --- Preparar datos constantes para esta ejecución ---
    tz = timezone(timedelta(hours=-3))
    # Usar el mismo timestamp para todas las entradas de esta ejecución
    timestamp_ejecucion = datetime.now(tz).isoformat() 

    if isinstance(MES_TRAIN, list):
        periodos_entrenamiento = MES_TRAIN + [MES_VALIDACION]
    else:
        periodos_entrenamiento = [MES_TRAIN, MES_VALIDACION]

    # --- Iterar sobre CADA mes en los resultados ---
    # 'resultados_test' es ahora un dict: {'mes1': {...}, 'mes2': {...}}
    
    if not isinstance(resultados_test, dict) or not resultados_test:
         logger.warning("No se recibieron resultados para guardar.")
         return

    # Iteramos sobre cada par (mes, resultados) en el diccionario
    for mes_test_actual, resultados_del_mes in resultados_test.items():
        
        # Construir la entrada para este mes específico
        iteracion_data = {
            # Usar la clave del diccionario como el mes de test
            'Mes_test': mes_test_actual, 
            'ganancia_suavizada_test': float(resultados_del_mes['ganancia_suavizada_test']),
            'ganancia_maxima_test': float(resultados_del_mes['ganancia_maxima_test']),
            'date_time': timestamp_ejecucion, # Timestamp de la ejecución
            'state': 'COMPLETE',
            'configuracion': {
                # Obtener n_semillas de los resultados de ESE mes
                'semillas': resultados_del_mes.get('n_semillas', 'N/A'), 
                'meses_train': periodos_entrenamiento
            },
            # Guardar el sub-diccionario completo de ese mes
            'resultados': resultados_del_mes 
        }

        # Agregar la iteración de ESTE MES a la lista
        datos_existentes.append(iteracion_data)
        
        # Loguear el resultado de este mes
        logger.info(f"Mes {mes_test_actual} -> Ganancia suavizada: {resultados_del_mes['ganancia_suavizada_test']:,.0f}" + "---" + f"Ganancia máxima: {resultados_del_mes['ganancia_maxima_test']:,.0f}" )

    # --- Guardar el archivo JSON (fuera del loop) ---
    
    # Obtener la ruta del directorio del archivo
    directorio_destino = os.path.dirname(archivo)
    
    # Crear el directorio y todos los directorios padres necesarios
    os.makedirs(directorio_destino, exist_ok=True)

    # Guardar todas las iteraciones (nuevas y viejas) en el archivo
    with open(archivo, 'w') as f:
        json.dump(datos_existentes, f, indent=2)
    
    logger.info(f"Resultados de {len(resultados_test)} mes(es) guardados en {archivo}")

#####################################################################################
  
# def evaluar_en_test_pesos(df: pl.DataFrame, mejores_params, semilla=SEMILLA[0], undersampling: float = 1) -> dict:
#     """
#     Evalúa el modelo con los mejores hiperparámetros en el conjunto de test.
#     Solo calcula la ganancia, sin usar sklearn.
  
#     Args:
#         df: Polars DataFrame con todos los datos
#         mejores_params: Mejores hiperparámetros encontrados por Optuna
  
#     Returns:
#         dict: Resultados de la evaluación en test (ganancia + estadísticas básicas)
#     """

#     logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
#     logger.info(f"Período de test: {MES_TEST}")
  
#     # Filtrado Polars
#     if isinstance(MES_TRAIN, list):
#         periodos = MES_TRAIN + [MES_VALIDACION]
#     else:
#         periodos = [MES_TRAIN, MES_VALIDACION]
    
#     df_train_completo = df.filter(pl.col('foto_mes').cast(pl.Utf8).is_in([str(p) for p in periodos]))
#     df_test = df.filter(pl.col('foto_mes').cast(pl.Utf8) == str(MES_TEST))
    
#     # Aplicar undersampling si es necesario
    
#     df_train_completo = aplicar_undersampling_clientes(df_train_completo, tasa=undersampling, semilla=SEMILLA[0])
    
#     # Entrenar modelo con mejores parámetros
#     logger.info("Entrenando modelo con mejores hiperparámetros...")
#     logger.info(f'Dimensiones df_train_completo: {df_train_completo.height, df_train_completo.width}, Dimensiones df_test: {df_test.height, df_test.width}')


#     # Datasets
#     X = df_train_completo.drop(['clase_ternaria', 'clase_peso']).to_pandas()
#     y = df_train_completo['clase_ternaria'].to_numpy()
#     weights = df_train_completo['clase_peso'].to_numpy()
    
#     train_data = lgb.Dataset(X, label=y, weight=weights)
#     logger.info(f"Tipo de dato de train_data: {type(train_data)}, Dimensiones de train_data: {train_data.data.shape}")
  
#     model = lgb.train(mejores_params, train_data, feval=lgb_gan_eval)
#     #guardar modelo entrenado
#     model.save_model(f'resultados/modelo_final_test_{conf.STUDY_NAME}_semilla_{semilla}.txt')
#     logger.info("Modelo guardado en resultados/")
#     logger.info("Modelo final para test entrenado. Calculando ganancia en Test...")

#     # Test
#     X_test = df_test.drop(['clase_ternaria', 'clase_peso']).to_pandas()
#     y_test = df_test['clase_ternaria'].to_numpy()
#     wetights_test = df_test['clase_peso'].to_numpy()
#     test_data = lgb.Dataset(X_test, label=y_test, weight=wetights_test)

#    # y_test_pesos = df_test['clase_peso'].to_numpy()
#    # y_test = np.where(y_test_pesos == 1.00002, 1, 0)

#     y_pred_proba = model.predict(X_test)

#     # Guardar predicciones ordenadas por probabilidad descendente (usando Polars para crear el DF y guardar CSV)
#     predicciones_test = pl.DataFrame({
#         'probabilidad': y_pred_proba,
#         'clase_ternaria': y_test
#     }).sort('probabilidad', descending=True)
    
#     predicciones_test.write_csv(f'resultados/predicciones_test_ordenadas_{conf.STUDY_NAME}_semilla_{semilla}.csv')

# #    # Buscar el umbral que maximiza la ganancia
#     # mejor_ganancia = -np.inf
#     # mejor_umbral = 0.5
#     # y_pred_binary = np.zeros_like(y_pred_proba, dtype=int)
#     # for umbral in np.linspace(0, 1, 201):
#     #     y_pred_bin = (y_pred_proba >= umbral).astype(int)
#     #     ganancia = calcular_ganancia(y_test, y_pred_bin)
#     #     if ganancia > mejor_ganancia:
#     #         mejor_ganancia = ganancia
#     #         mejor_umbral = umbral
#     #         y_pred_binary = y_pred_bin

#     ganancia_suavizada_test, ganancia_maxima_test = calcular_ganancias(y_pred_proba, test_data)

#     resultados = {
#         'ganancia_suavizada_test': float(ganancia_suavizada_test),
#         'ganancia_maxima_test': float(ganancia_maxima_test),
#       #  'umbral_optimo': float(mejor_umbral),
#       #  'total_predicciones': int(len(y_pred_binary)),
#       #  'predicciones_positivas': int(np.sum(y_pred_binary == 1)),
#       #  'porcentaje_positivas': float((np.sum(y_pred_binary == 1) / len(y_pred_binary)) * 100),
#         'semilla': semilla
#     }
#     return resultados

def evaluar_en_test_pesos(df: pl.DataFrame, mejores_params: dict, n_semillas: int, semilla_base=SEMILLA[0], undersampling: float = 1) -> dict:
    """
    Evalúa el modelo con los mejores hiperparámetros en el(los) conjunto(s) de test.
    
    Entrena N modelos con distintas semillas. Luego, para CADA mes en MES_TEST:
    1. Promedia sus predicciones.
    2. Calcula la ganancia sobre el promedio para ese mes.
    
    Args:
        df: Polars DataFrame con todos los datos
        mejores_params: Mejores hiperparámetros encontrados por Optuna
        n_semillas: int, número de modelos a entrenar y promediar
        semilla_base: int, semilla base para generar las N semillas
        undersampling: float, tasa de undersampling
    
    Returns:
        dict: Un diccionario donde cada clave es un mes de test (str)
              y cada valor es un dict con los resultados de ese mes.
    """

    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST (Promediando N Modelos) ===")

    # --- 1. Preparar Lista de Meses de Test ---
    
    # Asegurar que MES_TEST sea una lista para iterar
    if isinstance(MES_TEST, (str, int)):
        meses_test_lista = [str(MES_TEST)]
    elif isinstance(MES_TEST, list):
        meses_test_lista = [str(m) for m in MES_TEST]
    else:
        logger.error(f"MES_TEST tiene un tipo no válido: {type(MES_TEST)}")
        return {} # O lanzar un error

    logger.info(f"Períodos de test a evaluar: {meses_test_lista}")
    logger.info(f"Semilla base: {semilla_base}, N Modelos: {n_semillas}")

    # --- 2. Preparar Datos de Entrenamiento (Una sola vez) ---
    
    # Filtrado Polars
    if isinstance(MES_TRAIN, list):
        periodos = MES_TRAIN + [MES_VALIDACION]
    else:
        periodos = [MES_TRAIN, MES_VALIDACION]
    
    df_train_completo = df.filter(pl.col('foto_mes').cast(pl.Utf8).is_in([str(p) for p in periodos]))
    
    # Aplicar undersampling si es necesario (usando la semilla base)
    df_train_completo = aplicar_undersampling_clientes(df_train_completo, tasa=undersampling, semilla=semilla_base)
    
    logger.info(f'Dimensiones df_train_completo: {df_train_completo.height, df_train_completo.width}')

    # Convertir a lgb.Dataset
    X_train = df_train_completo.drop(['clase_ternaria', 'clase_peso']).to_pandas()
    y_train = df_train_completo['clase_ternaria'].to_numpy()
    weights_train = df_train_completo['clase_peso'].to_numpy()
    
    train_data = lgb.Dataset(X_train, label=y_train, weight=weights_train)
    logger.info(f"Tipo de dato de train_data: {type(train_data)}, Dimensiones de train_data: {train_data.data.shape}")


    # --- 3. Loop de Entrenamiento (N Semillas) ---
    
    logger.info(f"Iniciando entrenamiento de {n_semillas} modelos...")

    # 1. Generar N semillas aleatorias reproducibles a partir de la semilla base
    rng = np.random.RandomState(semilla_base)
    semillas_modelos = rng.randint(0, 2**32 - 1, size=n_semillas)
    
    # 2. Lista para almacenar los modelos entrenados
    modelos_entrenados = [] # Guardará tuplas (modelo, seed)

    for i, seed in enumerate(semillas_modelos):
        logger.info(f"Entrenando modelo {i+1}/{n_semillas} (Semilla: {seed})...")
        
        # 3. Copiar parámetros y asignar la semilla de esta iteración
        params_seed = copy.deepcopy(mejores_params)
        
        # Asignar semillas a parámetros de LightGBM
        params_seed['random_state'] = seed
      #  params_seed['seed'] = seed
        params_seed['bagging_seed'] = seed + 1  # Usar seeds distintas
        params_seed['feature_fraction_seed'] = seed + 2
        
        # 4. Entrenar el modelo
        model = lgb.train(params_seed, train_data, feval=lgb_gan_eval)
        
        # 5. Guardar modelo entrenado en disco
        model.save_model(f'resultados/modelo_final_test_{conf.STUDY_NAME}_semilla_{seed}.txt')

        # 6. Guardar modelo en memoria para predicción
        modelos_entrenados.append((model, seed))

    logger.info("Entrenamiento de los N modelos completado.")

    # --- 4. Loop de Evaluación (Por cada Mes de Test) ---

    todos_los_resultados = {} # Diccionario para guardar todos los resultados

    for mes_test_actual_str in meses_test_lista:
        logger.info(f"--- Evaluando en Período de Test: {mes_test_actual_str} ---")

        # 4.1. Filtrar y preparar datos de test para ESTE mes
        df_test_mes = df.filter(pl.col('foto_mes').cast(pl.Utf8) == mes_test_actual_str)
        
        if df_test_mes.height == 0:
            logger.warning(f"No se encontraron datos para el mes de test: {mes_test_actual_str}. Saltando...")
            continue

        logger.info(f'Dimensiones df_test ({mes_test_actual_str}): {df_test_mes.height, df_test_mes.width}')

        X_test_mes = df_test_mes.drop(['clase_ternaria', 'clase_peso']).to_pandas()
        y_test_mes = df_test_mes['clase_ternaria'].to_numpy()
        weights_test_mes = df_test_mes['clase_peso'].to_numpy()
        test_data_mes = lgb.Dataset(X_test_mes, label=y_test_mes, weight=weights_test_mes)

        # 4.2. Lista para almacenar las predicciones de cada modelo (para este mes)
        lista_predicciones_mes = []

        # 4.3. Predecir con cada modelo entrenado
        for model, seed in modelos_entrenados:
            logger.debug(f"Prediciendo con modelo (Semilla: {seed}) en mes {mes_test_actual_str}...")
            
            y_pred_proba_seed = model.predict(X_test_mes)
            lista_predicciones_mes.append(y_pred_proba_seed)

            # 4.4. Guardar resultados individuales (por modelo, por mes)
            df_resultados_seed = df_test_mes.select(
                'numero_de_cliente', 
                'clase_peso'
            ).with_columns(
                pl.Series('probabilidad', y_pred_proba_seed),
                pl.lit(seed).alias('semilla_modelo')
            ).sort('probabilidad', descending=True)

            # Modificar path para incluir el mes
            fname = f'resultados/predicciones_test_ordenadas_{conf.STUDY_NAME}_mes_{mes_test_actual_str}_semilla_modelo_{seed}.csv'
            df_resultados_seed.write_csv(fname)
        
        logger.info(f"Predicciones completadas para el mes {mes_test_actual_str}. Promediando...")

        # 4.5. Promediar las predicciones (para este mes)
        y_pred_proba_promedio = np.mean(lista_predicciones_mes, axis=0)
        
        # --- 4.6. Cálculo de Ganancia y Resultados (para este mes) ---

        df_resultados_promedio = df_test_mes.select(
            'numero_de_cliente', 
            'clase_peso'
        ).with_columns(
            pl.Series('probabilidad', y_pred_proba_promedio) # y_pred_proba es el array NumPy promediado
        ).sort('probabilidad', descending=True)

        # Modificar path para incluir el mes
        fname_promedio = f'resultados/predicciones_test_promediadas_{conf.STUDY_NAME}_mes_{mes_test_actual_str}_semilla_{semilla_base}_N{n_semillas}.csv'
        df_resultados_promedio.write_csv(fname_promedio)
        
        # Calcular ganancias usando la predicción promediada (para este mes)
        ganancia_suavizada_test, ganancia_maxima_test, envios_max_gan = calcular_ganancias(y_pred_proba_promedio, test_data_mes)

        resultados_mes = {
            'ganancia_suavizada_test': float(ganancia_suavizada_test),
            'ganancia_maxima_test': float(ganancia_maxima_test),
            'envios_max_gan': int(envios_max_gan),
            'porcentaje_envios_max_gan': float(envios_max_gan / len(y_test_mes)),
            'semilla_base': semilla_base,
            'n_semillas': n_semillas,
            'mes_test': mes_test_actual_str # Añadir el mes al dict de resultados
        }
        
        logger.info(f"Resultados Test Mes {mes_test_actual_str} (N={n_semillas}): Ganancia Suavizada = {ganancia_suavizada_test:,.0f}, Ganancia Máxima = {ganancia_maxima_test:,.0f}, Envios Máx Gan = {envios_max_gan}")
        
        # 4.7. Guardar resultados en el diccionario principal
        todos_los_resultados[mes_test_actual_str] = resultados_mes

    # --- Fin del Loop de Evaluación ---
    
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST COMPLETADA ===")
    return todos_los_resultados
########################################################################################################

def evaluar_en_test_colab(df: pl.DataFrame, mejores_params: dict, n_semillas: int, semilla_base=SEMILLA[0], undersampling: float = 1) -> dict:
    """
    Evalúa el modelo con los mejores hiperparámetros en el(los) conjunto(s) de test.
    Genera un Excel por mes de test con:
      - Hoja 1: Probabilidades por modelo y clase real.
      - Hoja 2: Clase real ordenada por ranking de probabilidad para cada modelo.
    """

    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST (Promediando N Modelos) ===")

    # --- 1. Preparar Lista de Meses de Test ---
    if isinstance(MES_TEST, (str, int)):
        meses_test_lista = [str(MES_TEST)]
    elif isinstance(MES_TEST, list):
        meses_test_lista = [str(m) for m in MES_TEST]
    else:
        logger.error(f"MES_TEST tiene un tipo no válido: {type(MES_TEST)}")
        return {}

    logger.info(f"Períodos de test a evaluar: {meses_test_lista}")
    logger.info(f"Semilla base: {semilla_base}, N Modelos: {n_semillas}")

    # --- 2. Preparar Datos de Entrenamiento (Una sola vez) ---
    if isinstance(MES_TRAIN, list):
        periodos = MES_TRAIN + [MES_VALIDACION]
    else:
        periodos = [MES_TRAIN, MES_VALIDACION]
    
    df_train_completo = df.filter(pl.col('foto_mes').cast(pl.Utf8).is_in([str(p) for p in periodos]))
    
    # Aplicar undersampling
    df_train_completo = aplicar_undersampling_clientes(df_train_completo, tasa=undersampling, semilla=semilla_base)
    
    # Convertir a lgb.Dataset
    X_train = df_train_completo.drop(['clase_ternaria', 'clase_peso']).to_pandas()
    y_train = df_train_completo['clase_ternaria'].to_numpy()
    weights_train = df_train_completo['clase_peso'].to_numpy()
    
    train_data = lgb.Dataset(X_train, label=y_train, weight=weights_train)

    # --- 3. Loop de Entrenamiento (N Semillas) ---
    logger.info(f"Iniciando entrenamiento de {n_semillas} modelos...")

    rng = np.random.RandomState(semilla_base)
    semillas_modelos = rng.randint(0, 2**32 - 1, size=n_semillas)
    
    modelos_entrenados = [] 

    for i, seed in enumerate(semillas_modelos):
        logger.info(f"Entrenando modelo {i+1}/{n_semillas} (Semilla: {seed})...")
        
        params_seed = copy.deepcopy(mejores_params)
        params_seed['random_state'] = seed
        params_seed['bagging_seed'] = seed + 1
        params_seed['feature_fraction_seed'] = seed + 2
        
        model = lgb.train(params_seed, train_data, feval=lgb_gan_eval)
        # Opcional: guardar modelo físico si quieres
        # model.save_model(f'resultados/modelo_..._{seed}.txt') 
        modelos_entrenados.append((model, seed))

    logger.info("Entrenamiento completado.")

    # --- 4. Loop de Evaluación (Por cada Mes de Test) ---

    todos_los_resultados = {} 

    for mes_test_actual_str in meses_test_lista:
        logger.info(f"--- Evaluando en Período de Test: {mes_test_actual_str} ---")

        # 4.1. Filtrar datos
        df_test_mes = df.filter(pl.col('foto_mes').cast(pl.Utf8) == mes_test_actual_str)
        
        if df_test_mes.height == 0:
            logger.warning(f"No hay datos para {mes_test_actual_str}")
            continue

        X_test_mes = df_test_mes.drop(['clase_ternaria', 'clase_peso']).to_pandas()
        y_test_mes = df_test_mes['clase_ternaria'].to_numpy()
        weights_test_mes = df_test_mes['clase_peso'].to_numpy()
        test_data_mes = lgb.Dataset(X_test_mes, label=y_test_mes, weight=weights_test_mes)

        # --- PREPARACIÓN DATOS EXCEL ---
        
        # Vector de Clase Real (1 si peso es 3.0 [osea 1.00002 de ganancia aprox], 0 sino)
        # Aseguramos convertir de Polars a Numpy
        clase_peso_np = df_test_mes['clase_peso'].to_numpy()
        # Asumimos que 1.00002 es el valor de ganancia alta, ajusta si tu lógica de peso es estricta 3.0
        # Aquí uso la lógica que pediste: si clase_peso > 1 (usualmente los positivos tienen peso alto) o == 3.0
        clase_real_binaria = np.where(weights_test_mes >= 3.0, 1, 0) 
        ids_clientes = df_test_mes['numero_de_cliente'].to_numpy()

        # Estructura para HOJA 1: Info Cliente + Predicciones
        data_sheet1 = {
            'numero_de_cliente': ids_clientes,
            'clase_real': clase_real_binaria
        }

        # Estructura para HOJA 2: Rankings
        data_sheet2 = {}

        lista_predicciones_mes = []

        # 4.2. Predecir con cada modelo
        for model, seed in modelos_entrenados:
            # Predecir
            y_pred_proba_seed = model.predict(X_test_mes)
            lista_predicciones_mes.append(y_pred_proba_seed)

            # A) Agregar columna a HOJA 1
            col_name = f'prob_seed_{seed}'
            data_sheet1[col_name] = y_pred_proba_seed

            # B) Calcular columna para HOJA 2 (Sort by prob desc)
            # Obtenemos los índices que ordenarían el array de mayor a menor
            indices_ordenados = np.argsort(-y_pred_proba_seed)
            
            # Reordenamos la clase real usando esos índices
            clase_real_ordenada = clase_real_binaria[indices_ordenados]
            
            # Guardamos en la estructura de la hoja 2
            data_sheet2[f'rank_seed_{seed}'] = clase_real_ordenada

        # 4.3. Crear DataFrames de Pandas
        df_sheet1 = pd.DataFrame(data_sheet1)
        df_sheet2 = pd.DataFrame(data_sheet2)

        # 4.4. Guardar Excel
        os.makedirs("resultados", exist_ok=True)
        excel_path = f'resultados/analisis_semillas_{conf.STUDY_NAME}_{mes_test_actual_str}.xlsx'
        
        logger.info(f"Guardando Excel en: {excel_path} ...")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer: # o engine='xlsxwriter'
            df_sheet1.to_excel(writer, sheet_name='Predicciones_Clientes', index=False)
            df_sheet2.to_excel(writer, sheet_name='Ranking_ClaseReal', index=False)
        
        logger.info("Excel guardado exitosamente.")

        # 4.5. Promedio y Ganancias (Lógica original)
        y_pred_proba_promedio = np.mean(lista_predicciones_mes, axis=0)
        
        ganancia_suavizada_test, ganancia_maxima_test, envios_max_gan = calcular_ganancias(y_pred_proba_promedio, test_data_mes)

        resultados_mes = {
            'ganancia_suavizada_test': float(ganancia_suavizada_test),
            'ganancia_maxima_test': float(ganancia_maxima_test),
            'envios_max_gan': int(envios_max_gan),
            'porcentaje_envios_max_gan': float(envios_max_gan / len(y_test_mes)),
            'semilla_base': semilla_base,
            'n_semillas': n_semillas,
            'mes_test': mes_test_actual_str
        }
        
        logger.info(f"Resultados Promedio Mes {mes_test_actual_str}: Ganancia Máx = {ganancia_maxima_test:,.0f}")
        todos_los_resultados[mes_test_actual_str] = resultados_mes

    logger.info("=== EVALUACIÓN COMPLETADA ===")
    return todos_los_resultados


############################################################################ OBJETIVO GANANCIA SEEDS 

def evaluar_en_test_zlgbm(df: pl.DataFrame, model: lgb.Booster) -> dict:
    """
    Evalúa un modelo YA ENTRENADO en el(los) conjunto(s) de test definidos en MES_TEST.
    
    Para CADA mes en MES_TEST:
    1. Filtra los datos.
    2. Predice usando el modelo pasado por parámetro.
    3. Calcula métricas de ganancia.
    4. Guarda las predicciones en CSV.
    
    Args:
        df: Polars DataFrame con todos los datos (incluyendo mes de test).
        model: Objeto lgb.Booster ya entrenado.
    
    Returns:
        dict: Un diccionario donde cada clave es un mes de test (str)
              y cada valor es un dict con los resultados de ese mes.
    """

    logger.info("=== EVALUACIÓN DE MODELO ÚNICO EN CONJUNTO DE TEST ===")

    # --- 1. Preparar Lista de Meses de Test ---
    # Usamos la variable global MES_TEST (o podrías pasarla como argumento si prefieres)
    if isinstance(MES_TEST, (str, int)):
        meses_test_lista = [str(MES_TEST)]
    elif isinstance(MES_TEST, list):
        meses_test_lista = [str(m) for m in MES_TEST]
    else:
        logger.error(f"MES_TEST tiene un tipo no válido: {type(MES_TEST)}")
        return {}

    logger.info(f"Períodos de test a evaluar: {meses_test_lista}")

    todos_los_resultados = {} # Diccionario para guardar todos los resultados

    # --- 2. Loop de Evaluación (Por cada Mes de Test) ---
    for mes_test_actual_str in meses_test_lista:
        logger.info(f"--- Evaluando en Período de Test: {mes_test_actual_str} ---")

        # 2.1. Filtrar y preparar datos de test para ESTE mes
        df_test_mes = df.filter(pl.col('foto_mes').cast(pl.Utf8) == mes_test_actual_str)
        
        if df_test_mes.height == 0:
            logger.warning(f"No se encontraron datos para el mes de test: {mes_test_actual_str}. Saltando...")
            continue

        # Preparamos X, y, weights
        # Importante: Asegurarse de dropear las mismas columnas que en el train
        X_test_mes = df_test_mes.drop(['clase_ternaria', 'clase_peso']).to_pandas()
        y_test_mes = df_test_mes['clase_ternaria'].to_numpy()
        weights_test_mes = df_test_mes['clase_peso'].to_numpy()
        
        # Creamos dataset de LGBM (necesario para la función calcular_ganancias)
        test_data_mes = lgb.Dataset(X_test_mes, label=y_test_mes, weight=weights_test_mes)

        logger.info(f'Dimensiones df_test ({mes_test_actual_str}): {df_test_mes.height, df_test_mes.width}')

        # 2.2. Predecir (Usando el modelo recibido)
        y_pred_proba = model.predict(X_test_mes)

        # 2.3. Guardar predicciones en CSV
        df_resultados = df_test_mes.select(
            'numero_de_cliente', 
            'clase_peso'
        ).with_columns(
            pl.Series('probabilidad', y_pred_proba)
        ).sort('probabilidad', descending=True)

        # Nombre de archivo simplificado
        fname = f'resultados/predicciones_test_{conf.STUDY_NAME}_mes_{mes_test_actual_str}_single_model.csv'
        df_resultados.write_csv(fname)
        logger.info(f"Predicciones guardadas en: {fname}")
        
        # 2.4. Calcular Ganancia y Métricas
        # Usamos la función auxiliar que ya tienes definida en tu entorno
        ganancia_suavizada_test, ganancia_maxima_test, envios_max_gan = calcular_ganancias(y_pred_proba, test_data_mes)

        resultados_mes = {
            'ganancia_suavizada_test': float(ganancia_suavizada_test),
            'ganancia_maxima_test': float(ganancia_maxima_test),
            'envios_max_gan': int(envios_max_gan),
            'porcentaje_envios_max_gan': float(envios_max_gan / len(y_test_mes)),
            'mes_test': mes_test_actual_str
        }
        
        logger.info(f"Resultados Test Mes {mes_test_actual_str}: Ganancia Máxima = {ganancia_maxima_test:,.0f} (Envíos: {envios_max_gan})")
        
        # Guardar en el diccionario principal
        todos_los_resultados[mes_test_actual_str] = resultados_mes

    logger.info("=== EVALUACIÓN COMPLETADA ===")
    return todos_los_resultados


####################################################################################################################
def objetivo_ganancia_seeds(trial: optuna.trial.Trial, df: pl.DataFrame, n_semillas: int, undersampling: float = 1) -> float:
    """
    Parameters:
    trial: trial de optuna
    df: dataframe con datos
    n_semillas: int, número de semillas aleatorias a generar y promediar
    undersampling: float, tasa de undersampling

    Description:
    Función objetivo que maximiza ganancia en mes de validación.
    Utiliza configuración YAML para períodos y semilla.
    Define parametros para el modelo LightGBM
    Preparar dataset para entrenamiento y validación
    Entrena modelo con función de ganancia personalizada
    Predecir y calcular ganancia
    Guardar cada iteración en JSON
  
    Returns:
    float: ganancia total
    """
    params = {
        'objective': 'binary', 'metric': 'None',
        'num_iterations': trial.suggest_int('num_iterations', conf.parametros_lgb.num_iterations[0], conf.parametros_lgb.num_iterations[1]),
        'num_leaves': trial.suggest_int('num_leaves', conf.parametros_lgb.num_leaves[0], conf.parametros_lgb.num_leaves[1]),
        'learning_rate': trial.suggest_float('learning_rate', conf.parametros_lgb.learning_rate[0], conf.parametros_lgb.learning_rate[1], log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', conf.parametros_lgb.feature_fraction[0], conf.parametros_lgb.feature_fraction[1]),
        'bagging_fraction': trial.suggest_float('bagging_fraction', conf.parametros_lgb.bagging_fraction[0], conf.parametros_lgb.bagging_fraction[1]),
        'min_child_samples': trial.suggest_int('min_child_samples', conf.parametros_lgb.min_child_samples[0], conf.parametros_lgb.min_child_samples[1]),
        'max_depth': trial.suggest_int('max_depth', conf.parametros_lgb.max_depth[0], conf.parametros_lgb.max_depth[1]),
        'reg_lambda': trial.suggest_float('reg_lambda', conf.parametros_lgb.reg_lambda[0], conf.parametros_lgb.reg_lambda[1]),
        'reg_alpha': trial.suggest_float('reg_alpha', conf.parametros_lgb.reg_alpha[0], conf.parametros_lgb.reg_alpha[1]),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', conf.parametros_lgb.min_gain_to_split[0], conf.parametros_lgb.min_gain_to_split[1]),
        'verbosity': -1,
        'is_unbalance': trial.suggest_categorical('is_unbalance', [True, False]),
        #'scale_pos_weight': 97,
        #'pos_bagging_fraction': 1.0, 
        #'neg_bagging_fraction': 0.01, 
        'bagging_freq': trial.suggest_int('bagging_freq', conf.parametros_lgb.bagging_freq[0], conf.parametros_lgb.bagging_freq[1]),
        'silent': True, 'bin': 31
    }
    
    # Preparar dataset para entrenamiento y validación
    
    if isinstance(MES_TRAIN, list):
        df_train = df.filter(pl.col('foto_mes').cast(pl.Utf8).is_in([str(m) for m in MES_TRAIN]))
    else:
        df_train = df.filter(pl.col('foto_mes').cast(pl.Utf8) == str(MES_TRAIN))
   
    df_val = df.filter(pl.col('foto_mes').cast(pl.Utf8) == str(MES_VALIDACION))
    
    # Aplicar undersampling si es necesario
    df_train = aplicar_undersampling_clientes(df_train, tasa=undersampling, semilla=SEMILLA[0])
    
    # Preparar datos para LGBM
    X_train = df_train.drop(['clase_ternaria', 'clase_peso']).to_pandas()
    y_train = df_train['clase_ternaria'].to_numpy()
    weights_train = df_train['clase_peso'].to_numpy()
    
    X_val = df_val.drop(['clase_ternaria', 'clase_peso']).to_pandas()
    y_val = df_val['clase_ternaria'].to_numpy()
    weights_val = df_val['clase_peso'].to_numpy()

    train_data = lgb.Dataset(X_train, label=y_train, weight=weights_train)
    val_data = lgb.Dataset(X_val, label=y_val, weight=weights_val, reference=train_data)
    
    # Entrenar modelos distintos por cada seed
    ganancia_med_total = 0
    ganancia_max_total = 0
    envios_max_gan_total = 0
    
    base_seed = SEMILLA[0]
    rng = np.random.RandomState(base_seed)
    semillas = rng.randint(0, 2**32 - 1, size=n_semillas)

    #semillas = SEMILLA if isinstance(SEMILLA, list) else [SEMILLA]
    
    for seed in semillas:
        params['random_state'] = seed
        model = lgb.train(params, train_data, valid_sets=[val_data], feval=lgb_gan_eval,
                          callbacks=[lgb.early_stopping(15), lgb.log_evaluation(0)])
        # Predecir y calcular ganancia
        y_pred_proba = model.predict(X_val)
        #_, ganancia_med_iter,  _ = lgb_gan_eval(y_pred_proba, val_data)
        ganancia_med_iter, ganancia_max_iter, envios_max_gan = calcular_ganancias(y_pred_proba, val_data)

        # Sumar a la ganancia de los modelos anteriores
        ganancia_med_total += ganancia_med_iter
        ganancia_max_total += ganancia_max_iter
        envios_max_gan_total += envios_max_gan

    # Calcular ganancia media de los modelos entrenados en la iteración
    ganancia_med = ganancia_med_total / len(semillas)
    ganancia_max = ganancia_max_total / len(semillas)
    envios_max_gan = envios_max_gan_total / len(semillas)
    
    # Guardar cada iteración en JSON
    guardar_iteracion(trial, ganancia_med)
    logger.info(f"Trial {trial.number}: Ganancia Media = {ganancia_med:,.0f}, Ganancia Max = {ganancia_max:,.0f}, Envios Max Gan = {envios_max_gan:,.0f}")
    return ganancia_med
   
#######################################################################################################

def optimizar_con_seed_pesos(df: pl.DataFrame, n_trials=50) -> optuna.Study:
    """
    Ejecuta optimización bayesiana con ganancia media de modelos entrenados con distintos seeds.
  
    Args:
        df: DataFrame con datos
        n_trials: Número de trials a ejecutar
  
    Returns:
        optuna.Study: Estudio de Optuna con resultados de Ganancia media
    """
    study_name = f"{conf.STUDY_NAME}"
  
    logger.info(f"Iniciando optimización con Seeds - {n_trials} trials")
    # Ajuste para visualización del log si MES_TRAIN es lista o escalar
    periodos = MES_TRAIN + [MES_VALIDACION] if isinstance(MES_TRAIN, list) else [MES_TRAIN, MES_VALIDACION]
    logger.info(f"Configuración: períodos={periodos}")
  
    # Crear estudio
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        # Manejo de SEMILLA si es lista o escalar para el sampler
        sampler=optuna.samplers.TPESampler(seed=SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA)
    )
  
    # Ejecutar optimización pasando el DataFrame de Polars
    study.optimize(lambda trial: objetivo_ganancia_seeds(trial, df), n_trials=n_trials)
  
    # Resultados
    logger.info("=== OPTIMIZACIÓN CON SEEDS COMPLETADA ===")
    logger.info(f"Número de trials completados: {len(study.trials)}")
    logger.info(f"Mejor ganancia promedio = {study.best_value:,.0f}")
    logger.info(f"Mejores parámetros: {study.best_params}")

    return study
##########################################################################################################

def crear_o_cargar_estudio(study_name: str = None, semilla: int = None) -> optuna.Study:
    """
    Crea un nuevo estudio de Optuna o carga uno existente desde SQLite.
  
    Args:
        study_name: Nombre del estudio (si es None, usa STUDY_NAME del config)
        semilla: Semilla para reproducibilidad
  
    Returns:
        optuna.Study: Estudio de Optuna (nuevo o cargado)
    """
    study_name = conf.STUDY_NAME
  
    if semilla is None:
        semilla = SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA
  
    # Crear carpeta para bases de datos si no existe
    path_db = os.path.join(conf.BUCKET_NAME, "optuna_db")
    os.makedirs(path_db, exist_ok=True)
  
    # Ruta completa de la base de datos
    db_file = os.path.join(path_db, f"{study_name}.db")
    storage = f"sqlite:///{db_file}"
  
    # Verificar si existe un estudio previo
    if os.path.exists(db_file):
        logger.info(f"⚡ Base de datos encontrada: {db_file}")
        logger.info(f"🔄 Cargando estudio existente: {study_name}")
  
        try:
            # Cargar estudio existente
            study = optuna.load_study(study_name=study_name, storage=storage)
            n_trials_previos = len(study.trials) # la cantidad de trials registrados en el estudio existente es la cantidad de trials previos
  
            logger.info(f"✅ Estudio cargado exitosamente")
            logger.info(f"📊 Trials previos: {n_trials_previos}")
  
            if n_trials_previos > 0:
                logger.info(f"🏆 Mejor ganancia hasta ahora: {study.best_value:,.0f}")
  
            return study
  
        except Exception as e:
            logger.warning(f"⚠️ No se pudo cargar el estudio: {e}")
            logger.info(f"🆕 Creando nuevo estudio...")
    else:
        logger.info(f"🆕 No se encontró base de datos previa")
        logger.info(f"📁 Creando nueva base de datos: {db_file}")
  
    # Crear nuevo estudio
    study = optuna.create_study(
        study_name = study_name,
        storage = storage,
        direction = 'maximize',
        sampler = optuna.samplers.TPESampler(seed=semilla),
        load_if_exists=True
    )
  
    logger.info(f"✅ Nuevo estudio creado: {study_name}")
    logger.info(f"💾 Storage: {storage}")
  
    return study

#########################################################################################################

def optimizar(df: pl.DataFrame, n_trials: int, study_name: str = None, n_semillas: int = 1, undersampling: float = 0.01) -> optuna.Study:
    """
    Args:
        df: DataFrame con datos
        n_trials: Número de trials a ejecutar
        study_name: Nombre del estudio (si es None, usa el de config.yaml)
        undersampling: Undersampling para entrenamiento
  
    Description:
       Ejecuta optimización bayesiana de hiperparámetros usando configuración YAML.
       Guarda cada iteración en un archivo JSON separado. 
       Pasos:
        1. Crear estudio de Optuna
        2. Ejecutar optimización
        3. Retornar estudio

    Returns:
        optuna.Study: Estudio de Optuna con resultados
    """
    study_name = conf.STUDY_NAME

    logger.info(f"Iniciando optimización con {n_trials} trials")
    logger.info(f"Configuración: TRAIN={MES_TRAIN}, VALID={MES_VALIDACION}, SEMILLA={SEMILLA}")
  
    # Crear o cargar estudio
    study = crear_o_cargar_estudio(study_name, SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA)

    # Calcular cuántos trials faltan
    trials_previos = len(study.trials)
    trials_a_ejecutar = max(0, n_trials - trials_previos)
  
    if trials_previos > 0:
        logger.info(f"🔄 Retomando desde trial {trials_previos}")
        logger.info(f"📝 Trials a ejecutar: {trials_a_ejecutar} (total objetivo: {n_trials})")
    else:
        logger.info(f"🆕 Nueva optimización: {n_trials} trials")
  
    # Ejecutar optimización
    if trials_a_ejecutar > 0:
        # Llama a la función objetivo que ya soporta Polars
        study.optimize(lambda trial: objetivo_ganancia_seeds(trial, df, n_semillas, undersampling), n_trials=trials_a_ejecutar)
        logger.info(f"🏆 Mejor ganancia: {study.best_value:,.0f}")
        logger.info(f"Mejores parámetros: {study.best_params}")
    else:
        logger.info(f"✅ Ya se completaron {n_trials} trials")
  
    return study

#######################################################################################################

def optimizar_zlgbm(df: pl.DataFrame, n_trials=50, undersampling: float = 0.01) -> optuna.Study:
    """
    Ejecuta optimización bayesiana de modelo ZLightGBM usando Polars DataFrame.
    """
    study_name = f"{conf.STUDY_NAME}"
  
    logger.info(f"Iniciando optimización con zLGBM - {n_trials} trials")
    periodos = MES_TRAIN + [MES_VALIDACION] if isinstance(MES_TRAIN, list) else [MES_TRAIN, MES_VALIDACION]
    logger.info(f"Configuración: períodos={periodos}")
  
    # Crear estudio
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA)
    )
  
    # Ejecutar optimización
    study.optimize(lambda trial: objetivo_ganancia_zlgbm(trial, df, undersampling), n_trials=n_trials)
    # Guardar mejor modelo del estudio
    mejor_modelo = study.best_trial.model
    mejor_modelo.save_model(f'resultados/mejor_modelo_ob_zlgbm_{conf.STUDY_NAME}.txt')

    # Resultados
    logger.info("=== OPTIMIZACIÓN CON zLGBM COMPLETADA ===")
    logger.info(f"Número de trials completados: {len(study.trials)}")
    logger.info(f"Mejor ganancia promedio = {study.best_value:,.0f}")
    logger.info(f"Mejores parámetros: {study.best_params}")

    return study

#######################################################################################################

def objetivo_ganancia_zlgbm(trial: optuna.trial.Trial, df: pl.DataFrame, undersampling: float = 1) -> float:
    """
        Parameters:
    trial: trial de optuna
    df: dataframe con datos
  
    Description:
    Función objetivo que maximiza ganancia en mes de validación.
    Utiliza configuración YAML para períodos y semilla.
    Define parametros para el modelo zLightGBM
    Preparar dataset para entrenamiento y validación
    Entrena modelo con función de ganancia personalizada
    Predecir y calcular ganancia
    Guardar cada iteración en JSON
  
    Returns:
    float: ganancia total
    """
    # Hiperparámetros a optimizar
    params = {
        'boosting': 'gbdt',
        'objective': 'binary',
        'metric': 'custom',
        'first_metric_only': False,
        'boost_from_average': True,
        'feature_pre_filter': False,
        'force_row_wise': True,
        'verbosity': -1,
        'random_state': SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA,
        'num_threads': 4,
        'feature_fraction': 0.50,
        'num_iterations': 9999,
        'canaritos': 100,
        'min_sum_hessian_in_leaf': 0.001,
        'min_child_samples': trial.suggest_int('min_child_samples', conf.parametros_lgb.min_child_samples[0], conf.parametros_lgb.min_child_samples[1]),
        'num_leaves': 999,
        'learning_rate': 1.0,
        'gradient_bound': trial.suggest_float('gradient_bound', conf.parametros_lgb.gradient_bound[0], conf.parametros_lgb.gradient_bound[1]),
        'bin': 31
    }
  
    # Preparar dataset para entrenamiento y validación con Polars
    # Casteamos 'foto_mes' a string para asegurar compatibilidad si MES_TRAIN/VALIDACION son strings
    if isinstance(MES_TRAIN, list):
        df_train = df.filter(pl.col('foto_mes').cast(pl.Utf8).is_in([str(m) for m in MES_TRAIN]))
    else:
        df_train = df.filter(pl.col('foto_mes').cast(pl.Utf8) == str(MES_TRAIN))
    
    df_val = df.filter(pl.col('foto_mes').cast(pl.Utf8) == str(MES_VALIDACION))
    
    # Aplicar undersampling si es necesario
    df_train = aplicar_undersampling_clientes(df_train, tasa=undersampling)
    
    # Targets y pesos a numpy
    y_train = df_train['clase_ternaria'].to_numpy()
    y_val = df_val['clase_ternaria'].to_numpy()
    weights_train = df_train['clase_peso'].to_numpy()
    weights_val = df_val['clase_peso'].to_numpy()

    

    # Features a pandas (recomendado para máxima compatibilidad con LGBM por ahora)
    # Se excluyen 'clase_ternaria' y 'clase_peso'
    X_train = df_train.drop(['clase_ternaria', 'clase_peso']).to_pandas()
    X_val = df_val.drop(['clase_ternaria', 'clase_peso']).to_pandas()

    logger.info('Targets y pesos preparados para LGBM')

    # Crear datasets de LightGBM
    train_data = lgb.Dataset(X_train, label=y_train, weight=weights_train)
    val_data = lgb.Dataset(X_val, label=y_val, weight=weights_val, reference=train_data)

    logger.info(f"Tipo de dato de train_data: {type(train_data)}, Dimensiones de train_data: {train_data.data.shape}")

    # Entrenar modelo
    model = lgb.train(
        params, 
        train_data,
        valid_sets=[val_data],
        feval=lgb_gan_eval  
     #   callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    logger.info("Modelo zLGBM entrenado en train/val")

    # Predecir y calcular ganancia
    y_pred_proba = model.predict(X_val)

    #_, ganancia_med_iter,  _ = lgb_gan_eval(y_pred_proba, val_data)
    ganancia_med, ganancia_max, _ = calcular_ganancias(y_pred_proba, val_data)
    logger.info("Ganancia calculada en validación")

    # Guardar iteración en JSON
    guardar_iteracion(trial, ganancia_med)
    
    logger.info(f"Trial {trial.number}: Ganancia meseta = {ganancia_med:,.0f}")
  
    return ganancia_med
##########################################################################################

def entrenar_zlgbm_unico(df: pl.DataFrame, params_override: dict = None, undersampling: float = 1.0):
    """
    Entrena un único modelo zLightGBM sin dependencias de Optuna.

    Parameters:
    -----------
    df : pl.DataFrame
        Dataframe con datos.
    params_override : dict, optional
        Diccionario con los mejores hiperparámetros encontrados. 
        Si es None, usa los defaults hardcodeados.
    undersampling : float
        Tasa de undersampling.

    Returns:
    --------
    model : lgb.Booster
        El modelo entrenado listo para guardar o predecir.
    """
    
    # 1. Definición de Hiperparámetros Base
    # Estos son los defaults, pero se sobreescriben con lo que pases en params_override
    params = {
        'boosting': 'gbdt',
        'objective': 'binary',
        'metric': 'custom',
        'first_metric_only': False,
        'boost_from_average': True,
        'feature_pre_filter': False,
        'force_row_wise': True,
        'verbosity': -1,
        'random_state': SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA,
        'num_threads': 4,
        'feature_fraction': 0.50,
        'num_iterations': 9999,
        'canaritos': 5,
        'min_sum_hessian_in_leaf': 0.001,
        'min_child_samples': 20, 
        'num_leaves': 999,
        'learning_rate': 1.0,
        'gradient_bound': 0.1,
        'bin': 31
    }

    # 2. Actualizar parámetros si se pasaron argumentos (ej. los ganadores de Optuna)
    if params_override:
        params.update(params_override)
        logger.info(f"Parámetros actualizados con configuración personalizada.")

    # 3. Preparar dataset para entrenamiento y validación con Polars
    if isinstance(MES_TRAIN, list):
        df_train = df.filter(pl.col('foto_mes').cast(pl.Utf8).is_in([str(m) for m in MES_TRAIN]))
    else:
        df_train = df.filter(pl.col('foto_mes').cast(pl.Utf8) == str(MES_TRAIN))
    
    df_val = df.filter(pl.col('foto_mes').cast(pl.Utf8) == str(MES_VALIDACION))
    
    # Aplicar undersampling
    df_train = aplicar_undersampling_clientes(df_train, tasa=undersampling)
    
    # Targets y pesos a numpy
    y_train = df_train['clase_ternaria'].to_numpy()
    y_val = df_val['clase_ternaria'].to_numpy()
    weights_train = df_train['clase_peso'].to_numpy()
    weights_val = df_val['clase_peso'].to_numpy()

    # Features a pandas
    X_train = df_train.drop(['clase_ternaria', 'clase_peso']).to_pandas()
    X_val = df_val.drop(['clase_ternaria', 'clase_peso']).to_pandas()

    logger.info('Targets y pesos preparados para LGBM Single Run')

    # Crear datasets de LightGBM
    train_data = lgb.Dataset(X_train, label=y_train, weight=weights_train)
    val_data = lgb.Dataset(X_val, label=y_val, weight=weights_val, reference=train_data)

    # 4. Entrenar modelo
    # Nota: Si usas num_iterations muy alto, es recomendable descomentar los callbacks
    model = lgb.train(
        params, 
        train_data,
        valid_sets=[val_data],
        feval=lgb_gan_eval
        # callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)] # Recomendado activar para un solo run
    )
    logger.info("Modelo zLGBM final entrenado")

    # 5. Validación final (opcional, solo para loguear performance)
    y_pred_proba = model.predict(X_val)
    ganancia_med, ganancia_max, envios_max_gan = calcular_ganancias(y_pred_proba, val_data)
    
    logger.info(f"Ganancia final en validación ({MES_VALIDACION}): {ganancia_med:,.0f}")
    logger.info(f"Ganancia máxima posible en validación ({MES_VALIDACION}): {ganancia_max:,.0f} con envíos = {envios_max_gan}")

    # 6. Retornar el objeto modelo
    return model

#################################################

def aplicar_undersampling_clientes(
    df: pl.DataFrame,
    tasa: float,
    col_cliente: str = "numero_de_cliente",
    col_target: str = "clase_ternaria",
    clase_mayoritaria: int = 0,
    semilla: int = 99999
) -> pl.DataFrame:
    """
    Aplica undersampling a nivel de cliente, manteniendo todos los registros
    de clientes que alguna vez tuvieron una clase positiva y muestreando
    a los que siempre fueron clase mayoritaria.
    """
    # Si la tasa es 1.0, no hacemos nada
    if tasa >= 1.0:
        return df

    # 1. Identificar el estatus máximo de cada cliente (si alguna vez fue > 0)
    df_clientes_status = df.group_by(col_cliente).agg(
        pl.col(col_target).max().alias("max_clase")
    )

    registros_originales_clase_mayoritaria = df.filter(pl.col(col_target) == clase_mayoritaria).height
    registros_originales_clase_minoritarias = df.filter(pl.col(col_target) != clase_mayoritaria).height

    # 2. Separar IDs de clientes    
    clientes_a_conservar = df_clientes_status.filter(pl.col("max_clase") != clase_mayoritaria)[col_cliente]
    clientes_mayoritaria_pura = df_clientes_status.filter(pl.col("max_clase") == clase_mayoritaria)[col_cliente]

    # 3. Muestrear los clientes de la clase mayoritaria
    clientes_muestreados = clientes_mayoritaria_pura.sample(fraction=tasa, seed=semilla)

    # 4. Unir listas de clientes autorizados
    clientes_finales = pl.concat([clientes_a_conservar, clientes_muestreados])

    # 5. Filtrar el dataframe original y mezclar
    df_filtrado = df.filter(pl.col(col_cliente).is_in(clientes_finales))
    
    logger.info(f"Undersampling aplicado: tasa={tasa}, registros originales={df.height} (C May: {registros_originales_clase_mayoritaria} / C Min: {registros_originales_clase_minoritarias}), registros finales={df_filtrado.height} (CMay : {df_filtrado.filter(pl.col('clase_ternaria') != 1).height} / CMin: {df_filtrado.filter(pl.col('clase_ternaria') == 1).height})")

    return df_filtrado.sample(fraction=1.0, shuffle=True, seed=semilla)