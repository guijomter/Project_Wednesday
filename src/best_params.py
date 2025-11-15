import json
import logging
from .config import *

logger = logging.getLogger(__name__)

def cargar_mejores_hiperparametros(archivo_base: str = None) -> dict:
    """
    Carga los mejores hiperparámetros desde el archivo JSON de iteraciones de Optuna.
  
    Args:
        archivo_base: Nombre base del archivo (si es None, usa STUDY_NAME)
  
    Returns:
        dict: Mejores hiperparámetros encontrados
    """
    if archivo_base is None:
        archivo_base = conf.STUDY_NAME   ## Fijarse que conf viene de config.py
  
    archivo = f"resultados/{archivo_base}_iteraciones.json"
  
    try:
        with open(archivo, 'r') as f:
            iteraciones = json.load(f)
  
        if not iteraciones:
            raise ValueError("No se encontraron iteraciones en el archivo")
  
        # Encontrar la iteración con mayor ganancia
        mejor_iteracion = max(iteraciones, key=lambda x: x['value'])
        mejores_params = mejor_iteracion['params']
        mejor_ganancia = mejor_iteracion['value']

        # Parámetros fijos
        params_fijos = {
          #  'scale_pos_weight': 97,
          #  'bagging_fraction': 1.0,
          #  'pos_bagging_fraction': 1.0,
          #  'neg_bagging_fraction': 0.01,
            'bagging_freq': 1,
            'objective': 'binary',
            'metric': 'None',
            'random_state': SEMILLA[0]
        }

        # Agregar los fijos al diccionario de Optuna
        mejores_params.update(params_fijos)

        logger.info(f"Mejores hiperparámetros cargados desde {archivo}")
        logger.info(f"Mejor ganancia encontrada: {mejor_ganancia:,.0f}")
        logger.info(f"Trial número: {mejor_iteracion['trial_number']}")
        logger.info(f"Parámetros: {mejores_params}")
  
        return mejores_params
  
    except FileNotFoundError:
        logger.error(f"No se encontró el archivo {archivo}")
        logger.error("Asegúrate de haber ejecutado la optimización con Optuna primero")
        raise
    except Exception as e:
        logger.error(f"Error al cargar mejores hiperparámetros: {e}")
        raise

def obtener_estadisticas_optuna(archivo_base=None):
    """
    Obtiene estadísticas de la optimización de Optuna.
  
    Args:
        archivo_base: Nombre base del archivo
  
    Returns:
        dict: Estadísticas de la optimización
    """
    if archivo_base is None:
        archivo_base = conf.STUDY_NAME  ######### Fijarse que conf viene de config.py
  
    archivo = f"resultados/{archivo_base}_iteraciones.json"
  
    try:
        with open(archivo, 'r') as f:
            iteraciones = json.load(f)
  
        ganancias = [iter['value'] for iter in iteraciones]
  
        estadisticas = {
            'total_trials': len(iteraciones),
            'mejor_ganancia': max(ganancias),
            'peor_ganancia': min(ganancias),
            'ganancia_promedio': sum(ganancias) / len(ganancias),
            'top_5_trials': sorted(iteraciones, key=lambda x: x['value'], reverse=True)[:5]
        }
  
        logger.info("Estadísticas de optimización:")
        logger.info(f"  Total trials: {estadisticas['total_trials']}")
        logger.info(f"  Mejor ganancia: {estadisticas['mejor_ganancia']:,.0f}")
        logger.info(f"  Ganancia promedio: {estadisticas['ganancia_promedio']:,.0f}")
  
        return estadisticas
  
    except Exception as e:
        logger.error(f"Error al obtener estadísticas: {e}")
        raise

def cargar_mejores_hiperparametros_zlgbm(archivo_base: str = None) -> dict:
    """
    Carga los mejores hiperparámetros desde el archivo JSON de iteraciones de Optuna.
  
    Args:
        archivo_base: Nombre base del archivo (si es None, usa STUDY_NAME)
  
    Returns:
        dict: Mejores hiperparámetros encontrados
    """
    if archivo_base is None:
        archivo_base = conf.STUDY_NAME   ## Fijarse que conf viene de config.py
  
    archivo = f"resultados/{archivo_base}_iteraciones.json"
  
    try:
        with open(archivo, 'r') as f:
            iteraciones = json.load(f)
  
        if not iteraciones:
            raise ValueError("No se encontraron iteraciones en el archivo")
  
        # Encontrar la iteración con mayor ganancia
        mejor_iteracion = max(iteraciones, key=lambda x: x['value'])
        mejores_params = mejor_iteracion['params']
        mejor_ganancia = mejor_iteracion['value']

        # Parámetros fijos
        params_fijos = {
            'boosting': 'gbdt',
            'objective': 'binary',
            'metric': 'custom',
            'first_metric_only': False,
            'boost_from_average': True,
            'feature_pre_filter': False,
            'force_row_wise': True,
            'verbosity': -100,
            'num_threads': 4,
            'feature_fraction': 0.50, # un nuevo default
            'num_iterations': 9999,   # dejo libre la cantidad de arboles, zLightGBM se detiene solo
            'canaritos': 100,
            'min_sum_hessian_in_leaf': 0.001,
            'min_child_samples': 20,
            #'min_data_in_leaf': 20,  # default de LightGBM
            'num_leaves': 999, # default de LightGBM
            'learning_rate': 1,
            'random_state': SEMILLA[0],
            'gradient_bound': 0.1,
            'bin': 31
        }

        # Agregar los fijos al diccionario de Optuna
        mejores_params.update(params_fijos)

        logger.info(f"Mejores hiperparámetros cargados desde {archivo}")
        logger.info(f"Mejor ganancia encontrada: {mejor_ganancia:,.0f}")
        logger.info(f"Trial número: {mejor_iteracion['trial_number']}")
        logger.info(f"Parámetros: {mejores_params}")
  
        return mejores_params
  
    except FileNotFoundError:
        logger.error(f"No se encontró el archivo {archivo}")
        logger.error("Asegúrate de haber ejecutado la optimización con Optuna primero")
        raise
    except Exception as e:
        logger.error(f"Error al cargar mejores hiperparámetros: {e}")
        raise