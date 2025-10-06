import yaml
import os
import logging
from types import SimpleNamespace

logger = logging.getLogger(__name__)

#Ruta del archivo de configuracion
PATH_CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "conf.yaml")

# try:
#     with open(PATH_CONFIG, "r") as f:
#         _cfgGeneral = yaml.safe_load(f)
#         _cfg = _cfgGeneral["competencia01"]

#         STUDY_NAME = _cfgGeneral.get("STUDY_NAME", "Wednesday")
#         DATA_PATH = _cfg.get("DATA_PATH", "../data/competencia.csv")
#         SEMILLA = _cfg.get("SEMILLA", [42])
#         MES_TRAIN = _cfg.get("MES_TRAIN", "202102")
#         MES_VALIDACION = _cfg.get("MES_VALIDACION", "202103")
#         MES_TEST = _cfg.get("MES_TEST", "202104")
#         GANANCIA_ACIERTO = _cfg.get("GANANCIA_ACIERTO", None)
#         COSTO_ESTIMULO = _cfg.get("COSTO_ESTIMULO", None)
#  # Configuración para entrenamiento final
#         FINAL_TRAIN = _cfg.get("FINAL_TRAIN", ["202101", "202102", "202103", "202104"])
#         FINAL_PREDIC = _cfg.get("FINAL_PREDIC", "202106")

# except Exception as e:
#     logger.error(f"Error al cargar el archivo de configuracion: {e}")
#     raise

def dict_to_namespace(d):
    """Convierte recursivamente un diccionario a SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d

def load_yaml_config(file_path=PATH_CONFIG):
    """Carga el YAML y devuelve un único objeto con acceso por puntos."""
    try:
        with open(file_path, "r") as f:
            config_data = yaml.safe_load(f)

        # Convertir todo el diccionario a un objeto con acceso por puntos
        conf = dict_to_namespace(config_data)

        # Hacer el objeto conf disponible globalmente
        globals()['conf'] = conf

        return conf

    except Exception as e:
        logger.error(f"Error al cargar el archivo de configuración: {e}")
        raise

# Cargar configuración automáticamente al importar el módulo
conf = load_yaml_config(PATH_CONFIG)

## Disponibilizar variables globales de la competencia01 para acceso directo (no hace falta poner conf.competencia01. delante)
cfg = conf.competencia01
globals().update(cfg.__dict__)


