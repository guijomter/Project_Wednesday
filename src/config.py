import yaml
import os
import logging
from types import SimpleNamespace

logger = logging.getLogger(__name__)

#Ruta del archivo de configuracion
PATH_CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "conf.yaml")

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


