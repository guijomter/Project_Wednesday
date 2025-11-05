import os
import pandas as pd
import tempfile
import logging

# Configura un logger (o usa el tuyo si ya lo tienes)
# Si ya tienes una variable 'logger' global, puedes borrar estas dos líneas
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def archivo_existe_en_bucket(gcs_path: str) -> bool:
    """
    Verifica si un archivo existe en GCS usando 'gsutil ls'.
    Es más rápido y fiable que os.path.exists() en un bucket montado.
    """
    # El comando retorna 0 (éxito) si el archivo existe
    # Usamos -q para suprimir la salida de 'ls' en caso de éxito
    logger.debug(f"Verificando existencia de: {gcs_path}")
    return os.system(f"gsutil -q ls {gcs_path}") == 0

def guardar_en_buckets(df: pd.DataFrame, gcs_path: str):
    """
    Guarda un DataFrame en GCS usando la estrategia de Parquet + gsutil.
    
    1. Crea un archivo Parquet temporal local.
    2. Sube el archivo local a GCS con 'gsutil cp' (muestra progreso).
    3. Borra el archivo temporal.
    """
    # 1. Crear un path local temporal único
    # Usamos el directorio /tmp/ y extraemos el nombre base del gcs_path
    base_name = os.path.basename(gcs_path)
    local_path = os.path.join(tempfile.gettempdir(), base_name)
    
    try:
        # 2. Guardar el DF en Parquet local (muy rápido)
        logger.info(f"Guardando Parquet temporal en: {local_path}")
        df.to_parquet(local_path, index=False, engine='pyarrow')
        logger.info("Guardado local completado.")

        # 3. Subir a GCS con gsutil (rápido y con barra de progreso)
        logger.info(f"Subiendo a GCS: {gcs_path}")
        # (os.system mostrará la salida de gsutil en tu consola)
        os.system(f"gsutil cp {local_path} {gcs_path}")
        logger.info("¡Subida completada!")

    finally:
        # 4. Limpiar el archivo temporal local, incluso si hay un error
        if os.path.exists(local_path):
            os.remove(local_path)
            logger.info(f"Archivo temporal {local_path} eliminado.")

def cargar_de_buckets(gcs_path: str) -> pd.DataFrame:
    """
    Carga un DataFrame desde GCS usando la estrategia de gsutil + Parquet.
    
    1. Crea un path temporal local.
    2. Descarga el archivo de GCS a local con 'gsutil cp'.
    3. Carga el Parquet local en un DataFrame de pandas.
    4. Borra el archivo temporal y retorna el DataFrame.
    """
    # 1. Crear un path local temporal único
    base_name = os.path.basename(gcs_path)
    local_path = os.path.join(tempfile.gettempdir(), base_name)
    
    try:
        # 2. Descargar de GCS con gsutil (rápido)
        logger.info(f"Descargando de GCS: {gcs_path}")
        os.system(f"gsutil cp {gcs_path} {local_path}")
        logger.info(f"Descargado en: {local_path}")
        
        # 3. Cargar el Parquet local (muy rápido)
        df = pd.read_parquet(local_path)
        logger.info("DataFrame cargado en memoria.")
        return df

    finally:
        # 4. Limpiar el archivo temporal
        if os.path.exists(local_path):
            os.remove(local_path)
            logger.info(f"Archivo temporal {local_path} eliminado.")