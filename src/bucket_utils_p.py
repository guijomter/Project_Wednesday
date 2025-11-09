import os
import polars as pl
import tempfile
import logging
import subprocess

# Configura un logger (o usa el tuyo si ya lo tienes)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def archivo_existe_en_bucket(bucket_path: str) -> bool:
    """
    Verifica si un archivo existe en GCS usando 'gsutil ls' con subprocess.
    """
    logger.debug(f"Verificando existencia de: {bucket_path}...")
    try:
        subprocess.run(
            ["gsutil", "-q", "ls", bucket_path], 
            check=True,        # Lanza excepción si el comando falla (retorno != 0)
            stdout=subprocess.PIPE, # No imprimir salida en la consola
            stderr=subprocess.PIPE  # No imprimir errores en la consola
        )
        # Si check=True pasa, el comando retornó 0 -> el archivo existe
        logger.debug("Resultado: El archivo SÍ existe.")
        return True
    except subprocess.CalledProcessError:
        # check=True falló, (gsutil retornó != 0) -> El archivo NO existe
        logger.debug("Resultado: El archivo NO existe.")
        return False
    except FileNotFoundError:
        # 'gsutil' no está instalado
        logger.error("'gsutil' no se encontró. Asegúrate de que esté instalado y en tu PATH.")
        raise # Relanzamos el error porque es un problema de entorno


def guardar_en_buckets(df: pl.DataFrame, gcs_path: str):
    """
    Guarda un polars.DataFrame en GCS usando la estrategia de Parquet + gsutil.
    
    1. Crea un archivo Parquet temporal local.
    2. Sube el archivo local a GCS con 'gsutil cp' (muestra progreso).
    3. Borra el archivo temporal.
    """
    # 1. Crear un path local temporal único
    base_name = os.path.basename(gcs_path)
    local_path = os.path.join(tempfile.gettempdir(), base_name)
    
    try:
        # 2. Guardar el DF en Parquet local (muy rápido)
        logger.info(f"Guardando Parquet temporal en: {local_path}")
        # Polars usa write_parquet en lugar de to_parquet
        df.write_parquet(local_path) 
        logger.info("Guardado local completado.")

        # 3. Subir a GCS con gsutil (rápido y con barra de progreso)
        logger.info(f"Subiendo a GCS: {gcs_path}")
        # Usamos subprocess.run para mejor manejo de errores que os.system si se desea,
        # pero manteniendo os.system por simplicidad como en el original si prefieres ver la salida directa.
        # Para mantener consistencia con tu estilo original usando os.system para ver el progreso:
        return_code = os.system(f"gsutil cp {local_path} {gcs_path}")
        if return_code != 0:
             raise Exception(f"Error al subir a GCS. Código de salida: {return_code}")
        logger.info("¡Subida completada!")

    finally:
        # 4. Limpiar el archivo temporal local, incluso si hay un error
        if os.path.exists(local_path):
            os.remove(local_path)
            logger.info(f"Archivo temporal {local_path} eliminado.")

def cargar_de_buckets(gcs_path: str) -> pl.DataFrame:
    """
    Carga un DataFrame desde GCS usando la estrategia de gsutil + Parquet y devuelve un polars.DataFrame.
    
    1. Crea un path temporal local.
    2. Descarga el archivo de GCS a local con 'gsutil cp'.
    3. Carga el Parquet local en un polars.DataFrame.
    4. Borra el archivo temporal y retorna el DataFrame.
    """
    # 1. Crear un path local temporal único
    base_name = os.path.basename(gcs_path)
    local_path = os.path.join(tempfile.gettempdir(), base_name)
    
    try:
        # 2. Descargar de GCS con gsutil (rápido)
        logger.info(f"Descargando de GCS: {gcs_path}")
        return_code = os.system(f"gsutil cp {gcs_path} {local_path}")
        if return_code != 0:
             raise Exception(f"Error al descargar de GCS. Código de salida: {return_code}")
        logger.info(f"Descargado en: {local_path}")
        
        # 3. Cargar el Parquet local (muy rápido con Polars)
        df = pl.read_parquet(local_path)
        logger.info("DataFrame cargado en memoria.")
        return df

    finally:
        # 4. Limpiar el archivo temporal
        if os.path.exists(local_path):
            os.remove(local_path)
            logger.info(f"Archivo temporal {local_path} eliminado.")