import polars as pl
import logging
import tempfile
import os

logger = logging.getLogger(__name__)

def cargar_datos(gcs_path: str) -> pl.DataFrame:
    '''
    Descarga un CSV o CSV.GZ desde GCS (gs://...) a un archivo temporal
    y lo carga en un polars.DataFrame, manejando la descompresión automáticamente.
    '''
    
    # 1. Detectar si el archivo está comprimido basado en la ruta GCS para el sufijo
    is_gzipped = gcs_path.endswith('.gz')
    file_suffix = ".csv.gz" if is_gzipped else ".csv"
    
    # 2. Usar tempfile.NamedTemporaryFile para un manejo seguro
    with tempfile.NamedTemporaryFile(suffix=file_suffix) as temp_file:
        local_path = temp_file.name
        
        logger.info(f"Descargando dataset desde {gcs_path} a {local_path}")
        try:
            # 3. Descargar de GCS con gsutil (rápido)
            return_code = os.system(f"gsutil cp {gcs_path} {local_path}")
            
            if return_code != 0:
                raise Exception(f"gsutil cp falló con código {return_code}")

            logger.info(f"Dataset descargado. Cargando en polars...")
            
            # 4. Cargar el CSV local. Polars detecta gzip automáticamente por la extensión .gz
            df = pl.read_csv(local_path)
            
            logger.info(f"Dataset cargado con {df.height} filas y {df.width} columnas")
            return df

        except Exception as e:
            logger.error(f"Error al cargar el dataset: {e}")
            raise
    
    # El archivo temporal se borra automáticamente al salir del 'with'

def convertir_clase_ternaria_a_target(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convierte clase_ternaria a target binario reemplazando en el mismo atributo:
    - CONTINUA = 0
    - BAJA+1 y BAJA+2 = 1
  
    Args:
        df: pl.DataFrame con columna 'clase_ternaria'
  
    Returns:
        pl.DataFrame: DataFrame con clase_ternaria convertida a valores binarios (0, 1)
    """
    # Contar valores originales para logging (usando filter y height para eficiencia en Polars)
    n_continua_orig = df.filter(pl.col('clase_ternaria') == 'CONTINUA').height
    n_baja1_orig = df.filter(pl.col('clase_ternaria') == 'BAJA+1').height
    n_baja2_orig = df.filter(pl.col('clase_ternaria') == 'BAJA+2').height
  
    # Convertir clase_ternaria a binario. 
    # Usamos replace para un mapeo directo. Es eficiente.
    # Aseguramos el tipo de dato correcto para la columna resultante (Int8 o similar es suficiente para 0/1)
    df_result = df.with_columns(
        pl.col('clase_ternaria').replace({
            'CONTINUA': 0,
            'BAJA+1': 1,
            'BAJA+2': 1
        }, default=None).cast(pl.Int8) # cast opcional pero recomendado para ahorrar memoria si eran strings
    )
  
    # Log de la conversión
    n_ceros = df_result.filter(pl.col('clase_ternaria') == 0).height
    n_unos = df_result.filter(pl.col('clase_ternaria') == 1).height
    total = n_ceros + n_unos
  
    logger.info(f"Conversión completada:")
    logger.info(f"  Original - CONTINUA: {n_continua_orig}, BAJA+1: {n_baja1_orig}, BAJA+2: {n_baja2_orig}")
    logger.info(f"  Binario - 0: {n_ceros}, 1: {n_unos}")
    if total > 0:
        logger.info(f"  Distribución: {n_unos / total * 100:.2f}% casos positivos")
  
    return df_result

#######################################

def convertir_clase_ternaria_a_target_peso(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convierte clase_ternaria a target binario y asigna pesos usando Polars.
    """
    # Contar valores originales para logging
    n_continua_orig = df.filter(pl.col('clase_ternaria') == 'CONTINUA').height
    n_baja1_orig = df.filter(pl.col('clase_ternaria') == 'BAJA+1').height
    n_baja2_orig = df.filter(pl.col('clase_ternaria') == 'BAJA+2').height
  
    # Asignar pesos y convertir clase binaria en una sola operación eficiente con with_columns
    df_result = df.with_columns([
        # Crear columna de pesos basada en condiciones
        pl.when(pl.col('clase_ternaria') == 'BAJA+2').then(1.00002)
          .when(pl.col('clase_ternaria') == 'BAJA+1').then(1.00001)
          .otherwise(1.0).alias('clase_peso'),
          
        # Convertir clase ternaria a binaria
        pl.col('clase_ternaria').replace({
            'CONTINUA': 0,
            'BAJA+1': 1,
            'BAJA+2': 1
        }, default=None).cast(pl.Int8)
    ])

    # Log de la conversión
    n_ceros = df_result.filter(pl.col('clase_ternaria') == 0).height
    n_unos = df_result.filter(pl.col('clase_ternaria') == 1).height
    total = n_ceros + n_unos
  
    logger.info(f"Conversión completada:")
    logger.info(f"  Original - CONTINUA: {n_continua_orig}, BAJA+1: {n_baja1_orig}, BAJA+2: {n_baja2_orig}")
    logger.info(f"  Binario - 0: {n_ceros}, 1: {n_unos}")
    if total > 0:
        logger.info(f"  Distribución: {n_unos / total * 100:.2f}% casos positivos")
  
    return df_result