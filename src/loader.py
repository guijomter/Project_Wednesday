# src/loader.py
import pandas as pd
import logging
import tempfile
import os


logger = logging.getLogger("__name__")

## Funcion para cargar datos
# def cargar_datos(path: str) -> pd.DataFrame | None:

#     '''
#     Carga un CSV desde 'path' y retorna un pandas.DataFrame.
#     '''

#     logger.info(f"Cargando dataset desde {path}")
#     try:
#         df = pd.read_csv(path)
#         logger.info(f"Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas")
#         return df
#     except Exception as e:
#         logger.error(f"Error al cargar el dataset: {e}")
#         raise

def cargar_datos(gcs_path: str) -> pd.DataFrame:
    '''
    Descarga un CSV desde GCS (gs://...) a un archivo temporal
    y lo carga en un pandas.DataFrame.
    '''
    
    # 1. Crear un path local temporal único
    # Usamos tempfile.NamedTemporaryFile para que nos dé un nombre único
    with tempfile.NamedTemporaryFile(suffix=".csv") as temp_file:
        local_path = temp_file.name
        
        logger.info(f"Descargando dataset desde {gcs_path} a {local_path}")
        try:
            # 2. Descargar de GCS con gsutil (rápido)
            # (os.system mostrará la salida de gsutil en tu consola)
            return_code = os.system(f"gsutil cp {gcs_path} {local_path}")
            
            if return_code != 0:
                raise Exception(f"gsutil cp falló con código {return_code}")

            logger.info(f"Dataset descargado. Cargando en pandas...")
            
            # 3. Cargar el CSV local (muy rápido)
            df = pd.read_csv(local_path)
            
            logger.info(f"Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas")
            return df

        except Exception as e:
            logger.error(f"Error al cargar el dataset: {e}")
            raise
    
    # El archivo temporal se borra automáticamente al salir del 'with'


def convertir_clase_ternaria_a_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte clase_ternaria a target binario reemplazando en el mismo atributo:
    - CONTINUA = 0
    - BAJA+1 y BAJA+2 = 1
  
    Args:
        df: DataFrame con columna 'clase_ternaria'
  
    Returns:
        pd.DataFrame: DataFrame con clase_ternaria convertida a valores binarios (0, 1)
    """
    # Crear copia del DataFrame para no modificar el original
    df_result = df.copy()
  
    # Contar valores originales para logging
    n_continua_orig = (df_result['clase_ternaria'] == 'CONTINUA').sum()
    n_baja1_orig = (df_result['clase_ternaria'] == 'BAJA+1').sum()
    n_baja2_orig = (df_result['clase_ternaria'] == 'BAJA+2').sum()
  
    # Convertir clase_ternaria a binario en el mismo atributo
    df_result['clase_ternaria'] = df_result['clase_ternaria'].map({
        'CONTINUA': 0,
        'BAJA+1': 1,
        'BAJA+2': 1
    })
  
    # Log de la conversión
    n_ceros = (df_result['clase_ternaria'] == 0).sum()
    n_unos = (df_result['clase_ternaria'] == 1).sum()
  
    logger.info(f"Conversión completada:")
    logger.info(f"  Original - CONTINUA: {n_continua_orig}, BAJA+1: {n_baja1_orig}, BAJA+2: {n_baja2_orig}")
    logger.info(f"  Binario - 0: {n_ceros}, 1: {n_unos}")
    logger.info(f"  Distribución: {n_unos/(n_ceros + n_unos)*100:.2f}% casos positivos")
  
    return df_result


#######################################

def convertir_clase_ternaria_a_target_peso(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte clase_ternaria a target binario reemplazando en el mismo atributo:
    - CONTINUA = 0
    - BAJA+1 y BAJA+2 = 1
  
    Args:
        df: DataFrame con columna 'clase_ternaria'
  
    Returns:
        pd.DataFrame: DataFrame con clase_ternaria convertida a valores binarios (0, 1)
    """
    # Crear copia del DataFrame para no modificar el original
    df_result = df.copy()
  
    # Contar valores originales para logging
    n_continua_orig = (df_result['clase_ternaria'] == 'CONTINUA').sum()
    n_baja1_orig = (df_result['clase_ternaria'] == 'BAJA+1').sum()
    n_baja2_orig = (df_result['clase_ternaria'] == 'BAJA+2').sum()
  
    # Asignar pesos según clase_ternaria  
    df_result['clase_peso'] = 1.0
    df_result.loc[df_result['clase_ternaria'] == 'BAJA+2', 'clase_peso'] = 1.00002
    df_result.loc[df_result['clase_ternaria'] == 'BAJA+1', 'clase_peso'] = 1.00001


    # Convertir clase_ternaria a binario en el mismo atributo
    df_result['clase_ternaria'] = df_result['clase_ternaria'].map({
        'CONTINUA': 0,
        'BAJA+1': 1,
        'BAJA+2': 1
    })
  
    # Log de la conversión
    n_ceros = (df_result['clase_ternaria'] == 0).sum()
    n_unos = (df_result['clase_ternaria'] == 1).sum()
  
    logger.info(f"Conversión completada:")
    logger.info(f"  Original - CONTINUA: {n_continua_orig}, BAJA+1: {n_baja1_orig}, BAJA+2: {n_baja2_orig}")
    logger.info(f"  Binario - 0: {n_ceros}, 1: {n_unos}")
    logger.info(f"  Distribución: {n_unos/(n_ceros + n_unos)*100:.2f}% casos positivos")
  
    return df_result