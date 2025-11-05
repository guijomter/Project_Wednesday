import duckdb
import pandas as pd
import subprocess
import os
from src.bucket_utils import archivo_existe_en_bucket


def procesar_y_guardar_clase_ternaria(input_csv_path: str, output_csv_path: str):
    """
    Ejecuta el pipeline de DuckDB sobre un CSV local y guarda el 
    resultado en otro CSV local.
    
    Argumentos:
    input_csv_path (str): Ruta al archivo CSV de entrada (local).
    output_csv_path (str): Ruta donde se guardará el nuevo CSV (local).
    """
    
    con = duckdb.connect(database=':memory:')
    
    try:
        print(f"[DuckDB] Cargando {input_csv_path}...")
        con.execute(f"""
            CREATE OR REPLACE TABLE competencia_01_crudo AS
            SELECT * FROM read_csv_auto('{input_csv_path}')
        """)
        
        print("[DuckDB] Generando tabla 'competencia_01' con clase_ternaria...")
        con.execute("""
            CREATE OR REPLACE TABLE competencia_01 AS
            WITH meses AS (
              SELECT DISTINCT foto_mes FROM competencia_01_crudo
            ),
            proximos_meses AS (
              SELECT
              foto_mes,
              LEAD(foto_mes, 1) OVER (ORDER BY foto_mes) AS mes_n_1,
              LEAD(foto_mes, 2) OVER (ORDER BY foto_mes) AS mes_n_2
              FROM meses
            )
            SELECT
            a.*,
            CASE 
                WHEN b.mes_n_2 IS NOT NULL AND LEAD(a.numero_de_cliente, 2) OVER(PARTITION BY a.numero_de_cliente ORDER BY a.foto_mes) IS NOT NULL THEN 'CONTINUA'
                WHEN b.mes_n_2 IS NOT NULL AND LEAD(a.numero_de_cliente, 1) OVER(PARTITION BY a.numero_de_cliente ORDER BY a.foto_mes) IS NOT NULL THEN 'BAJA+2'
                WHEN b.mes_n_1 IS NOT NULL AND LEAD(a.numero_de_cliente, 1) OVER(PARTITION BY a.numero_de_cliente ORDER BY a.foto_mes) IS NULL THEN 'BAJA+1'
                END AS clase_ternaria
            FROM competencia_01_crudo a
            INNER JOIN proximos_meses b
              ON a.foto_mes = b.foto_mes
        """)
        
        print(f"[DuckDB] Guardando resultado local en {output_csv_path}...")
        con.execute(f"""
            COPY competencia_01 
            TO '{output_csv_path}' (FORMAT CSV, HEADER)
        """)
        
    except Exception as e:
        print(f"[DuckDB] Ocurrió un error durante el procesamiento: {e}")
        raise
    finally:
        con.close()

# --- Función principal (Orquestador ACTUALIZADO) ---

def crear_clase_ternaria_gcs(input_bucket_path: str, output_bucket_path: str):
    """
    Orquesta el proceso completo "GCS-a-GCS":
    1. Verifica si el archivo de SALIDA ya existe en GCS.
    2. Si no existe:
        a. Define rutas temporales locales para la entrada y la salida.
        b. Descarga el archivo de ENTRADA de GCS a la ruta temporal.
        c. Procesa el archivo local temporal (con DuckDB).
        d. Sube el resultado a la ruta de SALIDA de GCS.
        e. Elimina ambos archivos temporales locales.
    """
    
    # 1. Verificar si el archivo de SALIDA ya existe
    if archivo_existe_en_bucket(output_bucket_path):
        print(f"Proceso omitido. El archivo {output_bucket_path} ya existe.")
        return

    print(f"Iniciando procesamiento GCS-a-GCS...")

    # 2a. Definir rutas temporales locales
    # Usamos os.path.basename para obtener nombres de archivo únicos
    local_temp_input = f"./temp_input_{os.path.basename(input_bucket_path)}"
    local_temp_output = f"./temp_output_{os.path.basename(output_bucket_path)}"

    try:
        # 2b. Descargar el archivo de ENTRADA desde GCS
        print(f"Descargando {input_bucket_path} a {local_temp_input}...")
        subprocess.run(
            ["gsutil", "cp", input_bucket_path, local_temp_input],
            check=True
        )
        
        # 2c. Procesar el archivo local temporal
        procesar_y_guardar_clase_ternaria(local_temp_input, local_temp_output)
        
        # 2d. Subir el resultado a GCS
        print(f"Subiendo {local_temp_output} a {output_bucket_path}...")
        subprocess.run(
            ["gsutil", "cp", local_temp_output, output_bucket_path],
            check=True
        )
        print("¡Subida completada!")

    except subprocess.CalledProcessError as e:
        print(f"Error de 'gsutil'. ¿El archivo de entrada existe? ¿Tienes permisos?")
        print(f"Detalle: {e}")
    except Exception as e:
        print(f"Ocurrió un error durante el procesamiento: {e}")
    
    finally:
        # 2e. Limpiar ambos archivos temporales locales
        print("Limpiando archivos temporales...")
        if os.path.exists(local_temp_input):
            os.remove(local_temp_input)
            print(f"Eliminado: {local_temp_input}")
        if os.path.exists(local_temp_output):
            os.remove(local_temp_output)
            print(f"Eliminado: {local_temp_output}")
            
    print("Proceso de creación de archivo csv con clase ternaria finalizado.")