import polars as pl
import duckdb
import sys
import yaml  # <--- Importamos la biblioteca YAML
from collections import defaultdict
import subprocess
from bucket_utils_p import archivo_existe_en_bucket
import os

def dq_interpolar_desde_yaml(
    input_csv_path: str, 
    output_csv_path: str, 
    yaml_config_path: str  # <--- Argumento modificado
):
    """
    Carga un dataset y aplica interpolación lineal para múltiples
    atributos y foto_meses especificados en un archivo de 
    configuración YAML.

    [LÓGICA]: Si un atributo en un 'bad_foto_mes' específico
    carece de vecinos (anterior o siguiente), se asigna NULL.

    Argumentos:
    input_csv_path (str): 
        Ruta al CSV de entrada.
    output_csv_path (str): 
        Ruta para el CSV de salida.
    yaml_config_path (str): 
        Ruta al archivo YAML que contiene las correcciones.
    """
    
    # --- 1. Cargar y parsear el YAML ---
    print(f"[YAML] Cargando configuración de DQ desde {yaml_config_path}...")
    try:
        with open(yaml_config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            
        raw_corrections = config_data.get('corrections', [])
        
        # Convertir la estructura de YAML a la lista de tuplas
        # que la función espera: [(attr, month), ...]
        corrections = []
        for item in raw_corrections:
            if 'attribute' in item and 'bad_month' in item:
                corrections.append((item['attribute'], item['bad_month']))
            else:
                print(f"[Advertencia YAML] Ignorando ítem mal formado: {item}")
                
    except FileNotFoundError:
        print(f"[Error] No se encontró el archivo YAML: {yaml_config_path}", file=sys.stderr)
        raise
    except yaml.YAMLError as e:
        print(f"[Error] Error al parsear el YAML: {e}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"[Error] Error inesperado al leer YAML: {e}", file=sys.stderr)
        raise

    # --- 2. Preparación para la Query Dinámica (Lógica existente) ---
    
    if not corrections:
        print("[Advertencia] No se proporcionaron correcciones en el YAML. "
              "Copiando el archivo de entrada al destino...")
        try:
            df_polars = pl.read_csv(input_csv_path)
            df_polars.write_csv(output_csv_path)
        except Exception as e:
            print(f"[Error] No se pudo copiar el archivo: {e}", file=sys.stderr)
        return

    # (El resto de la función es IDÉNTICA a la anterior)
    
    # Agrupamos correcciones por atributo
    corrections_map = defaultdict(list)
    for attr, month in corrections:
        corrections_map[attr].append(month)
    
    attributes_to_correct = list(corrections_map.keys())
    
    lag_lead_sql_parts = []
    case_sql_parts = []
    params_list = []
    
    print(f"[DuckDB] Preparando {len(corrections)} correcciones "
          f"para {len(attributes_to_correct)} atributos únicos...")

    for attr_name in attributes_to_correct:
        col_sql = f'"{attr_name}"'
        prev_col = f'"prev_{attr_name}"'
        next_col = f'"next_{attr_name}"'
        
        lag_lead_sql_parts.append(
            f"LAG({col_sql}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {prev_col}"
        )
        lag_lead_sql_parts.append(
            f"LEAD({col_sql}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {next_col}"
        )
        
        bad_months_list = corrections_map[attr_name]
        foto_mes_checks = " OR ".join(["foto_mes = ?"] * len(bad_months_list))
        params_list.extend(bad_months_list)
        
        case_str = f"""
            CASE
                WHEN ({foto_mes_checks}) THEN
                    CASE
                        WHEN {prev_col} IS NOT NULL AND {next_col} IS NOT NULL THEN
                            ({prev_col} + {next_col}) / 2.0
                        ELSE
                            NULL
                    END
                ELSE
                    {col_sql}
            END AS {col_sql}
        """
        case_sql_parts.append(case_str.strip())

    # --- 3. Ensamblado de la Query Completa ---
    
    helper_cols = []
    for attr_name in attributes_to_correct:
        helper_cols.append(f'"prev_{attr_name}"')
        helper_cols.append(f'"next_{attr_name}"')

    original_cols_to_exclude = [f'"{attr}"' for attr in attributes_to_correct]
    all_cols_to_exclude = ", ".join(original_cols_to_exclude + helper_cols)

    query = f"""
    WITH data_with_neighbors AS (
        SELECT
            *,
            {', '.join(lag_lead_sql_parts)}
        FROM df_polars
    )
    SELECT
        * EXCLUDE ({all_cols_to_exclude}),
        {', '.join(case_sql_parts)}
    FROM data_with_neighbors
    """

    # --- 4. Ejecución del Pipeline ---
    
    con = duckdb.connect(database=':memory:')
    try:
        # 1. Cargar
        print(f"[Polars] Cargando {input_csv_path}...")
        df_polars = pl.read_csv(input_csv_path)

        # 2. Ejecutar
        print(f"[DuckDB] Aplicando correcciones (Query con {len(params_list)} parámetros)...")
        result_df = con.execute(query, params_list).pl()
        
        # 3. Guardar
        print(f"[Polars] Guardando resultado en {output_csv_path}...")
        result_df.write_csv(output_csv_path)
        
        print(f"[DQ] Proceso completado. {output_csv_path} guardado.")

    except Exception as e:
        print(f"[Error] Ocurrió un error durante el procesamiento: {e}", file=sys.stderr)
        raise
    finally:
        con.close()
        print("[DuckDB] Conexión cerrada.")

##############################################################################################################

def dq_interpolar_gcs(
    input_bucket_path: str, 
    output_bucket_path: str, 
    yaml_config_path: str
):
    """
    Orquesta el proceso de DQ "GCS-a-GCS":
    1. Verifica si el archivo de SALIDA ya existe en GCS.
    2. Si no existe:
        a. Define rutas temporales locales para entrada y salida.
        b. Descarga el archivo de ENTRADA de GCS.
        c. Procesa el archivo local (con dq_interpolar_desde_yaml).
        d. Sube el resultado a GCS.
        e. Elimina archivos temporales locales.
        
    NOTA: Se asume que 'yaml_config_path' es una RUTA LOCAL
    accesible por el script.
    """
    
    # 1. Verificar si el archivo de SALIDA ya existe
    if archivo_existe_en_bucket(output_bucket_path):
        print(f"Proceso DQ omitido. El archivo {output_bucket_path} ya existe.")
        return

    print(f"Iniciando procesamiento DQ GCS-a-GCS...")

    # 2a. Definir rutas temporales locales
    # Usamos os.path.basename para obtener nombres de archivo únicos
    local_temp_input = f"./temp_input_dq_{os.path.basename(input_bucket_path)}"
    local_temp_output = f"./temp_output_dq_{os.path.basename(output_bucket_path)}"

    try:
        # 2b. Descargar el archivo de ENTRADA desde GCS
        print(f"Descargando {input_bucket_path} a {local_temp_input}...")
        subprocess.run(
            ["gsutil", "cp", input_bucket_path, local_temp_input],
            check=True,
            capture_output=True
        )
        
        # 2c. Procesar el archivo local temporal
        # (Aquí llamamos a la función de DQ que ya teníamos)
        dq_interpolar_desde_yaml(
            input_csv_path=local_temp_input, 
            output_csv_path=local_temp_output, 
            yaml_config_path=yaml_config_path
        )
        
        # 2d. Subir el resultado a GCS
        print(f"Subiendo {local_temp_output} a {output_bucket_path}...")
        subprocess.run(
            ["gsutil", "cp", local_temp_output, output_bucket_path],
            check=True,
            capture_output=True
        )
        print("¡Subida de archivo DQ completada!")

    except subprocess.CalledProcessError as e:
        print(f"Error de 'gsutil'. ¿El archivo de entrada existe? ¿Tienes permisos?")
        print(f"Detalle: {e.stderr.decode()}", file=sys.stderr)
    except Exception as e:
        print(f"Ocurrió un error durante el procesamiento GCS: {e}", file=sys.stderr)
    
    finally:
        # 2e. Limpiar ambos archivos temporales locales
        print("Limpiando archivos temporales de DQ...")
        if os.path.exists(local_temp_input):
            os.remove(local_temp_input)
            print(f"Eliminado: {local_temp_input}")
        if os.path.exists(local_temp_output):
            os.remove(local_temp_output)
            print(f"Eliminado: {local_temp_output}")
            
    print("Proceso de Data Quality (DQ) GCS-a-GCS finalizado.")