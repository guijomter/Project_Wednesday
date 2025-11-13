import polars as pl
import duckdb
import sys
import yaml  # <--- Importamos la biblioteca YAML
from collections import defaultdict
import subprocess
from src.bucket_utils_p import archivo_existe_en_bucket
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

def ctrx_adjust(input_csv_path: str, output_csv_path: str):
    """
    [PASO 2 LOCAL] Carga un dataset y aplica la lógica de negocio 
    a 'ctrx_quarter' basada en 'cliente_antiguedad'.
    """
    
    con = duckdb.connect(database=':memory:')
    
    try:
        # 1. Cargar datos (el resultado del Paso 1)
        print(f"[Polars Paso 2] Cargando {input_csv_path}...")
        df_polars = pl.read_csv(input_csv_path)

        # 2. Definir la consulta SQL
        query = """
        SELECT
            * EXCLUDE (ctrx_quarter),
            CASE
                WHEN cliente_antiguedad = 1 THEN ctrx_quarter * 5.0
                WHEN cliente_antiguedad = 2 THEN ctrx_quarter * 2.0
                WHEN cliente_antiguedad = 3 THEN ctrx_quarter * 1.2
                ELSE ctrx_quarter
            END AS ctrx_quarter
        FROM df_polars
        """
        
        print("[DuckDB Paso 2] Ajustando 'ctrx_quarter' según antigüedad...")
        
        # 3. Ejecutar
        result_df = con.execute(query).pl()
        
        # 4. Guardar resultado final
        print(f"[Polars Paso 2] Guardando resultado final en {output_csv_path}...")
        result_df.write_csv(output_csv_path)
        
        print("[DQ] Proceso local (Paso 2) completado.")

    except Exception as e:
        print(f"[Error Paso 2] {e}", file=sys.stderr)
        raise
    finally:
        con.close()


############################################################################################################

def data_quality_gcs(
    input_bucket_path: str, 
    output_bucket_path: str, 
    yaml_config_path: str
):
    """
    Orquesta el pipeline de DQ "GCS-a-GCS" en dos pasos locales:
    1. Descarga el archivo de entrada.
    2. Llama a _local_step_1_interpolate.
    3. Llama a _local_step_2_ctrx_adjust.
    4. Sube el resultado final.
    5. Limpia todos los archivos temporales.
    """
    
    # 1. Verificar si el archivo de SALIDA FINAL ya existe
    if archivo_existe_en_bucket(output_bucket_path):
        print(f"Proceso DQ COMBINADO omitido. El archivo {output_bucket_path} ya existe.")
        return

    print(f"Iniciando pipeline DQ GCS-a-GCS (Encadenado)...")

    # 2a. Definir rutas temporales locales
    base_input_name = os.path.basename(input_bucket_path)
    base_output_name = os.path.basename(output_bucket_path)
    
    local_temp_input = f"./temp_input_{base_input_name}"
    local_temp_intermediate = f"./temp_intermediate_{base_output_name}"
    local_temp_output = f"./temp_output_final_{base_output_name}"

    try:
        # 2b. Descargar el archivo de ENTRADA (Solo 1 descarga)
        print(f"Descargando {input_bucket_path} a {local_temp_input}...")
        subprocess.run(
            ["gsutil", "cp", input_bucket_path, local_temp_input],
            check=True,
            capture_output=True
        )
        
        # --- 2c. Ejecutar PASO 1 LOCAL ---
        print("\n--- Iniciando DQ Paso 1 (Interpolación) ---")
        dq_interpolar_desde_yaml(
            input_csv_path=local_temp_input, 
            output_csv_path=local_temp_intermediate, 
            yaml_config_path=yaml_config_path
        )
        
        # --- 2d. Ejecutar PASO 2 LOCAL ---
        print("\n--- Iniciando DQ Paso 2 (Ajuste ctrx_quarter) ---")
        ctrx_adjust(
            input_csv_path=local_temp_intermediate, # <-- Input es el output del Paso 1
            output_csv_path=local_temp_output      # <-- Output final
        )
        
        # 2e. Subir el resultado FINAL (Solo 1 subida)
        print(f"\nSubiendo resultado final {local_temp_output} a {output_bucket_path}...")
        subprocess.run(
            ["gsutil", "cp", local_temp_output, output_bucket_path],
            check=True,
            capture_output=True
        )
        print("¡Subida de archivo DQ (Pipeline) completada!")

    except subprocess.CalledProcessError as e:
        print(f"Error de 'gsutil' (Pipeline).")
        print(f"Detalle: {e.stderr.decode()}", file=sys.stderr)
    except Exception as e:
        print(f"Ocurrió un error durante el GCS (Pipeline): {e}", file=sys.stderr)
    
    finally:
        # 2f. Limpiar TODOS los archivos temporales locales
        print("Limpiando archivos temporales del pipeline...")
        for f in [local_temp_input, local_temp_intermediate, local_temp_output]:
            if os.path.exists(f):
                os.remove(f)
                print(f"Eliminado: {f}")
            
    print("Proceso de Data Quality (Pipeline) GCS-a-GCS finalizado.")