# src/features_p.py      VERSIÓN DEL FEATURE ENGINEERING USANDO POLARS COMO BACKEND DE DUCKDB
import polars as pl
import duckdb
import logging
import os
import yaml
from .config import load_yaml_config

logger = logging.getLogger(__name__)

def feature_engineering_lag(df: pl.DataFrame, columnas: list[str], cant_lag: int = 1) -> pl.DataFrame:
    """
    Genera variables de lag para los atributos especificados utilizando SQL y Polars.
    """
    logger.info(f"Realizando feature engineering con {cant_lag} lags para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar lags")
        return df
  
    sql = "SELECT *"
    for attr in columnas:
        if attr in df.columns:
            for i in range(1, cant_lag + 1):
                sql += f", lag({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_lag_{i}"
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    sql += " FROM df"
    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL usando Polars como backend
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    # <-- CAMBIO CLAVE: .pl() en lugar de .df()
    df_result = con.execute(sql).pl()
    con.close()
  
    logger.info(f"Feature engineering completado. DataFrame resultante con {df_result.width} columnas")
    return df_result

######################################################################################

def feature_engineering_delta_lag(df: pl.DataFrame, columnas: list[str], cant_lag: int = 1) -> pl.DataFrame:
    logger.info(f"Realizando feature engineering de delta lag con {cant_lag} lags para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar delta lags")
        return df

    sql = "SELECT *"
    for attr in columnas:
        if attr in df.columns:
            for i in range(1, cant_lag + 1):
                sql += (
                    f", {attr} - lag({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) "
                    f"AS {attr}_delta_lag_{i}"
                )
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    sql += " FROM df"
    logger.debug(f"Consulta SQL: {sql}")

    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df_result = con.execute(sql).pl()
    con.close()

    logger.info(f"Feature engineering de delta lag completado. DataFrame resultante con {df_result.width} columnas")
    return df_result

####################################################################################

def feature_engineering_percentil(df: pl.DataFrame, columnas: list[str]) -> pl.DataFrame:
    logger.info(f"Realizando feature engineering con percentiles aproximados para {len(columnas) if columnas else 0} atributos")

    if not columnas:
        logger.warning("No se especificaron atributos para generar percentiles")
        return df

    df_result = df.clone()

    for attr in columnas:
        if attr not in df_result.columns:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")
            continue

        n_percentiles = 100
        percentiles = [round(i / n_percentiles, 2) for i in range(1, n_percentiles)]

        sql_limites = f"""
        WITH limites AS (
            SELECT 
                foto_mes,
                unnest(quantile_cont({attr}, {percentiles})) AS valor_limite,
                unnest(range(1, {n_percentiles})) AS percentil
            FROM df_result
            GROUP BY foto_mes
        )
        """

        sql_join = f"""
        SELECT 
            d.*, 
            MAX(l.percentil) AS {attr}_percentil
        FROM df_result d
        JOIN limites l
            ON d.foto_mes = l.foto_mes
           AND d.{attr} >= l.valor_limite
        GROUP BY ALL
        """

        con = duckdb.connect(database=":memory:")
        # Registramos el dataframe actual (que puede haber sido modificado en el loop anterior)
        con.register("df_result", df_result)
        df_result = con.execute(sql_limites + sql_join).pl()
        con.close()

        logger.debug(f"Consulta SQL para {attr} ejecutada")

    logger.info(f"Feature engineering completado. DataFrame resultante con {df_result.width} columnas")
    return df_result

#######################################################################################

def feature_engineering_rank(df: pl.DataFrame, columnas: list[str]) -> pl.DataFrame:
    logger.info(f"Realizando feature engineering con ranking normalizado para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar rankings")
        return df

    sql = "SELECT *"
    for attr in columnas:
        if attr in df.columns:
            sql += f"\n, (DENSE_RANK() OVER (PARTITION BY foto_mes ORDER BY {attr}) - 1) * 1.0 / (COUNT(*) OVER (PARTITION BY foto_mes) - 1) AS {attr}_rank"
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    sql += " FROM df"
    logger.debug(f"Consulta SQL: {sql}")

    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df_result = con.execute(sql).pl()
    con.close()

    logger.info(f"Feature engineering completado. DataFrame resultante con {df_result.width} columnas")
    return df_result

###################################################################################

def feature_engineering_drop(df: pl.DataFrame, columnas_a_eliminar: list[str]) -> pl.DataFrame:
    """
    Elimina las columnas especificadas del DataFrame (Versión Polars nativa, más eficiente que SQL para esto).
    """
    logger.info(f"Eliminando {len(columnas_a_eliminar) if columnas_a_eliminar else 0} columnas")
  
    if not columnas_a_eliminar:
        logger.warning("No se especificaron columnas para eliminar")
        return df
  
    columnas_existentes = [col for col in columnas_a_eliminar if col in df.columns]
    columnas_no_existentes = [col for col in columnas_a_eliminar if col not in df.columns]
  
    if columnas_no_existentes:
        logger.warning(f"Las siguientes columnas no existen en el DataFrame y no se pueden eliminar: {columnas_no_existentes}")
  
    # En Polars, drop toma una lista directamente, no requiere keyword 'columns='
    df = df.drop(columnas_existentes)
  
    logger.info(f"Columnas eliminadas. DataFrame resultante con {df.width} columnas")
    return df

####################################################################################

def feature_engineering_max_ultimos_n_meses(df: pl.DataFrame, columnas: list[str], n_meses: int = 3) -> pl.DataFrame:
    logger.info(f"Realizando feature engineering con máximo de los últimos {n_meses} meses para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar máximos")
        return df
    if n_meses < 1:
        logger.warning("La cantidad de meses debe ser al menos 1")
        return df

    sql = "SELECT *"
    for attr in columnas:
        if attr in df.columns:
            sql += (
                f"\n, max({attr}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes "
                f"ROWS BETWEEN {n_meses - 1} PRECEDING AND CURRENT ROW) AS {attr}_max_ult_{n_meses}m"
            )
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    sql += " FROM df"
    logger.debug(f"Consulta SQL: {sql}")

    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df_result = con.execute(sql).pl()
    con.close()

    logger.info(f"Feature engineering completado. DataFrame resultante con {df_result.width} columnas")
    return df_result

#####################################################################################

def feature_engineering_min_ultimos_n_meses(df: pl.DataFrame, columnas: list[str], n_meses: int = 3) -> pl.DataFrame:
    logger.info(f"Realizando feature engineering con mínimo de los últimos {n_meses} meses para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar mínimos")
        return df
    if n_meses < 1:
        logger.warning("La cantidad de meses debe ser al menos 1")
        return df

    sql = "SELECT *"
    for attr in columnas:
        if attr in df.columns:
            sql += (
                f"\n, min({attr}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes "
                f"ROWS BETWEEN {n_meses - 1} PRECEDING AND CURRENT ROW) AS {attr}_min_ult_{n_meses}m"
            )
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    sql += " FROM df"
    logger.debug(f"Consulta SQL: {sql}")

    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df_result = con.execute(sql).pl()
    con.close()

    logger.info(f"Feature engineering completado. DataFrame resultante con {df_result.width} columnas")
    return df_result

###################################################

def feature_engineering_ratios(df: pl.DataFrame, ratios: list[dict]) -> pl.DataFrame:
    logger.info(f"Realizando feature engineering de ratios para {len(ratios) if ratios else 0} combinaciones")

    if not ratios:
        logger.warning("No se especificaron ratios para generar")
        return df

    sql = "SELECT *"
    for ratio in ratios:
        numerador = ratio.get("numerador")
        denominador = ratio.get("denominador")
        nombre = ratio.get("nombre", f"{numerador}_{denominador}_ratio")
        if numerador not in df.columns or denominador not in df.columns:
            logger.warning(f"Alguna columna no existe en el DataFrame: {numerador}, {denominador}")
            continue
        sql += f", CASE WHEN {numerador} IS NULL OR {denominador} IS NULL THEN NULL ELSE {numerador} / NULLIF({denominador}, 0) END AS {nombre}"

    sql += " FROM df"
    logger.debug(f"Consulta SQL: {sql}")

    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df_result = con.execute(sql).pl()
    con.close()

    logger.info(f"Feature engineering de ratios completado. DataFrame resultante con {df_result.width} columnas")
    return df_result

######################################################

def feature_engineering_cambio_estado(df: pl.DataFrame, columnas: list[str], cant_lag: int = 1) -> pl.DataFrame:
    logger.info(f"Realizando feature engineering de cambio categórico con {cant_lag} lags para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar cambios categóricos")
        return df

    sql = "SELECT *"
    for attr in columnas:
        if attr in df.columns:
            for i in range(1, cant_lag + 1):
                sql += (
                    f", CASE WHEN {attr} IS NULL OR lag({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) IS NULL "
                    f"THEN NULL ELSE CAST({attr} != lag({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS INTEGER) END "
                    f"AS {attr}_cambio_lag_{i}"
                )
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    sql += " FROM df"
    logger.debug(f"Consulta SQL: {sql}")

    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df_result = con.execute(sql).pl()
    con.close()

    logger.info(f"Feature engineering de cambio categórico completado. DataFrame resultante con {df_result.width} columnas")
    return df_result
############################################################################################################

def feature_engineering_canaritos(df: pl.DataFrame, cant: int = 1) -> pl.DataFrame:
    """
    Genera 'cant' nuevas columnas con valores aleatorios distribuidos uniformemente entre 0 y 1.
    Útil para técnicas de selección de variables (boruta, etc).
    """
    logger.info(f"Generando {cant} variables 'canarito' con valores aleatorios")

    if cant < 1:
        logger.warning("La cantidad de canaritos debe ser al menos 1")
        return df

    sql = "SELECT *"
    for i in range(1, cant + 1):
        # DuckDB usa random() para generar floats entre 0 y 1
        sql += f", random() AS canarito_{i}"

    sql += " FROM df"
    logger.debug(f"Consulta SQL: {sql}")

    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df_result = con.execute(sql).pl()
    con.close()

    logger.info(f"Feature engineering de canaritos completado. DataFrame resultante con {df_result.width} columnas")
    return df_result

####################################################################################################################

def feature_engineering_slope(df: pl.DataFrame, columnas: list[str], n_meses: int = 6) -> pl.DataFrame:
    """
    Calcula la pendiente de la regresión lineal (tendencia) de los últimos n_meses.
    """
    logger.info(f"Realizando feature engineering de slope (tendencia) de últimos {n_meses} meses para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar slopes")
        return df
    if n_meses < 2:
        logger.warning("La cantidad de meses para calcular slope debe ser al menos 2")
        return df

    # Se crea un eje temporal lineal (monotonic_month) para evitar el salto numérico 
    # que ocurre en foto_mes entre diciembre (ej. 202312) y enero (ej. 202401).
    # Fórmula: (año * 12) + mes
    x_axis = "(CAST(foto_mes / 100 AS INTEGER) * 12 + (foto_mes % 100))"

    sql = "SELECT *"
    for attr in columnas:
        if attr in df.columns:
            sql += (
                f"\n, regr_slope({attr}, {x_axis}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes "
                f"ROWS BETWEEN {n_meses - 1} PRECEDING AND CURRENT ROW) AS {attr}_slope_ult_{n_meses}m"
            )
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    sql += " FROM df"
    logger.debug(f"Consulta SQL: {sql}")

    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df_result = con.execute(sql).pl()
    con.close()

    logger.info(f"Feature engineering de slope completado. DataFrame resultante con {df_result.width} columnas")
    return df_result
####################################################################################################################

def feature_engineering_sum(df: pl.DataFrame, sumas: list[dict]) -> pl.DataFrame:
    """
    Suma dos o más columnas. Útil para consolidar gastos (ej. VISA + Master).
    Maneja NULLs convirtiéndolos a 0 para la suma.
    """
    logger.info(f"Realizando feature engineering de sumas para {len(sumas) if sumas else 0} combinaciones")

    if not sumas:
        logger.warning("No se especificaron sumas para generar")
        return df

    sql = "SELECT *"
    for item in sumas:
        variables = item.get("variables")
        nombre = item.get("nombre")
        
        if not variables or not isinstance(variables, list):
             logger.warning(f"Formato incorrecto para suma: {item}")
             continue
             
        # Validar que existan (opcional, pero recomendado para evitar errores en SQL)
        vars_existentes = [v for v in variables if v in df.columns]
        if len(vars_existentes) < 2:
             logger.warning(f"No hay suficientes variables válidas para sumar en: {nombre or variables}")
             continue

        # Generar nombre automático si no existe
        if not nombre:
            nombre = "suma_" + "_".join(vars_existentes)

        # Construir la parte de la suma en SQL: COALESCE(col1, 0) + COALESCE(col2, 0) + ...
        suma_sql = " + ".join([f"COALESCE({v}, 0)" for v in vars_existentes])
        sql += f", ({suma_sql}) AS {nombre}"

    sql += " FROM df"
    logger.debug(f"Consulta SQL: {sql}")

    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df_result = con.execute(sql).pl()
    con.close()

    logger.info(f"Feature engineering de sumas completado. DataFrame resultante con {df_result.width} columnas")
    return df_result

####################################################################################################################

def feature_engineering_diff(df: pl.DataFrame, diffs: list[dict]) -> pl.DataFrame:
    """
    Resta dos columnas (v1 - v2). Útil para calcular netos o brechas.
    Se asume que 'variables' es una lista ordenada: [minuendo, sustraendo].
    Maneja NULLs convirtiéndolos a 0.
    """
    logger.info(f"Realizando feature engineering de restas para {len(diffs) if diffs else 0} combinaciones")

    if not diffs:
        logger.warning("No se especificaron restas para generar")
        return df

    sql = "SELECT *"
    for item in diffs:
        variables = item.get("variables")
        nombre = item.get("nombre")
        
        if not variables or not isinstance(variables, list) or len(variables) != 2:
             logger.warning(f"Se requieren exactamente 2 variables para restar [v1, v2], se recibió: {variables}")
             continue
             
        v1, v2 = variables[0], variables[1]
        
        # Validar existencia
        if v1 not in df.columns or v2 not in df.columns:
             logger.warning(f"Alguna variable no existe para la resta: {v1}, {v2}")
             continue

        # Generar nombre automático si no existe (ej. v1_minus_v2)
        if not nombre:
            nombre = f"{v1}_minus_{v2}"

        # SQL: COALESCE(v1, 0) - COALESCE(v2, 0)
        sql += f", (COALESCE({v1}, 0) - COALESCE({v2}, 0)) AS {nombre}"

    sql += " FROM df"
    logger.debug(f"Consulta SQL: {sql}")

    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df_result = con.execute(sql).pl()
    con.close()

    logger.info(f"Feature engineering de restas completado. DataFrame resultante con {df_result.width} columnas")
    return df_result
####################################################################################################################

def feature_engineering_greatest(df: pl.DataFrame, groups: list[dict]) -> pl.DataFrame:
    """
    Calcula el valor máximo (greatest) entre un grupo de columnas para cada fila.
    Ignora los valores NULL al comparar.
    """
    logger.info(f"Realizando feature engineering de 'greatest' (máximo entre columnas) para {len(groups) if groups else 0} grupos")

    if not groups:
        logger.warning("No se especificaron grupos para 'greatest'")
        return df

    sql = "SELECT *"
    for item in groups:
        variables = item.get("variables")
        nombre = item.get("nombre")
        
        if not variables or not isinstance(variables, list) or len(variables) < 2:
             logger.warning(f"Se requieren al menos 2 variables para 'greatest', se recibió: {variables}")
             continue
             
        # Validar existencia
        vars_existentes = [v for v in variables if v in df.columns]
        if len(vars_existentes) < 2:
             logger.warning(f"No hay suficientes variables válidas para 'greatest' en: {nombre or variables}")
             continue

        # Generar nombre automático si no existe
        if not nombre:
            nombre = "greatest_" + "_".join(vars_existentes)

        # SQL: GREATEST(col1, col2, col3, ...)
        vars_sql = ", ".join(vars_existentes)
        sql += f", GREATEST({vars_sql}) AS {nombre}"

    sql += " FROM df"
    logger.debug(f"Consulta SQL: {sql}")

    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df_result = con.execute(sql).pl()
    con.close()

    logger.info(f"Feature engineering de 'greatest' completado. DataFrame resultante con {df_result.width} columnas")
    return df_result
#####################################################################################################################

def feature_engineering_is_negative(df: pl.DataFrame, columnas: list[str]) -> pl.DataFrame:
    """
    Genera una bandera (1 o 0) que indica si el valor de una columna es negativo.
    Devuelve 1 si valor < 0, y 0 si valor >= 0 o NULL.
    """
    logger.info(f"Realizando feature engineering de 'is_negative' para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para 'is_negative'")
        return df

    sql = "SELECT *"
    for attr in columnas:
        if attr in df.columns:
            # La lógica CASE WHEN (col < 0) THEN 1 ELSE 0 maneja automáticamente los NULLs
            # (NULL < 0 es falso, por lo que caen en el ELSE 0)
            sql += f", CASE WHEN {attr} < 0 THEN 1 ELSE 0 END AS {attr}_is_negative"
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    sql += " FROM df"
    logger.debug(f"Consulta SQL: {sql}")

    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df_result = con.execute(sql).pl()
    con.close()

    logger.info(f"Feature engineering de 'is_negative' completado. DataFrame resultante con {df_result.width} columnas")
    return df_result

####################################################################################################################

FEATURES_CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "features.yaml")

def feature_engineering(
    df: pl.DataFrame,
    fe_etapa: str,
) -> pl.DataFrame:
    """
    Aplica múltiples técnicas de feature engineering sobre los atributos especificados en el archivo features.yaml.
    Recibe y devuelve un pl.DataFrame.
    """
    config = load_yaml_config(FEATURES_CONFIG)
    config_dict = vars(config)
    if fe_etapa not in config_dict:
        logger.warning(f"La etapa de operaciones '{fe_etapa}' no está definida en el archivo de configuración {FEATURES_CONFIG}.yaml.")
        return df
    operaciones_config = vars(config_dict[fe_etapa]) if isinstance(config_dict[fe_etapa], object) else config_dict[fe_etapa]
    
    # Polars .clone() en lugar de pandas .copy()
    df_result = df.clone()

    for op, op_cfg in operaciones_config.items():
        if not op_cfg:
            continue
        configs = op_cfg if isinstance(op_cfg, list) else [op_cfg]
        
        # Bucle principal de aplicación de features
        for cfg in configs:
             # Convert SimpleNamespace to dict if needed
            if not isinstance(cfg, dict) and hasattr(cfg, "__dict__"):
                cfg = vars(cfg)
            if op == "ratios":
                # (Lógica de ratios original mantenida...)
                ratio_dicts = configs if isinstance(op_cfg, list) else [op_cfg]
                expanded_ratios = []
                for ratio in ratio_dicts:
                     # ... (expansión de ratios igual que antes) ...
                     if not isinstance(ratio, dict) and hasattr(ratio, "__dict__"):
                        ratio = vars(ratio)
                     numerador = ratio.get("numerador")
                     denominador = ratio.get("denominador")
                     nombre = ratio.get("nombre")
                     if isinstance(numerador, list) and isinstance(denominador, list):
                        if isinstance(nombre, list):
                            for n, d, nm in zip(numerador, denominador, nombre):
                                expanded_ratios.append({"numerador": n, "denominador": d, "nombre": nm})
                        else:
                            for n, d in zip(numerador, denominador):
                                nm = f"{n}_{d}_ratio"
                                expanded_ratios.append({"numerador": n, "denominador": d, "nombre": nm})
                     elif isinstance(numerador, list):
                        for idx, n in enumerate(numerador):
                            d = denominador[idx] if isinstance(denominador, list) and idx < len(denominador) else denominador
                            nm = nombre[idx] if isinstance(nombre, list) and idx < len(nombre) else f"{n}_{d}_ratio"
                            expanded_ratios.append({"numerador": n, "denominador": d, "nombre": nm})
                     elif isinstance(denominador, list):
                         for idx, d in enumerate(denominador):
                            n = numerador[idx] if isinstance(numerador, list) and idx < len(numerador) else numerador
                            nm = nombre[idx] if isinstance(nombre, list) and idx < len(nombre) else f"{n}_{d}_ratio"
                            expanded_ratios.append({"numerador": n, "denominador": d, "nombre": nm})
                     elif isinstance(nombre, list):
                        for idx, nm in enumerate(nombre):
                            n = numerador[idx] if isinstance(numerador, list) and idx < len(numerador) else numerador
                            d = denominador[idx] if isinstance(denominador, list) and idx < len(denominador) else denominador
                            expanded_ratios.append({"numerador": n, "denominador": d, "nombre": nm})
                     else:
                        expanded_ratios.append(ratio)

                df_result = feature_engineering_ratios(df_result, expanded_ratios)
                break 
            if op == "sum":
                sumas_dicts = [vars(c) if not isinstance(c, dict) and hasattr(c, "__dict__") else c for c in configs]
                df_result = feature_engineering_sum(df_result, sumas_dicts)
                break 

            if op == "diff":
                diffs_dicts = [vars(c) if not isinstance(c, dict) and hasattr(c, "__dict__") else c for c in configs]
                df_result = feature_engineering_diff(df_result, diffs_dicts)
                break 

            if op == "greatest":
                groups_dicts = [vars(c) if not isinstance(c, dict) and hasattr(c, "__dict__") else c for c in configs]
                df_result = feature_engineering_greatest(df_result, groups_dicts)
                break 
            if op == "canaritos":
                # 1. Obtener configuración de cantidad
                cant = cfg.get("cant", 1) if isinstance(cfg, dict) else (cfg if isinstance(cfg, int) else 1)
                
                # 2. Guardar columnas actuales antes de aplicar canaritos
                cols_antes = df_result.columns
                
                # 3. Generar los canaritos (se agregarán al final por defecto)
                df_result = feature_engineering_canaritos(df_result, cant=cant)
                
                # 4. Identificar cuáles son las columnas nuevas
                cols_despues = df_result.columns
                nuevas_cols = [c for c in cols_despues if c not in cols_antes]
                
                # 5. Reordenar: Nuevas (canaritos) primero + Viejas después
                # Polars permite reordenar simplemente pasando la lista de nombres a select
                df_result = df_result.select(nuevas_cols + cols_antes)
                
                logger.info(f"Operación '{op}' aplicada. {cant} variables generadas y movidas al inicio del dataset.")
                continue

            if not isinstance(cfg, dict) and hasattr(cfg, "__dict__"):
                cfg = vars(cfg)
            columnas = cfg.get("columnas") if isinstance(cfg, dict) else cfg
            if not columnas:
                continue
            if not isinstance(columnas, list):
                columnas = [columnas]

            if op == "lag":
                cant_lag = cfg.get("cant_lag", 1) if isinstance(cfg, dict) else 1
                df_result = feature_engineering_lag(df_result, columnas, cant_lag)
                logger.info(f"Operación '{op}' aplicada. Nuevas columnas generadas.")
            elif op == "delta_lag":
                cant_lag = cfg.get("cant_lag", 1) if isinstance(cfg, dict) else 1
                df_result = feature_engineering_delta_lag(df_result, columnas, cant_lag)
                logger.info(f"Operación '{op}' aplicada. Nuevas columnas generadas.")
            elif op == "percentil":
                df_result = feature_engineering_percentil(df_result, columnas)
                logger.info(f"Operación '{op}' aplicada. Nuevas columnas generadas.")
            elif op == "rank":
                df_result = feature_engineering_rank(df_result, columnas)
                logger.info(f"Operación '{op}' aplicada. Nuevas columnas generadas.")
            elif op == "max":
                n_meses = cfg.get("n_meses", 3) if isinstance(cfg, dict) else 3
                df_result = feature_engineering_max_ultimos_n_meses(df_result, columnas, n_meses)
                logger.info(f"Operación '{op}' aplicada. Nuevas columnas generadas.")
            elif op == "min":
                n_meses = cfg.get("n_meses", 3) if isinstance(cfg, dict) else 3
                df_result = feature_engineering_min_ultimos_n_meses(df_result, columnas, n_meses)
                logger.info(f"Operación '{op}' aplicada. Nuevas columnas generadas.")
            elif op == "slope":
                n_meses = cfg.get("n_meses", 6) if isinstance(cfg, dict) else 6 # Default de 6 meses para tendencia
                df_result = feature_engineering_slope(df_result, columnas, n_meses)
                logger.info(f"Operación '{op}' aplicada. Nuevas columnas generadas.")
            elif op == "cambio_estado":
                cant_lag = cfg.get("cant_lag", 1) if isinstance(cfg, dict) else 1
                df_result = feature_engineering_cambio_estado(df_result, columnas, cant_lag)
                logger.info(f"Operación '{op}' aplicada. Nuevas columnas generadas.")
            elif op == "is_negative":
                df_result = feature_engineering_is_negative(df_result, columnas)
                logger.info(f"Operación '{op}' aplicada. Nuevas columnas generadas.")
            elif op == "drop": 
                df_result = feature_engineering_drop(df_result, columnas)
                logger.info(f"Operación '{op}' aplicada. Columnas eliminadas: {columnas}")
            else:
                logger.warning(f"Operación '{op}' no reconocida. Se omite.")

    return df_result


########################################################################################

def run_feature_pipeline(df: pl.DataFrame, stages: list[str]) -> pl.DataFrame:
    """
    Ejecuta el pipeline de feature engineering por etapas.
    """
    logger.info(f"Iniciando pipeline de {len(stages)} etapas...")
    
    # Esta es la lógica clave: el 'df' se actualiza en cada iteración
    for stage_name in stages:
        if not stage_name: # Ignora etapas vacías o nulas
            continue
            
        logger.info(f"--- Ejecutando etapa: {stage_name} ---")
        try:
            # La salida de una etapa es la entrada de la siguiente
            df = feature_engineering(df, fe_etapa=stage_name)
            logger.info(f"Etapa '{stage_name}' completada. Shape actual: {df.shape}")
        except Exception as e:
            logger.error(f"FALLÓ la etapa '{stage_name}': {e}")
            # Dependiendo de tu necesidad, puedes parar aquí
            raise e 

    logger.info("Pipeline de feature engineering finalizado exitosamente.")
    return df