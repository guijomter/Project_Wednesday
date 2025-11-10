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

FEATURES_CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "features.yaml")

def feature_engineering(
    df: pl.DataFrame,
    competencia: str,
) -> pl.DataFrame:
    """
    Aplica múltiples técnicas de feature engineering sobre los atributos especificados en el archivo features.yaml.
    Recibe y devuelve un pl.DataFrame.
    """
    config = load_yaml_config(FEATURES_CONFIG)
    config_dict = vars(config)
    if competencia not in config_dict:
        logger.warning(f"La competencia '{competencia}' no está definida en el archivo de configuración {FEATURES_CONFIG}.yaml.")
        return df
    operaciones_config = vars(config_dict[competencia]) if isinstance(config_dict[competencia], object) else config_dict[competencia]
    
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

            if op == "canaritos":
                # Soporta formato yaml: "canaritos: 10" o "canaritos: {cant: 10}"
                cant = cfg.get("cant", 1) if isinstance(cfg, dict) else (cfg if isinstance(cfg, int) else 1)
                df_result = feature_engineering_canaritos(df_result, cant=cant)
                logger.info(f"Operación '{op}' aplicada. {cant} variables generadas.")
                continue # Saltamos el resto del bucle ya que no requiere 'columnas'

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
            elif op == "cambio_estado":
                cant_lag = cfg.get("cant_lag", 1) if isinstance(cfg, dict) else 1
                df_result = feature_engineering_cambio_estado(df_result, columnas, cant_lag)
                logger.info(f"Operación '{op}' aplicada. Nuevas columnas generadas.")
            elif op == "drop": 
                df_result = feature_engineering_drop(df_result, columnas)
                logger.info(f"Operación '{op}' aplicada. Columnas eliminadas: {columnas}")
            else:
                logger.warning(f"Operación '{op}' no reconocida. Se omite.")

    return df_result