# src/features.py
import pandas as pd
import duckdb
import logging
import os
import yaml
from .config import load_yaml_config

logger = logging.getLogger("__name__")

def feature_engineering_lag(df: pd.DataFrame, columnas: list[str], cant_lag: int = 1) -> pd.DataFrame:
    """
    Genera variables de lag para los atributos especificados utilizando SQL.
  
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list
        Lista de atributos para los cuales generar lags. Si es None, no se generan lags.
    cant_lag : int, default=1
        Cantidad de lags a generar para cada atributo
  
    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de lag agregadas
    """

    logger.info(f"Realizando feature engineering con {cant_lag} lags para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar lags")
        return df
  
    # Construir la consulta SQL
    sql = "SELECT *"
  
    # Agregar los lags para los atributos especificados
    for attr in columnas:
        if attr in df.columns:
            for i in range(1, cant_lag + 1):
                sql += f", lag({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_lag_{i}"
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    # Completar la consulta
    sql += " FROM df"

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df()
    con.close()

    #print(df.head())
  
    logger.info(f"Feature engineering completado. DataFrame resultante con {df.shape[1]} columnas")

    return df
######################################################################################

def feature_engineering_delta_lag(df: pd.DataFrame, columnas: list[str], cant_lag: int = 1) -> pd.DataFrame:
    """
    Genera variables de delta (diferencia) entre el valor actual y el valor lag para los atributos especificados utilizando SQL.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list
        Lista de atributos para los cuales generar deltas de lag. Si es None, no se generan deltas.
    cant_lag : int, default=1
        Cantidad de lags a considerar para calcular el delta

    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de delta de lag agregadas
    """

    logger.info(f"Realizando feature engineering de delta lag con {cant_lag} lags para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar delta lags")
        return df

    # Construir la consulta SQL
    sql = "SELECT *"

    # Agregar los delta lags para los atributos especificados
    for attr in columnas:
        if attr in df.columns:
            for i in range(1, cant_lag + 1):
                sql += (
                    f", {attr} - lag({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) "
                    f"AS {attr}_delta_lag_{i}"
                )
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    # Completar la consulta
    sql += " FROM df"

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df()
    con.close()

    logger.info(f"Feature engineering de delta lag completado. DataFrame resultante con {df.shape[1]} columnas")

    return df

####################################################################################

# def feature_engineering_percentil(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
#     """
#     Genera variables de percentil para los atributos especificados utilizando SQL.
  
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         DataFrame con los datos
#     columnas : list
#         Lista de atributos para los cuales generar los percentiles. Si es None, no se generan 
  
#     Returns:
#     --------
#     pd.DataFrame
#         DataFrame con las variables de percentil agregadas
#     """

#     logger.info(f"Realizando feature engineering con percentiles para {len(columnas) if columnas else 0} atributos")

#     if columnas is None or len(columnas) == 0:
#         logger.warning("No se especificaron atributos para generar percentiles")
#         return df
  
#     # Construir la consulta SQL
#     sql = "SELECT *"
  
#     # Agregar los lags para los atributos especificados
#     for attr in columnas:
#         if attr in df.columns:
#             sql += f"\n, ROUND(percent_rank() OVER (PARTITION BY foto_mes ORDER BY {attr}) * 100) AS {attr}_percentil"
#             #sql += f"\n, ntile(100) over (partition by foto_mes order by {attr}) AS {attr}_percentil"
#         else:
#             logger.warning(f"El atributo {attr} no existe en el DataFrame")

#     # Completar la consulta
#     sql += " FROM df"

#     logger.debug(f"Consulta SQL: {sql}")

#     # Ejecutar la consulta SQL
#     con = duckdb.connect(database=":memory:")
#     con.register("df", df)
#     df = con.execute(sql).df()
#     con.close()

#     #print(df.head())
  
#     logger.info(f"Feature engineering completado. DataFrame resultante con {df.shape[1]} columnas")

#     return df

########################## NUEVA VERSIÓN CON PERCENTILES APROXIMADOS #############################

def feature_engineering_percentil(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """
    Genera variables de percentil aproximado para los atributos especificados,
    calculando previamente los límites de percentil por grupo (foto_mes)
    y luego asignándolos mediante un JOIN.
  
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list[str]
        Lista de atributos para los cuales generar los percentiles.
  
    Returns
    -------
    pd.DataFrame
        DataFrame con las variables de percentil agregadas
    """

    logger.info(f"Realizando feature engineering con percentiles aproximados para {len(columnas) if columnas else 0} atributos")

    if not columnas:
        logger.warning("No se especificaron atributos para generar percentiles")
        return df

         # para cada columna, generamos un bloque SQL
    for attr in columnas:
        if attr not in df.columns:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")
            continue

        # número de percentiles a calcular (por ejemplo 100)
        n_percentiles = 100
        percentiles = [round(i / n_percentiles, 2) for i in range(1, n_percentiles)]

        # CTE que calcula los límites
        sql_limites = f"""
        WITH limites AS (
            SELECT 
                foto_mes,
                unnest(quantile_cont({attr}, {percentiles})) AS valor_limite,
                unnest(range(1, {n_percentiles})) AS percentil
            FROM df
            GROUP BY foto_mes
        )
        """

        # Join para asignar el percentil a cada registro
        sql_join = f"""
        SELECT 
            d.*, 
            MAX(l.percentil) AS {attr}_percentil
        FROM df d
        JOIN limites l
            ON d.foto_mes = l.foto_mes
           AND d.{attr} >= l.valor_limite
        GROUP BY ALL
        """

        # Ejecutar la consulta SQL
        con = duckdb.connect(database=":memory:")
        con.register("df", df)
        df = con.execute(sql_limites + sql_join).df()
        con.close()

        logger.debug(f"Consulta SQL: {sql_limites + sql_join}")

    # con.close()
    logger.info(f"Feature engineering completado. DataFrame resultante con {df.shape[1]} columnas")

    return df


#######################################################################################
def feature_engineering_rank(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """
    Genera variables de ranking normalizado (0 a 1) para los atributos especificados utilizando SQL.
    Parameters:
    -----------
    df : pd.DataFrame

        DataFrame con los datos
    columnas : list 
        Lista de atributos para los cuales generar los rankings. Si es None, no se generan.
    Returns:
    --------
    pd.DataFrame    
        DataFrame con las variables de ranking agregadas
    """

    logger.info(f"Realizando feature engineering con ranking normalizado para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar rankings")
        return df

    # Construir la consulta SQL
    sql = "SELECT *"

    # Agregar los rankings para los atributos especificados
    for attr in columnas:
        if attr in df.columns:
            sql += f"\n, (DENSE_RANK() OVER (PARTITION BY foto_mes ORDER BY {attr}) - 1) * 1.0 / (COUNT(*) OVER (PARTITION BY foto_mes) - 1) AS {attr}_rank"
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    # Completar la consulta
    sql += " FROM df"

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df()
    con.close()

    logger.info(f"Feature engineering completado. DataFrame resultante con {df.shape[1]} columnas")

    return df
###################################################################################

def feature_engineering_drop(df: pd.DataFrame, columnas_a_eliminar: list[str]) -> pd.DataFrame:
    """
    Elimina las columnas especificadas del DataFrame.
  
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas_a_eliminar : list
        Lista de nombres de columnas a eliminar
  
    Returns:
    --------
    pd.DataFrame
        DataFrame con las columnas eliminadas
    """
  
    logger.info(f"Eliminando {len(columnas_a_eliminar) if columnas_a_eliminar else 0} columnas")
  
    if not columnas_a_eliminar:
        logger.warning("No se especificaron columnas para eliminar")
        return df
  
    columnas_existentes = [col for col in columnas_a_eliminar if col in df.columns]
    columnas_no_existentes = [col for col in columnas_a_eliminar if col not in df.columns]
  
    if columnas_no_existentes:
        logger.warning(f"Las siguientes columnas no existen en el DataFrame y no se pueden eliminar: {columnas_no_existentes}")
  
    df = df.drop(columns=columnas_existentes)
  
    logger.info(f"Columnas eliminadas. DataFrame resultante con {df.shape[1]} columnas")
  
    return df
####################################################################################

def feature_engineering_max_ultimos_n_meses(df: pd.DataFrame, columnas: list[str], n_meses: int = 3) -> pd.DataFrame:
    """
    Genera variables con el máximo de los últimos n meses para los atributos especificados utilizando SQL.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list
        Lista de atributos para los cuales generar el máximo de los últimos n meses. Si es None, no se generan.
    n_meses : int, default=3
        Cantidad de meses a considerar para calcular el máximo (incluye el mes actual).

    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de máximo de los últimos n meses agregadas
    """

    logger.info(f"Realizando feature engineering con máximo de los últimos {n_meses} meses para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar máximos")
        return df

    if n_meses < 1:
        logger.warning("La cantidad de meses debe ser al menos 1")
        return df

    # Construir la consulta SQL
    sql = "SELECT *"

    # Agregar el máximo de los últimos n meses para los atributos especificados
    for attr in columnas:
        if attr in df.columns:
            sql += (
                f"\n, max({attr}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes "
                f"ROWS BETWEEN {n_meses - 1} PRECEDING AND CURRENT ROW) AS {attr}_max_ult_{n_meses}m"
            )
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    # Completar la consulta
    sql += " FROM df"

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df()
    con.close()

    logger.info(f"Feature engineering completado. DataFrame resultante con {df.shape[1]} columnas")

    return df
#####################################################################################

def feature_engineering_min_ultimos_n_meses(df: pd.DataFrame, columnas: list[str], n_meses: int = 3) -> pd.DataFrame:
    """
    Genera variables con el mínimo de los últimos n meses para los atributos especificados utilizando SQL.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list
        Lista de atributos para los cuales generar el mínimo de los últimos n meses. Si es None, no se generan.
    n_meses : int, default=3
        Cantidad de meses a considerar para calcular el mínimo (incluye el mes actual).

    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de mínimo de los últimos n meses agregadas
    """

    logger.info(f"Realizando feature engineering con mínimo de los últimos {n_meses} meses para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar mínimos")
        return df

    if n_meses < 1:
        logger.warning("La cantidad de meses debe ser al menos 1")
        return df

    # Construir la consulta SQL
    sql = "SELECT *"

    # Agregar el mínimo de los últimos n meses para los atributos especificados
    for attr in columnas:
        if attr in df.columns:
            sql += (
                f"\n, min({attr}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes "
                f"ROWS BETWEEN {n_meses - 1} PRECEDING AND CURRENT ROW) AS {attr}_min_ult_{n_meses}m"
            )
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    # Completar la consulta
    sql += " FROM df"

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df()
    con.close()

    logger.info(f"Feature engineering completado. DataFrame resultante con {df.shape[1]} columnas")

    return df
###################################################
def feature_engineering_ratios(df: pd.DataFrame, ratios: list[dict]) -> pd.DataFrame:
    """
    Genera variables de ratio entre columnas especificadas utilizando SQL.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    ratios : list[dict]
        Lista de diccionarios con las claves 'numerador', 'denominador' y 'nombre' para cada ratio.
        Ejemplo: [{'numerador': 'col1', 'denominador': 'col2', 'nombre': 'col1_col2_ratio'}]

    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de ratio agregadas
    """
    logger.info(f"Realizando feature engineering de ratios para {len(ratios) if ratios else 0} combinaciones")

    if not ratios:
        logger.warning("No se especificaron ratios para generar")
        return df

    # Construir la consulta SQL
    sql = "SELECT *"
    for ratio in ratios:
        numerador = ratio.get("numerador")
        denominador = ratio.get("denominador")
        nombre = ratio.get("nombre", f"{numerador}_{denominador}_ratio")
        if numerador not in df.columns or denominador not in df.columns:
            logger.warning(f"Alguna columna no existe en el DataFrame: {numerador}, {denominador}")
            continue
        # Usar NULLIF para evitar división por cero
        sql += f", CASE WHEN {numerador} IS NULL OR {denominador} IS NULL THEN NULL ELSE {numerador} / NULLIF({denominador}, 0) END AS {nombre}"

    sql += " FROM df"

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df_result = con.execute(sql).df()
    con.close()

    logger.info(f"Feature engineering de ratios completado. DataFrame resultante con {df_result.shape[1]} columnas")
    return df_result
######################################################
def feature_engineering_cambio_estado(df: pd.DataFrame, columnas: list[str], cant_lag: int = 1) -> pd.DataFrame:
    """
    Genera variables binarias que indican si hubo un cambio en variables categóricas respecto al lag especificado.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list
        Lista de atributos categóricos para los cuales detectar cambios.
    cant_lag : int, default=1
        Cantidad de lags a comparar (por defecto 1)

    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de cambio categórico agregadas
    """
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
    df_result = con.execute(sql).df()
    con.close()

    logger.info(f"Feature engineering de cambio categórico completado. DataFrame resultante con {df_result.shape[1]} columnas")

    return df_result

#########################
FEATURES_CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "features.yaml")
def feature_engineering(
    # La operación 'ratios' puede generar más de una columna si 'numerador', 'denominador' y 'nombre' son listas.
    df: pd.DataFrame,
    competencia: str,
) -> pd.DataFrame:
    """
    Aplica múltiples técnicas de feature engineering sobre los atributos especificados en el archivo features.yaml,
    considerando la configuración para una competencia específica.

    Ejemplo de formato de features.yaml:
    ------------------------------------
    competencia1:
      lag:
        - columnas: [col1, col2]
          cant_lag: 1
        - columnas: [col1, col2]
          cant_lag: 2
      delta_lag:
        - columnas: [col3]
          cant_lag: 1
      percentil:
        - columnas: [col4]
      max:
        - columnas: [col5]
          n_meses: 3
      min:
        - columnas: [col6]
          n_meses: 2
      ratios:
        # Puede ser un solo ratio
        - numerador: col1
          denominador: col2
          nombre: col1_col2_ratio
        # O múltiples ratios en paralelo (listas del mismo largo)
        - numerador: [col3, col4]
          denominador: [col5, col6]
          nombre: [col3_col5_ratio, col4_col6_ratio]
        # O listas de numerador/denominador con un solo nombre (se generarán nombres automáticos)
        - numerador: [col7, col8]
          denominador: [col9, col10]
      cambio_estado:
        - columnas: [col7, col8]
          cant_lag: 1
    competencia2:
      lag:
        - columnas: [col7]
          cant_lag: 1
      percentil:
        - columnas: [col8]

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos originales.
    competencia : str
        Clave de la competencia dentro del archivo features.yaml.

    Returns:
    --------
    pd.DataFrame
        DataFrame con los atributos originales y los nuevos generados.
    """
    config = load_yaml_config(FEATURES_CONFIG)
    config_dict = vars(config)
    if competencia not in config_dict:
        logger.warning(f"La competencia '{competencia}' no está definida en el archivo de configuración {FEATURES_CONFIG}.yaml.")
        return df
    operaciones_config = vars(config_dict[competencia]) if isinstance(config_dict[competencia], object) else config_dict[competencia]
    df_result = df.copy()

    # Calcular la cantidad total de nuevas columnas a generar
    total_nuevas_columnas = 0
    for op, op_cfg in operaciones_config.items():
        if not op_cfg:
            continue
        configs = op_cfg if isinstance(op_cfg, list) else [op_cfg]
        for cfg in configs:
            # Convert SimpleNamespace to dict if needed
            if not isinstance(cfg, dict) and hasattr(cfg, "__dict__"):
                cfg = vars(cfg)
            if op == "ratios":
                # Puede ser una lista de dicts o un solo dict
                ratio_dicts = configs if isinstance(op_cfg, list) else [op_cfg]
                for ratio in ratio_dicts:
                    if not isinstance(ratio, dict) and hasattr(ratio, "__dict__"):
                        ratio = vars(ratio)
                    numerador = ratio.get("numerador")
                    denominador = ratio.get("denominador")
                    nombre = ratio.get("nombre")
                    # Si alguno es lista, todos deben ser listas de igual longitud
                    if isinstance(numerador, list) and isinstance(denominador, list) and isinstance(nombre, list):
                        total_nuevas_columnas += len(nombre)
                    elif isinstance(numerador, list) and isinstance(denominador, list):
                        total_nuevas_columnas += min(len(numerador), len(denominador))
                    elif isinstance(nombre, list):
                        total_nuevas_columnas += len(nombre)
                    else:
                        total_nuevas_columnas += 1
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
                total_nuevas_columnas += len(columnas) * cant_lag
            elif op in ("max", "min"):
                total_nuevas_columnas += len(columnas)
            elif op == "percentil":
                total_nuevas_columnas += len(columnas)
            elif op == "rank":
                total_nuevas_columnas += len(columnas)
            elif op == "delta_lag":
                cant_lag = cfg.get("cant_lag", 1) if isinstance(cfg, dict) else 1
                total_nuevas_columnas += len(columnas) * cant_lag
            elif op == "drop":
                total_nuevas_columnas -= len(columnas)
            elif op == "cambio_estado":
                cant_lag = cfg.get("cant_lag", 1) if isinstance(cfg, dict) else 1
                total_nuevas_columnas += len(columnas) * cant_lag
        for cfg in configs:
            # Convert SimpleNamespace to dict if needed
            if not isinstance(cfg, dict) and hasattr(cfg, "__dict__"):
                cfg = vars(cfg)
            if op == "ratios":
                ratio_dicts = configs if isinstance(op_cfg, list) else [op_cfg]
                expanded_ratios = []
                for ratio in ratio_dicts:
                    if not isinstance(ratio, dict) and hasattr(ratio, "__dict__"):
                        ratio = vars(ratio)
                    numerador = ratio.get("numerador")
                    denominador = ratio.get("denominador")
                    nombre = ratio.get("nombre")
                    # Si alguno es lista, expandir a varios ratios
                    if isinstance(numerador, list) and isinstance(denominador, list):
                        # Si nombre es lista, usarla, si no, generar nombres
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
               # logger.info(f"Operación '{op}' aplicada. Nombre de nuevas variables: {[r.get('nombre', f'{r.get('numerador')}_{r.get('denominador')}_ratio') for r in expanded_ratios]}")
                break  # ratios handled for all at once
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
                logger.info(f"Operación '{op}' aplicada. Nombre de nuevas variables: {[f'{col}_lag_{i}' for col in columnas for i in range(1, cant_lag + 1)]}")
            elif op == "delta_lag":
                cant_lag = cfg.get("cant_lag", 1) if isinstance(cfg, dict) else 1
                df_result = feature_engineering_delta_lag(df_result, columnas, cant_lag)
                logger.info(f"Operación '{op}' aplicada. Nombre de nuevas variables: {[f'{col}_delta_lag_{i}' for col in columnas for i in range(1, cant_lag + 1)]}")
            elif op == "percentil":
                df_result = feature_engineering_percentil(df_result, columnas)
                logger.info(f"Operación '{op}' aplicada. Nombre de nuevas variables: {[f'{col}_percentil' for col in columnas]}")
            elif op == "rank":
                df_result = feature_engineering_rank(df_result, columnas)
                logger.info(f"Operación '{op}' aplicada. Nombre de nuevas variables: {[f'{col}_rank' for col in columnas]}")
            elif op == "max":
                n_meses = cfg.get("n_meses", 3) if isinstance(cfg, dict) else 3
                df_result = feature_engineering_max_ultimos_n_meses(df_result, columnas, n_meses)
                logger.info(f"Operación '{op}' aplicada. Nombre de nuevas variables: {[f'{col}_max_ult_{n_meses}m' for col in columnas]}")
            elif op == "min":
                n_meses = cfg.get("n_meses", 3) if isinstance(cfg, dict) else 3
                df_result = feature_engineering_min_ultimos_n_meses(df_result, columnas, n_meses)
                logger.info(f"Operación '{op}' aplicada. Nombre de nuevas variables: {[f'{col}_min_ult_{n_meses}m' for col in columnas]}")
            elif op == "cambio_estado":
                cant_lag = cfg.get("cant_lag", 1) if isinstance(cfg, dict) else 1
                df_result = feature_engineering_cambio_estado(df_result, columnas, cant_lag)
                logger.info(f"Operación '{op}' aplicada. Nombre de nuevas variables: {[f'{col}_cambio_lag_{i}' for col in columnas for i in range(1, cant_lag + 1)]}")
            elif op == "drop": 
                df_result = feature_engineering_drop(df_result, columnas)
                logger.info(f"Operación '{op}' aplicada. Columnas eliminadas: {columnas}")
            else:
                logger.warning(f"Operación '{op}' no reconocida. Se omite.")

    return df_result
