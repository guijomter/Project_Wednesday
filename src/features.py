# src/features.py
import pandas as pd
import duckdb
import logging

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

####################################################################################

def feature_engineering_percentil(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """
    Genera variables de percentil para los atributos especificados utilizando SQL.
  
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list
        Lista de atributos para los cuales generar los percentiles. Si es None, no se generan 
  
    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de percentil agregadas
    """

    logger.info(f"Realizando feature engineering con percentiles para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar percentiles")
        return df
  
    # Construir la consulta SQL
    sql = "SELECT *"
  
    # Agregar los lags para los atributos especificados
    for attr in columnas:
        if attr in df.columns:
            sql += f"\n, ntile(100) over (partition by foto_mes order by {attr}) AS {attr}_percentil"
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