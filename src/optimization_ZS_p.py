import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd  # Necesario para FLAML
import polars as pl
from flaml.default import preprocess_and_suggest_hyperparams

from .config import *
from .gain_function import calcular_ganancia

logger = logging.getLogger(__name__)


def _resolve_seed() -> int:
    """Obtiene la primera semilla de la lista de configuración."""
    if isinstance(SEMILLA, list):
        return SEMILLA[0]
    return int(SEMILLA)


def _split_train_validation(
    df: pl.DataFrame,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Divide los datos en train y validación según los meses de config.
    ASUME que df['clase_ternaria'] ya es binario (0/1).
    """
    logger.debug(f"Dividiendo datos ZS: Train={MES_TRAIN}, Val={MES_VALIDACION}")
    
    if isinstance(MES_TRAIN, list):
        df_train = df.filter(pl.col("foto_mes").is_in(MES_TRAIN))
    else:
        df_train = df.filter(pl.col("foto_mes") == MES_TRAIN)

    df_val = df.filter(pl.col("foto_mes") == MES_VALIDACION)

    # Aseguramos el tipo de dato, aunque ya debería estar correcto
    df_train = df_train.with_columns(pl.col("clase_ternaria").cast(pl.Int8))
    df_val = df_val.with_columns(pl.col("clase_ternaria").cast(pl.Int8))

    logger.debug(f"Tamaño Train ZS: {df_train.shape}, Tamaño Val ZS: {df_val.shape}")
    return df_train, df_val


def _prepare_matrices(
    df_train: pl.DataFrame,
    df_val: pl.DataFrame,
    feature_subset: Optional[Any] = None,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Prepara las matrices y realiza el handoff de Polars a Pandas/NumPy para FLAML.
    """
    if feature_subset is not None:
        X_train_pl = df_train.select(feature_subset)
        X_val_pl = df_val.select(feature_subset)
    else:
        # Excluimos 'clase_ternaria' y 'clase_peso' (si existe)
        cols_to_drop = [col for col in ["clase_ternaria", "clase_peso"] if col in df_train.columns]
        X_train_pl = df_train.drop(cols_to_drop)
        X_val_pl = df_val.drop(cols_to_drop)

    y_train = df_train.get_column("clase_ternaria").to_numpy(dtype=np.int8)
    y_val = df_val.get_column("clase_ternaria").to_numpy(dtype=np.int8)

    # Handoff a Pandas para FLAML
    X_train_pd = X_train_pl.to_pandas()
    X_val_pd = X_val_pl.to_pandas()

    return X_train_pd, y_train, X_val_pd, y_val


def _calcular_ganancia_desde_probabilidades(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float]:
    """
    Calcula ganancia máxima y umbral sugerido. (Sin cambios, usa NumPy).
    """
    if y_pred.ndim > 1:
        y_pred = y_pred[:, 1]

    ganancia_maxima, ganancias_acumuladas = calcular_ganancia(
        y_pred=y_pred, y_true=y_true
    )

    if ganancias_acumuladas.size > 0:
        orden = np.argsort(y_pred)[::-1]
        y_pred_sorted = y_pred[orden]
        idx_max = int(np.argmax(ganancias_acumuladas))
        umbral_sugerido = float(y_pred_sorted[idx_max])
    else:
        umbral_sugerido = 0.5

    return ganancia_maxima, umbral_sugerido


def preparar_datos_zero_shot(
    df: pl.DataFrame,
    feature_subset: Optional[Any] = None,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Wrapper que toma Polars y devuelve Pandas/NumPy listos para FLAML.
    """
    df_train, df_val = _split_train_validation(df)
    return _prepare_matrices(df_train, df_val, feature_subset=feature_subset)


def _sugerir_y_entrenar_con_flaml(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Entrena con FLAML. (Sin cambios, usa Pandas/NumPy).
    """
    (
        hyperparams,
        estimator_class,
        X_train_transformed,
        y_train_transformed,
        feature_transformer,
        label_transformer,
    ) = preprocess_and_suggest_hyperparams("classification", X_train, y_train, "lgbm")

    estimator_kwargs = dict(hyperparams)
    estimator_kwargs.setdefault("random_state", _resolve_seed())
    estimator_kwargs.setdefault("n_jobs", -1)

    modelo = estimator_class(**estimator_kwargs)
    modelo.fit(X_train_transformed, y_train_transformed)

    if feature_transformer is not None:
        X_val_transformed = feature_transformer.transform(X_val)
    else:
        X_val_transformed = X_val

    if hasattr(modelo, "predict_proba"):
        proba_val = modelo.predict_proba(X_val_transformed)[:, 1]
    else:
        proba_val = modelo.predict(X_val_transformed)

    if label_transformer is not None:
        y_val_transformed = label_transformer.transform(y_val)
    else:
        y_val_transformed = y_val

    return (
        hyperparams,
        modelo.get_params(),
        np.asarray(proba_val),
        np.asarray(y_val_transformed),
    )


def _construir_parametros_lightgbm(
    hyperparams: Dict[str, Any],
    modelo_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Construye diccionario de parámetros para LightGBM. (Sin cambios, usa dicts).
    """
    rename_map = {
        "n_estimators": "num_iterations",
        "subsample": "bagging_fraction",
        "colsample_bytree": "feature_fraction",
        "min_child_samples": "min_data_in_leaf",
        "n_jobs": "num_threads",
        "random_state": "seed",
    }

    combinados = {**modelo_params, **hyperparams}
    resultado: Dict[str, Any] = {}

    for clave, valor in combinados.items():
        nueva_clave = rename_map.get(clave, clave)
        resultado[nueva_clave] = valor

    resultado["objective"] = "binary"
    resultado["metric"] = "None"
    resultado["verbose"] = -1
    resultado["verbosity"] = -1
    resultado["seed"] = int(resultado.get("seed", _resolve_seed()))

    if "bagging_fraction" not in resultado and "subsample" in combinados:
        resultado["bagging_fraction"] = combinados["subsample"]
    if "feature_fraction" not in resultado and "colsample_bytree" in combinados:
        resultado["feature_fraction"] = combinados["colsample_bytree"]

    claves_a_eliminar = [
        "n_estimators", "subsample", "colsample_bytree", "n_jobs", "random_state"
    ]
    for clave in claves_a_eliminar:
        resultado.pop(clave, None)

    return resultado


def _persistir_resultados(
    archivo_base: str,
    params_flaml: Dict[str, Any],
    params_lightgbm: Dict[str, Any],
    ganancia_validacion: float,
    umbral_sugerido: float,
) -> Dict[str, str]:
    """
    Guarda los resultados en archivos JSON. (Usa conf. en lugar de variables globales).
    """
    os.makedirs("resultados", exist_ok=True)

    iter_path = os.path.join("resultados", f"{archivo_base}_zs_iteraciones.json")
    best_path = os.path.join("resultados", f"{archivo_base}_zs_best_params.json")

    contenido_existente = []
    trial_number = 0
    if os.path.exists(iter_path):
        try:
            with open(iter_path, "r", encoding="utf-8") as f:
                contenido_existente = json.load(f)
            if not isinstance(contenido_existente, list):
                contenido_existente = []
            trial_number = len(contenido_existente)
        except json.JSONDecodeError:
            contenido_existente = []
            trial_number = 0

    configuracion = {
        "semilla": SEMILLA if isinstance(SEMILLA, list) else [SEMILLA],
        "mes_train": MES_TRAIN if isinstance(MES_TRAIN, list) else [MES_TRAIN],
    }

    registro = {
        "trial_number": trial_number,
        "params": params_lightgbm,
        "value": float(ganancia_validacion),
        "datetime": datetime.now().isoformat(),
        "state": "COMPLETE",
        "configuracion": configuracion,
        "umbral_sugerido": umbral_sugerido,
        "params_flaml": params_flaml,
    }
    contenido_existente.append(registro)

    with open(iter_path, "w", encoding="utf-8") as f:
        json.dump(contenido_existente, f, indent=2)

    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(
            {"params_lightgbm": params_lightgbm, "params_flaml": params_flaml},
            f,
            indent=2,
        )

    return {"iteraciones": iter_path, "best_params": best_path}


def optimizar_zero_shot(
    df: pl.DataFrame,
    feature_subset: Optional[Any] = None,
    archivo_base: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Función principal de optimización ZS con Polars.
    """
    if archivo_base is None:
        archivo_base = conf.STUDY_NAME

    # 1. Preparar datos (Polars -> Pandas/NumPy)
    X_train, y_train, X_val, y_val = preparar_datos_zero_shot(df, feature_subset)

    # 2. Entrenar con FLAML
    (
        hyperparams,
        modelo_params,
        proba_val,
        y_val_transformed,
    ) = _sugerir_y_entrenar_con_flaml(X_train, y_train, X_val, y_val)

    # 3. Calcular ganancia
    ganancia_val, umbral_sugerido = _calcular_ganancia_desde_probabilidades(
        y_val_transformed.astype(np.int32),
        np.clip(proba_val, 0.0, 1.0),
    )

    # 4. Preparar y guardar resultados
    params_lightgbm = _construir_parametros_lightgbm(hyperparams, modelo_params)
    paths = _persistir_resultados(
        archivo_base,
        hyperparams,
        params_lightgbm,
        ganancia_val,
        umbral_sugerido,
    )

    logger.info(
        f"FLAML Zero-Shot - Ganancia VALID={ganancia_val:,.0f} | Umbral={umbral_sugerido:.4f}"
    )

    return {
        "ganancia_validacion": ganancia_val,
        "umbral_sugerido": umbral_sugerido,
        "best_params_lightgbm": params_lightgbm,
        "best_params_flaml": hyperparams,
        "paths": paths,
    }


def predecir_zero_shot(
    modelo,
    X: pl.DataFrame,
    umbral: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predice usando un modelo FLAML/scikit-learn, aceptando un DataFrame de Polars.
    """
    # Handoff a Pandas para predicción
    X_pd = X.to_pandas()
    
    proba = modelo.predict(X_pd)
    positive_proba = np.clip(proba, 0.0, 1.0)

    if umbral is None:
        umbral = 0.025  # Umbral por defecto

    predicciones = (positive_proba > umbral).astype(int)
    return predicciones, positive_proba