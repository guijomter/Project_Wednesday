# src/grafico_test.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from datetime import datetime
from .config import *
#from .gain_function_polars import calcular_ganancia_maxima_polars
from .gain_function import analisis_ganancia_completo_polars

logger = logging.getLogger(__name__)

def calcular_ganancia_acumulada_optimizada(y_true: np.ndarray, y_pred_proba: np.ndarray) -> tuple:
    """
    Calcula la ganancia acumulada ordenando las predicciones de mayor a menor probabilidad.
    Versión optimizada para grandes datasets.
  
    Args:
        y_true: Valores verdaderos (0 o 1)
        y_pred_proba: Probabilidades predichas
  
    Returns:
        tuple: (ganancias_acumuladas, indices_ordenados, umbral_optimo)
    """
    logger.info("Calculando ganancia acumulada optimizada...")
  
    # Ordenar por probabilidad descendente
    indices_ordenados = np.argsort(y_pred_proba)[::-1]
    y_true_ordenado = y_true[indices_ordenados]
    y_pred_proba_ordenado = y_pred_proba[indices_ordenados]
  
    # Calcular ganancia acumulada vectorizada
    ganancias_individuales = np.where(y_true_ordenado == 1, GANANCIA_ACIERTO, -COSTO_ESTIMULO)
    ganancias_acumuladas = np.cumsum(ganancias_individuales)
  
    # Encontrar el punto de ganancia máxima
    indice_maximo = np.argmax(ganancias_acumuladas)
    umbral_optimo = y_pred_proba_ordenado[indice_maximo]
  
    logger.info(f"Ganancia máxima: {ganancias_acumuladas[indice_maximo]:,.0f} en posición {indice_maximo}")
    logger.info(f"Umbral óptimo: {umbral_optimo:.6f}")
  
    return ganancias_acumuladas, indices_ordenados, umbral_optimo

def crear_grafico_ganancia_avanzado(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                   titulo_personalizado: str = None) -> str:
    """
    Crea un gráfico avanzado de ganancia acumulada con múltiples elementos informativos.
  
    Args:
        y_true: Valores verdaderos
        y_pred_proba: Probabilidades predichas
        titulo_personalizado: Título personalizado para el gráfico
  
    Returns:
        str: Ruta del archivo del gráfico guardado
    """
    logger.info("Generando gráfico de ganancia avanzado...")
  
    # Calcular ganancia acumulada
    ganancias_acumuladas, indices_ordenados, umbral_optimo = calcular_ganancia_acumulada_optimizada(y_true, y_pred_proba)
  
    # Encontrar estadísticas clave
    ganancia_maxima = np.max(ganancias_acumuladas)
    indice_maximo = np.argmax(ganancias_acumuladas)
  
    # Calcular puntos de referencia
    umbral_025 = 0.025
    clientes_sobre_025 = np.sum(y_pred_proba >= umbral_025)
  
    # Filtrar datos para visualización (solo mostrar región relevante)
    umbral_ganancia = ganancia_maxima * 0.6  # Mostrar desde 60% de la ganancia máxima
    indices_relevantes = ganancias_acumuladas >= umbral_ganancia
    x_relevante = np.where(indices_relevantes)[0]
    y_relevante = ganancias_acumuladas[indices_relevantes]
  
    # Configurar estilo del gráfico
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
  
    # Gráfico principal: Ganancia acumulada
    ax1.plot(x_relevante, y_relevante, color='blue', linewidth=3, label='Ganancia Acumulada', alpha=0.8)
  
    # Marcar ganancia máxima
    ax1.scatter(indice_maximo, ganancia_maxima, color='red', s=150, zorder=5, 
               label=f'Ganancia Máxima: {ganancia_maxima:,.0f}')
  
    # Líneas de referencia
    ax1.axvline(x=indice_maximo, color='red', linestyle='--', alpha=0.7, 
               label=f'Corte Óptimo (cliente {indice_maximo:,})')
    ax1.axvline(x=clientes_sobre_025, color='purple', linestyle='-.', alpha=0.8, linewidth=2,
               label=f'Umbral 0.025 (cliente {clientes_sobre_025:,})')
  
    # Anotación de ganancia máxima
    ax1.annotate(f'Máximo: {ganancia_maxima:,.0f}\nUmbral: {umbral_optimo:.4f}', 
                xy=(indice_maximo, ganancia_maxima),
                xytext=(indice_maximo + len(x_relevante) * 0.15, ganancia_maxima * 1.1),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, fontweight='bold', color='red',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='red'))
  
    # Configurar primer gráfico
    ax1.set_xlabel('Clientes ordenados por probabilidad', fontsize=12)
    ax1.set_ylabel('Ganancia Acumulada', fontsize=12)
    titulo = titulo_personalizado or f'Ganancia Acumulada Optimizada - {conf.STUDY_NAME}'
    ax1.set_title(titulo, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
  
    # Segundo gráfico: Distribución de probabilidades
    ax2.hist(y_pred_proba, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    ax2.axvline(x=umbral_optimo, color='red', linestyle='--', linewidth=2, 
               label=f'Umbral Óptimo: {umbral_optimo:.4f}')
    ax2.axvline(x=umbral_025, color='purple', linestyle='-.', linewidth=2, 
               label=f'Umbral 0.025')
  
    ax2.set_xlabel('Probabilidad Predicha', fontsize=12)
    ax2.set_ylabel('Densidad', fontsize=12)
    ax2.set_title('Distribución de Probabilidades Predichas', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
  
    # Ajustar layout
    plt.tight_layout()
  
    # Guardar gráfico con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("resultados", exist_ok=True)
    ruta_archivo = f"resultados/{conf.STUDY_NAME}_ganancia_avanzado_{timestamp}.png"
  
    plt.savefig(ruta_archivo, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
  
    # Guardar datos del gráfico en CSV
    ruta_datos = f"resultados/{conf.STUDY_NAME}_datos_ganancia_{timestamp}.csv"
    df_datos = pd.DataFrame({
        'posicion': range(len(ganancias_acumuladas)),
        'ganancia_acumulada': ganancias_acumuladas,
        'probabilidad_ordenada': y_pred_proba[indices_ordenados]
    })
    df_datos.to_csv(ruta_datos, index=False)
  
    logger.info(f"Gráfico avanzado guardado: {ruta_archivo}")
    logger.info(f"Datos guardados: {ruta_datos}")
  
    return ruta_archivo

def generar_reporte_visual_completo(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   titulo_estudio: str = None) -> dict:
    """
    Genera un reporte visual completo con todos los gráficos y análisis.
  
    Args:
        y_true: Valores verdaderos
        y_pred_proba: Probabilidades predichas
        titulo_estudio: Título personalizado para el estudio
  
    Returns:
        dict: Rutas de todos los archivos generados y estadísticas
    """
    logger.info("=== GENERANDO REPORTE VISUAL COMPLETO ===")
  
    titulo = titulo_estudio or f"Análisis Completo - {conf.STUDY_NAME}"
  
    # 1. Gráfico de ganancia avanzado
    ruta_ganancia = crear_grafico_ganancia_avanzado(y_true, y_pred_proba, titulo)
  
    # 2. Análisis con Polars para estadísticas precisas

    analisis_polars = analisis_ganancia_completo_polars(y_true, y_pred_proba)
  
    # 3. Guardar resumen del reporte
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_resumen = f"resultados/{conf.STUDY_NAME}_reporte_resumen_{timestamp}.json"
  
    reporte_completo = {
        'metadata': {
            'timestamp': timestamp,
            'titulo_estudio': titulo,
            'total_clientes': len(y_true),
            'distribucion_target': {
                'positivos': int(np.sum(y_true == 1)),
                'negativos': int(np.sum(y_true == 0)),
                'porcentaje_positivos': float(np.mean(y_true) * 100)
            }
        },
        'archivos_generados': {
            'grafico_ganancia': ruta_ganancia,
            'resumen_json': ruta_resumen
        },
        'analisis_polars': analisis_polars,
        'estadisticas_clave': {
            'ganancia_maxima': analisis_polars['ganancia_maxima'],
            'umbral_optimo': analisis_polars['umbral_optimo'],
            'clientes_optimos': analisis_polars['clientes_optimos'],
            'mejora_vs_025': analisis_polars['mejora_vs_025']
        }
    }
  
    # Guardar resumen en JSON
    import json
    with open(ruta_resumen, 'w') as f:
        json.dump(reporte_completo, f, indent=2, default=str)
  
    logger.info("=== REPORTE VISUAL COMPLETADO ===")
    logger.info(f"Archivos generados:")
    logger.info(f"  - Gráfico ganancia: {ruta_ganancia}")
    logger.info(f"  - Resumen JSON: {ruta_resumen}")
 #   logger.info(f"Ganancia máxima encontrada: {reporte_completo['estadisticas_clave']['ganancia_maxima']:,.0f}")
  
    return reporte_completo

# Función que recibe un dataframe con predicciones de probabilidad y valores verdaderos y genera el gráfico de test completo
def generar_grafico_test_completo(df: pd.DataFrame) -> str:
    """
    Genera el gráfico de test completo a partir de un DataFrame con predicciones y valores verdaderos.
  
    Args:
        df: DataFrame que contiene las columnas 'clase_ternaria' y 'probabilidad'
  
    Returns:
        str: Ruta del archivo del gráfico generado
    """
    if 'clase_ternaria' not in df.columns or 'probabilidad' not in df.columns:
        raise ValueError("El DataFrame debe contener las columnas 'clase_ternaria' y 'probabilidad'.")
  
    y_true = df['clase_ternaria'].values
    y_pred_proba = df['probabilidad'].values
  
    titulo = f"Gráfico de Test Completo - {conf.STUDY_NAME}"
  
    ruta_grafico = crear_grafico_ganancia_avanzado(y_true, y_pred_proba, titulo)
  
    return ruta_grafico

# Generar gráfico de test avanzado sin guardar en disco 
def crear_grafico_ganancia_avanzado_multi(y_true, y_pred_proba_list, seeds=None, labels=None, titulo_personalizado=None):
    """
    Grafica múltiples curvas de ganancia acumulada en el mismo gráfico.

    Args:
        y_true: array de valores verdaderos (0/1), común para todas las curvas
        y_pred_proba_list: lista de arrays de probabilidades predichas
        seeds: lista de identificadores (ej: semillas) para cada curva (opcional)
        labels: lista de etiquetas para cada curva (opcional, si se provee seeds, se ignora labels)
        titulo_personalizado: título personalizado para el gráfico (opcional)

    Returns:
        fig: objeto matplotlib Figure
    """
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(15, 7))

    n_curvas = len(y_pred_proba_list)
    colores = plt.cm.get_cmap('tab10', n_curvas)

    # Construir etiquetas usando seeds si están presentes
    if seeds is not None:
        etiquetas = [f"Seed {seed}" for seed in seeds]
    elif labels is not None:
        etiquetas = labels
    else:
        etiquetas = [f"Curva {i+1}" for i in range(n_curvas)]

    indices_maximos = []
    ganancias_maximas = []

    for idx, y_pred_proba in enumerate(y_pred_proba_list):
        ganancias_acumuladas, indices_ordenados, umbral_optimo = calcular_ganancia_acumulada_optimizada(
            np.array(y_true), np.array(y_pred_proba)
        )
        ganancia_maxima = np.max(ganancias_acumuladas)
        indice_maximo = np.argmax(ganancias_acumuladas)
        indices_maximos.append(indice_maximo)
        ganancias_maximas.append(ganancia_maxima)

        umbral_ganancia = ganancia_maxima * 0.6  # Mostrar desde 60% de la ganancia máxima
        indices_relevantes = ganancias_acumuladas >= umbral_ganancia
        x_relevante = np.where(indices_relevantes)[0]
        y_relevante = ganancias_acumuladas[indices_relevantes]


        ax.plot(
            x_relevante, #range(len(ganancias_acumuladas)),
            y_relevante,  #ganancias_acumuladas,
            label=f"{etiquetas[idx]} (max: {ganancia_maxima:,.0f})",
            linewidth=2,
            color=colores(idx)
        )

    resultados = pd.DataFrame({
        'seed': seeds,
        'indice_maximo': indices_maximos,
        'ganancia_maxima': ganancias_maximas    
    }).sort_values(by='ganancia_maxima', ascending=False)
    print(resultados)
       
    ax.set_xlabel('Clientes ordenados por probabilidad', fontsize=12)
    ax.set_ylabel('Ganancia Acumulada', fontsize=12)
    titulo = titulo_personalizado or f'Ganancia Acumulada Optimizada - {conf.STUDY_NAME}'
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    plt.show()

    #guardar gráfico con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("resultados", exist_ok=True)
    ruta_archivo = f"resultados/{conf.STUDY_NAME}_ganancia_multi_{timestamp}.png"
    plt.savefig(ruta_archivo, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Gráfico múltiple guardado: {ruta_archivo}")
    
    #plt.tight_layout()
    return resultados

################################################################

def crear_grafico_ganancia_envios(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                   titulo_personalizado: str = None) -> str:
    """
    Crea un gráfico de ganancia acumulada por cantidad de envios
  
    Args:
        y_true: Valores verdaderos
        y_pred_proba: Probabilidades predichas
        titulo_personalizado: Título personalizado para el gráfico
  
    Returns:
        str: Ruta del archivo del gráfico guardado
    """
    logger.info("Generando gráfico de ganancia por envíos...")
  
    ganancias_acumuladas, indices_ordenados, umbral_optimo = calcular_ganancia_acumulada_optimizada(
            np.array(y_true), np.array(y_pred_proba)
        )
    ganancia_maxima = np.max(ganancias_acumuladas)
    indice_maximo = np.argmax(ganancias_acumuladas)
    
    umbral_ganancia = ganancia_maxima * 0.6  # Mostrar desde 60% de la ganancia máxima
    indices_relevantes = ganancias_acumuladas >= umbral_ganancia
    x_relevante = np.where(indices_relevantes)[0]
    y_relevante = ganancias_acumuladas[indices_relevantes]

    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(15, 7))

    ax.plot(
        x_relevante, #range(len(ganancias_acumuladas)),
        y_relevante,  #ganancias_acumuladas,
    #    label=f"{etiquetas[idx]} (max: {ganancia_maxima:,.0f})",
        linewidth=2
    #    color=colores(idx)
    )

    # Configurar primer gráfico
    ax.set_xlabel('Clientes ordenados por probabilidad', fontsize=12)
    ax.set_ylabel('Ganancia Acumulada', fontsize=12)
    titulo = titulo_personalizado or f'Ganancia Acumulada Optimizada - {conf.STUDY_NAME}'
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
  
    plt.show()
  
    return fig