from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def run_kmeans_clustering(
    df_scaled: pd.DataFrame,
    n_clusters: int,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ejecuta KMeans sobre el dataset escalado (nhanes_model_input_scaled).

    Parámetros
    ----------
    df_scaled : DataFrame
        Datos de entrada ya estandarizados (por ejemplo, nhanes_model_input_scaled).
        Se asume que puede contener 'HTN_label', que se usa solo para evaluación.
    n_clusters : int
        Número de clusters KMeans.
    random_state : int
        Semilla para reproducibilidad.

    Retorna
    -------
    clustering_dataset : DataFrame
        DataFrame original + columna 'cluster'.
    clustering_metrics : dict
        Métricas globales (inertia, silhouette, tamaños, distribución HTN_label por cluster, etc.).
    """
    df = df_scaled.copy()

    # Separar features numéricas para clustering
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    has_label = "HTN_label" in feature_cols

    if has_label:
        feature_cols.remove("HTN_label")

    X = df[feature_cols].values

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    cluster_labels = kmeans.fit_predict(X)

    df["cluster"] = cluster_labels

    # Métricas
    metrics: Dict[str, Any] = {}
    metrics["n_clusters"] = int(n_clusters)
    metrics["inertia"] = float(kmeans.inertia_)

    try:
        metrics["silhouette_score"] = float(silhouette_score(X, cluster_labels))
    except Exception:
        metrics["silhouette_score"] = None

    # Tamaños de cluster
    unique, counts = np.unique(cluster_labels, return_counts=True)
    metrics["cluster_sizes"] = {int(k): int(v) for k, v in zip(unique, counts)}

    # Distribución de HTN_label por cluster si existe
    if "HTN_label" in df.columns:
        crosstab = pd.crosstab(df["cluster"], df["HTN_label"], normalize="index")
        metrics["htn_label_distribution_by_cluster"] = crosstab.to_dict()

    return df, metrics
