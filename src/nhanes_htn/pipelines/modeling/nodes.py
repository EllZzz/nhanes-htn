from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
    precision_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler


def _build_feature_matrices(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    A partir de nhanes_supervised_dataset construye:
    - X: features numéricas (sin id, SBP_mean ni HTN_label)
    - y_reg: target de regresión (SBP_mean)
    - y_clf: target de clasificación (HTN_label)
    """

    # Usamos solo columnas numéricas
    numeric = df.select_dtypes(include="number").copy()

    feature_cols = [
        c for c in numeric.columns
        if c not in ["SBP_mean", "HTN_label", "id"]
    ]

    X = numeric[feature_cols]
    y_reg = numeric["SBP_mean"]
    y_clf = numeric["HTN_label"].astype(int)

    return X, y_reg, y_clf


# --------------------------------------------------------------------
# 1) Regresión: predecir SBP_mean
# --------------------------------------------------------------------
def train_evaluate_regression(
    data: pd.DataFrame, params: Dict
) -> Tuple[SkPipeline, Dict]:
    """
    Entrena un modelo de regresión (RandomForest + StandardScaler)
    para predecir SBP_mean y devuelve:
    - modelo entrenado (pipeline sklearn)
    - métricas (RMSE, MAE, R2) en el set de prueba
    """

    X, y_reg, _ = _build_feature_matrices(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_reg,
        test_size=params.get("test_size", 0.2),
        random_state=params.get("random_state", 42),
    )

    model = SkPipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=params.get("n_estimators", 300),
                    random_state=params.get("random_state", 42),
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }

    return model, metrics


# --------------------------------------------------------------------
# 2) Clasificación: predecir HTN_label
# --------------------------------------------------------------------
def train_evaluate_classification(
    data: pd.DataFrame, params: Dict
) -> Tuple[SkPipeline, Dict]:
    """
    Entrena un modelo de clasificación (RandomForest + StandardScaler)
    para predecir HTN_label y devuelve:
    - modelo entrenado (pipeline sklearn)
    - métricas (accuracy, AUC, precision, recall, F1)
    """

    X, _, y_clf = _build_feature_matrices(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_clf,
        test_size=params.get("test_size", 0.2),
        random_state=params.get("random_state", 42),
        stratify=y_clf,
    )

    model = SkPipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=params.get("n_estimators", 300),
                    random_state=params.get("random_state", 42),
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics = {
        "accuracy": float(acc),
        "auc": float(auc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }

    return model, metrics


# --------------------------------------------------------------------
# 3) Clustering (no supervisado): KMeans sobre las mismas features
# --------------------------------------------------------------------
def run_clustering(
    data: pd.DataFrame, params: Dict
) -> Tuple[pd.DataFrame, Dict]:
    """
    Aplica KMeans sobre las features numéricas y devuelve:
    - df_clustered: mismo dataset con columna 'cluster'
    - métricas: silhouette_score
    """

    X, _, _ = _build_feature_matrices(data)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_clusters = params.get("n_clusters", 4)
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=params.get("random_state", 42),
        n_init=10,
    )

    labels = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)

    df_clustered = data.copy()
    df_clustered["cluster"] = labels

    metrics = {
        "n_clusters": int(n_clusters),
        "silhouette_score": float(sil),
    }

    return df_clustered, metrics
