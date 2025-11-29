from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# 1. Split común para regresión y clasificación
def split_data(
    features: pd.DataFrame,
    supervised: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Une features + targets y divide en train/test.

    - X: nhanes_features_selected
    - y_reg: SBP_mean
    - y_clf: HTN_label
    """
    # Alinear por índice
    df = features.join(
        supervised[["SBP_mean", "HTN_label"]],
        how="inner",
    )

    X = df[features.columns]
    y_reg = df["SBP_mean"]
    y_clf = df["HTN_label"].astype(int)

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X,
        y_reg,
        y_clf,
        test_size=test_size,
        random_state=random_state,
        stratify=y_clf,  # estratificado por hipertensión
    )

    return X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test


# 2. Modelos de regresión
def train_regression_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, object]:
    """Entrena al menos 2 modelos de regresión y devuelve un diccionario."""
    models: Dict[str, object] = {}

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models["linear_regression"] = lr

    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    models["random_forest"] = rf

    return models


def evaluate_regression_models(
    models: Dict[str, object],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
):
    """
    Calcula métricas (RMSE, R²) en train y test para cada modelo
    y selecciona el mejor según RMSE_test.
    """
    metrics: Dict[str, dict] = {}
    best_model_name = None
    best_rmse = np.inf

    for name, model in models.items():
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)

        rmse_train = float(np.sqrt(mse_train))
        rmse_test = float(np.sqrt(mse_test))

        r2_train = float(r2_score(y_train, y_pred_train))
        r2_test = float(r2_score(y_test, y_pred_test))

        metrics[name] = {
            "rmse_train": rmse_train,
            "rmse_test": rmse_test,
            "r2_train": r2_train,
            "r2_test": r2_test,
        }

        if rmse_test < best_rmse:
            best_rmse = rmse_test
            best_model_name = name

    best_model = models[best_model_name]
    metrics["best_model"] = best_model_name

    # outputs para el catálogo
    return best_model, metrics


# 3. Modelos de clasificación
def train_classification_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, object]:
    """Entrena ≥4 clasificadores para HTN_label."""
    models: Dict[str, object] = {}

    log_reg = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
    )
    log_reg.fit(X_train, y_train)
    models["logistic_regression"] = log_reg

    rf_clf = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf_clf.fit(X_train, y_train)
    models["random_forest"] = rf_clf

    svc = SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=42,
    )
    svc.fit(X_train, y_train)
    models["svc_rbf"] = svc

    knn = KNeighborsClassifier(n_neighbors=15)
    knn.fit(X_train, y_train)
    models["knn"] = knn

    return models


def evaluate_classification_models(
    models: Dict[str, object],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
):
    """
    Calcula accuracy, precision, recall, F1 y ROC-AUC en test para cada clasificador
    y selecciona el mejor según F1_test (prioriza equilibrio).
    """
    metrics: Dict[str, dict] = {}
    best_model_name = None
    best_f1 = -np.inf

    for name, model in models.items():
        y_pred_test = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_proba_test = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            # escalar 0–1
            y_proba_test = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            y_proba_test = None

        acc = float(accuracy_score(y_test, y_pred_test))
        prec = float(precision_score(y_test, y_pred_test))
        rec = float(recall_score(y_test, y_pred_test))
        f1 = float(f1_score(y_test, y_pred_test))
        auc = float(roc_auc_score(y_test, y_proba_test)) if y_proba_test is not None else None

        metrics[name] = {
            "accuracy_test": acc,
            "precision_test": prec,
            "recall_test": rec,
            "f1_test": f1,
            "roc_auc_test": auc,
        }

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    best_model = models[best_model_name]
    metrics["best_model"] = best_model_name

    return best_model, metrics
