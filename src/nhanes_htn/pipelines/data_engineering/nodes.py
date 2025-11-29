from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SBP_COLS = ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"]
DBP_COLS = ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"]


def merge_cycle(
    demo: pd.DataFrame,
    bmx: pd.DataFrame,
    bpx: pd.DataFrame,
    paq: pd.DataFrame,
    cycle_label: str,
) -> pd.DataFrame:
    """Une DEMO, BMX, BPX y PAQ de un ciclo y agrega la columna 'cycle'."""
    df = (
        demo
        .merge(bmx, on="SEQN", how="inner")
        .merge(bpx, on="SEQN", how="inner")
        .merge(paq, on="SEQN", how="left")
    )
    df["cycle"] = cycle_label
    return df


def concat_cycles(set1: pd.DataFrame, set2: pd.DataFrame) -> pd.DataFrame:
    """Concatena los dos ciclos en un solo DataFrame."""
    return pd.concat([set1, set2], ignore_index=True)


def filter_adults_examined(df: pd.DataFrame, min_age: int = 18) -> pd.DataFrame:
    """Filtra adultos (RIDAGEYR >= min_age). Si existe RIDSTATR, exige MEC."""
    mask = df["RIDAGEYR"] >= min_age
    if "RIDSTATR" in df.columns:
        mask &= df["RIDSTATR"] == 2
    return df.loc[mask].copy()


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra columnas a los nombres legibles que usaste en el notebook."""
    rename_map = {
        # DEMO
        "SEQN": "id",
        "RIDAGEYR": "age_years",
        "RIAGENDR": "gender_code",
        "RIDRETH3": "race_ethnicity_code",
        "DMDEDUC2": "education_level_code",
        "INDFMPIR": "income_poverty_ratio",
        # BMX
        "BMXWT": "weight_kg",
        "BMXHT": "height_cm",
        "BMXBMI": "bmi",
        "BMXWAIST": "waist_cm",
        # PAQ - transporte
        "PAQ635": "transport_activity_code",
        "PAQ640": "transport_activity_days",
        "PAD645": "transport_activity_minutes",
        # PAQ - recreación vigorosa
        "PAQ650": "vigorous_recreation_code",
        "PAQ655": "vigorous_recreation_days",
        "PAD660": "vigorous_recreation_minutes",
        # PAQ - recreación moderada
        "PAQ665": "moderate_recreation_code",
        "PAQ670": "moderate_recreation_days",
        "PAD675": "moderate_recreation_minutes",
        # PAQ - trabajo
        "PAQ620": "work_activity_code",
        "PAD630": "work_activity_minutes",
        # PAQ - sedentarismo
        "PAD680": "sedentary_minutes",
    }
    df = df.rename(columns=rename_map)
    return df


def compute_bp_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula SBP_mean, DBP_mean y HTN_label a partir de BPXSY/BPXDI."""

    df = df.copy()

    sbp_cols = [c for c in SBP_COLS if c in df.columns]
    dbp_cols = [c for c in DBP_COLS if c in df.columns]

    # Códigos 0 -> NaN
    df[sbp_cols] = df[sbp_cols].replace(0, np.nan)
    df[dbp_cols] = df[dbp_cols].replace(0, np.nan)

    df["SBP_mean"] = df[sbp_cols].mean(axis=1, skipna=True)
    df["DBP_mean"] = df[dbp_cols].mean(axis=1, skipna=True)

    # Definición clásica de HTA (>=140/90)
    df["HTN_label"] = (
        (df["SBP_mean"] >= 140) | (df["DBP_mean"] >= 90)
    ).astype("Int64")

    return df


def clean_activity_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la limpieza de códigos especiales (9999, 77, 99, 9) que hiciste en el notebook."""

    df = df.copy()

    activity_minutes_cols = [
        "transport_activity_minutes",
        "vigorous_recreation_minutes",
        "moderate_recreation_minutes",
        "work_activity_minutes",
        "sedentary_minutes",
    ]

    activity_days_cols = [
        "transport_activity_days",
        "vigorous_recreation_days",
        "moderate_recreation_days",
    ]

    # 9999 / 7777 -> NaN
    for col in activity_minutes_cols + activity_days_cols:
        if col in df.columns:
            df[col] = df[col].replace({7777: np.nan, 9999: np.nan})

    # minutos físicamente imposibles (>1440) -> NaN
    max_minutes_per_day = 1440
    for col in activity_minutes_cols:
        if col in df.columns:
            df.loc[df[col] > max_minutes_per_day, col] = np.nan

    # codes 'don't know' (9) -> NaN
    code_cols_to_nan = [
        "transport_activity_code",
        "vigorous_recreation_code",
        "moderate_recreation_code",
        "work_activity_code",
    ]
    for col in code_cols_to_nan:
        if col in df.columns:
            df.loc[df[col] == 9, col] = np.nan

    # days 'refused' (77) / 'don't know' (99) -> NaN
    days_cols_to_nan = [
        "vigorous_recreation_days",
        "transport_activity_days",
        "moderate_recreation_days",
    ]
    for col in days_cols_to_nan:
        if col in df.columns:
            df.loc[df[col].isin([77, 99]), col] = np.nan

    return df


def impute_and_cast(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa NaN según tu estrategia y ajusta tipos. Resultado = nhanes_supervised_dataset."""

    df_clean = df.copy()

    # 1) Imputación numérica con mediana
    num_median_cols = [
        "income_poverty_ratio",
        "weight_kg", "height_cm", "bmi", "waist_cm",
        "sedentary_minutes",
        "SBP_mean", "DBP_mean",
    ]
    for col in num_median_cols:
        if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
            median_value = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_value)

    # 2) Minutos/días de actividad física -> 0
    activity_zero_cols = [
        "transport_activity_minutes", "transport_activity_days",
        "vigorous_recreation_minutes", "vigorous_recreation_days",
        "moderate_recreation_minutes", "moderate_recreation_days",
        "work_activity_minutes",
    ]
    for col in activity_zero_cols:
        if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(0)

    # 3) Categóricas -> moda
    cat_mode_cols = [
        "education_level_code",
        "gender_code",
        "race_ethnicity_code",
        "transport_activity_code",
        "vigorous_recreation_code",
        "moderate_recreation_code",
        "work_activity_code",
    ]
    for col in cat_mode_cols:
        if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
            mode_value = df_clean[col].mode(dropna=True)[0]
            df_clean[col] = df_clean[col].fillna(mode_value)

    # 4) Ajuste de tipos
    int_cols = [
        "id",
        "gender_code",
        "race_ethnicity_code",
        "education_level_code",
        "transport_activity_code",
        "vigorous_recreation_code",
        "moderate_recreation_code",
        "work_activity_code",
        "HTN_label",
    ]
    for col in int_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype("int64")

    if "age_years" in df_clean.columns:
        df_clean["age_years"] = df_clean["age_years"].astype("int64")

    if "cycle" in df_clean.columns:
        df_clean["cycle"] = df_clean["cycle"].astype("category")

    return df_clean


def scale_features(
    df: pd.DataFrame,
    columns_to_scale: list[str],
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Aplica StandardScaler a las columnas numéricas indicadas.

    - `df`: DataFrame ya limpio (por ejemplo, nhanes_supervised_dataset).
    - `columns_to_scale`: lista de nombres de columnas a estandarizar.

    Devuelve:
      - df_scaled: copia del DataFrame con esas columnas escaladas.
      - scaler: objeto StandardScaler ajustado (para reutilizarlo en inferencia).
    """

    df_scaled = df.copy()

    # Filtramos solo columnas que realmente existan, por seguridad
    cols = [c for c in columns_to_scale if c in df_scaled.columns]

    if cols:
        scaler = StandardScaler()
        df_scaled[cols] = scaler.fit_transform(df_scaled[cols])
    else:
        # Si no hay columnas válidas, se devuelve un scaler "vacío"
        scaler = StandardScaler()

    return df_scaled, scaler
