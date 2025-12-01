import numpy as np
import pandas as pd
from typing import Optional


# ============================================================
# 1. Diccionarios de nombres de columnas NHANES
#    (ajusta si tus archivos usan otros nombres)
# ============================================================

# 1.1 Tabaco (SMQ / SMQFAM)
SMOKING_COLS = {
    "id": "SEQN",
    "ever_100_cigs": "SMQ020",      # 1 = Yes, 2 = No, 7/9 = missing
    "current_status": "SMQ040",     # 1 = Every day, 2 = Some days, 3 = Not at all
    "age_start_regular": "SMD030",  # edad de inicio consumo regular
    "age_quit": "SMD055",           # edad de cese
    "cigs_per_day": "SMD650",       # cigarrillos/día (ajusta si usas otra variable)
}

# Tabaco pasivo en el hogar (SMQFAM, SMD***)
SMOKING_FAM_COLS = {
    "id": "SEQN",
    "n_smokers_home": "SMD460",      # # personas que fuman tabaco en el hogar
    "n_smokers_inside": "SMD470",    # # personas que fuman dentro de la casa
    "days_smoked_inside": "SMD480",  # # días en la última semana que alguien fumó dentro
}

# 1.2 Alcohol (ALQ)
ALCOHOL_COLS = {
    "id": "SEQN",
    "freq_12m": "ALQ120Q",      # frecuencia (número)
    "freq_12m_unit": "ALQ120U", # unidad (día/semana/mes/año)
    "drinks_per_day": "ALQ130", # tragos típicos por día de consumo
    "binge_days": "ALQ141Q",    # # de días con ≥4/5 tragos
    "binge_unit": "ALQ141U",    # unidad (semana/mes/año)
}

# 1.3 Sueño (SLQ / SLD)
SLEEP_COLS = {
    "id": "SEQN",
    "hours_sleep": "SLD012",        # horas habituales de sueño
    "snoring_freq": "SLQ030",       # frecuencia de ronquidos
    "apnea_obs": "SLQ040",          # frecuencia de "snort or stop breathing"
    "daytime_sleepiness": "SLQ120", # somnolencia diurna
}

# 1.4 Depresión (DPQ – PHQ-9)
DPQ_ITEMS = {
    "id": "SEQN",
    "items": [
        "DPQ010", "DPQ020", "DPQ030",
        "DPQ040", "DPQ050", "DPQ060",
        "DPQ070", "DPQ080", "DPQ090",
    ],
}


# ============================================================
# 2. Utilidad para recodificar códigos NHANES de “missing”
# ============================================================

def nhanes_missing_to_nan(
    series: pd.Series,
    extra_missing: Optional[set] = None,
) -> pd.Series:
    """
    Reemplaza códigos NHANES típicos de 'missing' por NaN.
    extra_missing permite agregar otros códigos específicos del ítem.
    """
    missing_codes = {7, 9, 77, 99, 777, 999, 9999}
    if extra_missing:
        missing_codes.update(extra_missing)
    return series.replace(list(missing_codes), np.nan)


# ============================================================
# 3. Derivación de features por módulo
# ============================================================

# 3.1 Tabaco (SMQ + SMQFAM)
def build_smoking_features(
    smq_df: pd.DataFrame,
    smqfam_df: pd.DataFrame,
    core_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye features de tabaco a partir de SMQ, SMQFAM y datos demográficos (para la edad).
    Retorna un DataFrame con una fila por participante (id/SEQN) y columnas derivadas.
    """

    # ---------- Subset y renombrado SMQ ----------
    cols = [SMOKING_COLS["id"]]
    for key in ["ever_100_cigs", "current_status", "age_start_regular", "age_quit", "cigs_per_day"]:
        col = SMOKING_COLS.get(key)
        if col in smq_df.columns:
            cols.append(col)

    smq_sub = smq_df[cols].copy()
    smq_sub = smq_sub.rename(columns={SMOKING_COLS["id"]: "id"})

    # Limpiar códigos de missing
    for key in ["ever_100_cigs", "current_status", "age_start_regular", "age_quit", "cigs_per_day"]:
        col = SMOKING_COLS.get(key)
        if col in smq_sub.columns:
            smq_sub[col] = nhanes_missing_to_nan(smq_sub[col])

    # Merge con edad actual desde el core
    if "id" in core_df.columns and "age_years" in core_df.columns:
        age_map = core_df[["id", "age_years"]].drop_duplicates()
        smq_sub = smq_sub.merge(age_map, on="id", how="left")

    ever_col = SMOKING_COLS.get("ever_100_cigs")
    status_col = SMOKING_COLS.get("current_status")

    # ---------- Indicadores ever/current/former/never ----------
    if ever_col in smq_sub.columns:
        smq_sub["ever_smoker"] = np.where(smq_sub[ever_col] == 1, 1, 0)
    else:
        smq_sub["ever_smoker"] = np.nan

    smq_sub["current_smoker"] = np.nan
    smq_sub["former_smoker"] = np.nan
    smq_sub["never_smoker"] = np.nan

    if status_col in smq_sub.columns:
        # 1 = every day, 2 = some days, 3 = not at all
        smq_sub["current_smoker"] = np.where(
            smq_sub[status_col].isin([1, 2]),
            1,
            np.where(smq_sub[status_col].isin([3]), 0, np.nan),
        )

        smq_sub["former_smoker"] = np.where(
            (smq_sub["ever_smoker"] == 1) & (smq_sub["current_smoker"] == 0),
            1,
            np.where(
                smq_sub["ever_smoker"].isna() | smq_sub["current_smoker"].isna(),
                np.nan,
                0,
            ),
        )

        smq_sub["never_smoker"] = np.where(
            (smq_sub["ever_smoker"] == 0) & (~smq_sub[status_col].isna()),
            1,
            np.where(
                smq_sub["ever_smoker"].isna() | smq_sub[status_col].isna(),
                np.nan,
                0,
            ),
        )

    # ---------- Años fumando y pack-years aprox ----------
    age_start_col = SMOKING_COLS.get("age_start_regular")
    age_quit_col = SMOKING_COLS.get("age_quit")
    cigs_col = SMOKING_COLS.get("cigs_per_day")

    smq_sub["years_smoking_aprox"] = np.nan

    if age_start_col in smq_sub.columns and "age_years" in smq_sub.columns:
        # Fumador actual: edad actual - edad inicio
        mask_current = smq_sub["current_smoker"] == 1
        smq_sub.loc[mask_current, "years_smoking_aprox"] = (
            smq_sub.loc[mask_current, "age_years"]
            - smq_sub.loc[mask_current, age_start_col]
        )

        # Exfumador: edad cese - edad inicio
        if age_quit_col in smq_sub.columns:
            mask_former = (smq_sub["former_smoker"] == 1) & (~smq_sub[age_quit_col].isna())
            smq_sub.loc[mask_former, "years_smoking_aprox"] = (
                smq_sub.loc[mask_former, age_quit_col]
                - smq_sub.loc[mask_former, age_start_col]
            )

    smq_sub.loc[smq_sub["years_smoking_aprox"] < 0, "years_smoking_aprox"] = np.nan

    if cigs_col in smq_sub.columns:
        smq_sub["pack_years_aprox"] = (
            smq_sub[cigs_col] * smq_sub["years_smoking_aprox"] / 20.0
        )
    else:
        smq_sub["pack_years_aprox"] = np.nan

    # pack_years_aprox = 0 para nunca fumadores (si sabemos que nunca fumó)
    smq_sub.loc[smq_sub["never_smoker"] == 1, "pack_years_aprox"] = 0.0

    # ---------- Tabaco pasivo (SMQFAM) ----------
    fam_cols = [SMOKING_FAM_COLS["id"]]
    for key in ["n_smokers_home", "n_smokers_inside", "days_smoked_inside"]:
        col = SMOKING_FAM_COLS.get(key)
        if col in smqfam_df.columns:
            fam_cols.append(col)

    smqfam_sub = smqfam_df[fam_cols].copy().rename(
        columns={SMOKING_FAM_COLS["id"]: "id"}
    )

    # Missing → NaN
    for key in ["n_smokers_home", "n_smokers_inside", "days_smoked_inside"]:
        col = SMOKING_FAM_COLS.get(key)
        if col in smqfam_sub.columns:
            smqfam_sub[col] = nhanes_missing_to_nan(smqfam_sub[col])

    # 0 = nadie; >=1 = exposición; NaN = desconocido
    if SMOKING_FAM_COLS.get("n_smokers_home") in smqfam_sub.columns:
        col = SMOKING_FAM_COLS["n_smokers_home"]
        smqfam_sub["secondhand_smoke_home"] = np.where(
            smqfam_sub[col].isna(),
            np.nan,
            np.where(smqfam_sub[col] >= 1, 1, 0),
        )
    else:
        smqfam_sub["secondhand_smoke_home"] = np.nan

    if SMOKING_FAM_COLS.get("n_smokers_inside") in smqfam_sub.columns:
        col = SMOKING_FAM_COLS["n_smokers_inside"]
        smqfam_sub["secondhand_smoke_inside"] = np.where(
            smqfam_sub[col].isna(),
            np.nan,
            np.where(smqfam_sub[col] >= 1, 1, 0),
        )
    else:
        smqfam_sub["secondhand_smoke_inside"] = np.nan

    # ---------- Frecuencia de humo dentro de casa ----------
    inside_col = SMOKING_FAM_COLS.get("n_smokers_inside")
    days_col = SMOKING_FAM_COLS.get("days_smoked_inside")

    smqfam_sub["smoke_inside_freq_cat"] = np.nan

    if inside_col in smqfam_sub.columns and days_col in smqfam_sub.columns:
        inside = smqfam_sub[inside_col]
        days = smqfam_sub[days_col]

        # Nadie fuma dentro de casa → "0_days"
        mask_no_inside = inside == 0
        smqfam_sub.loc[mask_no_inside, "smoke_inside_freq_cat"] = "0_days"

        # Hay fumadores dentro de casa
        mask_has_inside = inside >= 1

        smqfam_sub.loc[mask_has_inside & (days == 0), "smoke_inside_freq_cat"] = "0_days"
        smqfam_sub.loc[mask_has_inside & days.between(1, 3, inclusive="both"), "smoke_inside_freq_cat"] = "1-3_days"
        smqfam_sub.loc[mask_has_inside & days.between(4, 7, inclusive="both"), "smoke_inside_freq_cat"] = "4-7_days"
        # Si days es NaN en alguien con inside>=1, se deja NaN (desconocido)

    # ---------- Merge final ----------
    smoking_features = (
        smq_sub[["id", "current_smoker", "former_smoker", "never_smoker", "pack_years_aprox"]]
        .merge(
            smqfam_sub[["id", "secondhand_smoke_home", "secondhand_smoke_inside", "smoke_inside_freq_cat"]],
            on="id",
            how="left",
        )
        .drop_duplicates(subset="id")
    )

    return smoking_features


# 3.2 Alcohol (ALQ)
def categorize_alcohol(row: pd.Series) -> str:
    """
    Categoriza el nivel de consumo de alcohol usando drinks_per_week
    y, como respaldo, la frecuencia cruda (freq_12m).
    """
    dpw = row.get("drinks_per_week", np.nan)
    freq_raw = row.get(ALCOHOL_COLS["freq_12m"], np.nan)

    # Si explícitamente reporta frecuencia 0 → "none"
    if pd.notna(freq_raw) and freq_raw == 0:
        return "none"

    # Si no sabemos cuántos tragos/semana y tampoco tenemos frecuencia útil → "unknown"
    if pd.isna(dpw):
        return "unknown"

    if dpw == 0:
        return "none"
    if 0 < dpw <= 7:
        return "low"
    if 7 < dpw <= 14:
        return "moderate"
    return "heavy"


def build_alcohol_features(alq_df: pd.DataFrame) -> pd.DataFrame:
    cols = [ALCOHOL_COLS["id"]]
    for key in ["freq_12m", "freq_12m_unit", "drinks_per_day", "binge_days", "binge_unit"]:
        col = ALCOHOL_COLS.get(key)
        if col in alq_df.columns:
            cols.append(col)

    alq_sub = alq_df[cols].copy().rename(columns={ALCOHOL_COLS["id"]: "id"})

    # Limpiar missing
    for key in ["freq_12m", "freq_12m_unit", "drinks_per_day", "binge_days", "binge_unit"]:
        col = ALCOHOL_COLS.get(key)
        if col in alq_sub.columns:
            alq_sub[col] = nhanes_missing_to_nan(alq_sub[col])

    freq_col = ALCOHOL_COLS.get("freq_12m")
    unit_col = ALCOHOL_COLS.get("freq_12m_unit")

    # ---------- Días de consumo por semana ----------
    alq_sub["drinking_days_per_week"] = np.nan

    if freq_col in alq_sub.columns and unit_col in alq_sub.columns:
        f = alq_sub[freq_col]
        u = alq_sub[unit_col]

        # Supuesto de unidad:
        # 1 = per day, 2 = per week, 3 = per month, 4 = per year
        days_per_week = np.full(len(alq_sub), np.nan, dtype=float)
        days_per_week = np.where(u == 1, f * 7, days_per_week)
        days_per_week = np.where(u == 2, f, days_per_week)
        days_per_week = np.where(u == 3, f / 4.0, days_per_week)
        days_per_week = np.where(u == 4, f / 52.0, days_per_week)

        alq_sub["drinking_days_per_week"] = days_per_week

    # ---------- Tragos por semana ----------
    drinks_day_col = ALCOHOL_COLS.get("drinks_per_day")
    if drinks_day_col in alq_sub.columns:
        alq_sub["drinks_per_drinking_day"] = alq_sub[drinks_day_col]
    else:
        alq_sub["drinks_per_drinking_day"] = np.nan

    alq_sub["drinks_per_week"] = (
        alq_sub["drinking_days_per_week"] * alq_sub["drinks_per_drinking_day"]
    )

    # ---------- Binge drinking con ALQ141Q/ALQ141U ----------
    binge_days_col = ALCOHOL_COLS.get("binge_days")
    binge_unit_col = ALCOHOL_COLS.get("binge_unit")

    alq_sub["binge_days_per_month"] = np.nan
    alq_sub["binge_drinker"] = np.nan

    if binge_days_col in alq_sub.columns and binge_unit_col in alq_sub.columns:
        b = alq_sub[binge_days_col]
        u = alq_sub[binge_unit_col]

        # Misma lógica de unidades
        binge_per_year = np.full(len(alq_sub), np.nan, dtype=float)
        binge_per_year = np.where(u == 1, b * 365, binge_per_year)   # diario
        binge_per_year = np.where(u == 2, b * 52, binge_per_year)    # semanal
        binge_per_year = np.where(u == 3, b * 12, binge_per_year)    # mensual
        binge_per_year = np.where(u == 4, b, binge_per_year)         # anual

        binge_per_month = binge_per_year / 12.0
        alq_sub["binge_days_per_month"] = binge_per_month

        # Binge_drinker = 1 si ≥1 día de binge al mes; 0 si 0; NaN si no sabemos
        alq_sub.loc[binge_per_month >= 1, "binge_drinker"] = 1
        alq_sub.loc[binge_per_month == 0, "binge_drinker"] = 0

    # ---------- Categoría de consumo total ----------
    alq_sub["alcohol_level"] = alq_sub.apply(categorize_alcohol, axis=1)

    alcohol_features = alq_sub[
        ["id", "alcohol_level", "binge_drinker", "drinks_per_week"]
    ].drop_duplicates("id")

    return alcohol_features


# 3.3 Sueño (SLQ / SLD)
def build_sleep_features(slq_df: pd.DataFrame) -> pd.DataFrame:
    cols = [SLEEP_COLS["id"]]
    for key in ["hours_sleep", "snoring_freq", "apnea_obs", "daytime_sleepiness"]:
        col = SLEEP_COLS.get(key)
        if col in slq_df.columns:
            cols.append(col)

    slq_sub = slq_df[cols].copy().rename(columns={SLEEP_COLS["id"]: "id"})

    # Limpiar missing
    for key in ["hours_sleep", "snoring_freq", "apnea_obs", "daytime_sleepiness"]:
        col = SLEEP_COLS.get(key)
        if col in slq_sub.columns:
            slq_sub[col] = nhanes_missing_to_nan(slq_sub[col])

    # ---------- Categoría de duración de sueño ----------
    hours_col = SLEEP_COLS.get("hours_sleep")
    slq_sub["sleep_duration_cat"] = np.nan
    if hours_col in slq_sub.columns:
        h = slq_sub[hours_col]
        slq_sub["sleep_duration_cat"] = pd.cut(
            h,
            bins=[0, 5, 7, 8, 24],
            labels=["<=5h", "6-7h", "7-8h", ">8h"],
            right=True,
            include_lowest=True,
        )

    # ---------- Probable OSA ----------
    snore_col = SLEEP_COLS.get("snoring_freq")
    apnea_col = SLEEP_COLS.get("apnea_obs")
    sleepy_col = SLEEP_COLS.get("daytime_sleepiness")

    # Supuesto:
    # 1=Never, 2=Rarely, 3=Sometimes, 4=Often, 5=Almost always
    def probable_osa(row: pd.Series) -> float:
        snore = row.get(snore_col, np.nan)
        apnea = row.get(apnea_col, np.nan)
        sleepy = row.get(sleepy_col, np.nan)

        # Si no contestó nada del bloque → NaN
        if pd.isna(snore) and pd.isna(apnea) and pd.isna(sleepy):
            return np.nan

        snore_high = pd.notna(snore) and snore >= 3
        apnea_any = pd.notna(apnea) and apnea >= 2
        sleepy_high = pd.notna(sleepy) and sleepy >= 3

        if snore_high and (apnea_any or sleepy_high):
            return 1
        return 0

    slq_sub["probable_OSA"] = slq_sub.apply(probable_osa, axis=1)

    sleep_features = slq_sub[["id", "sleep_duration_cat", "probable_OSA"]].drop_duplicates("id")

    return sleep_features


# 3.4 Depresión (DPQ – PHQ-9)
def build_depression_features(dpq_df: pd.DataFrame) -> pd.DataFrame:
    cols = [DPQ_ITEMS["id"]] + [
        c for c in DPQ_ITEMS["items"] if c in dpq_df.columns
    ]
    dpq_sub = dpq_df[cols].copy().rename(columns={DPQ_ITEMS["id"]: "id"})

    # Limpiar missing en cada ítem
    for c in DPQ_ITEMS["items"]:
        if c in dpq_sub.columns:
            dpq_sub[c] = nhanes_missing_to_nan(dpq_sub[c])

    existing_items = [c for c in DPQ_ITEMS["items"] if c in dpq_sub.columns]
    dpq_sub["depression_score"] = dpq_sub[existing_items].sum(axis=1, min_count=1)

    def phq9_severity(score: float) -> Optional[str]:
        if pd.isna(score):
            return np.nan
        if score <= 4:
            return "none"
        if score <= 9:
            return "mild"
        if score <= 14:
            return "moderate"
        if score <= 19:
            return "moderately_severe"
        return "severe"

    dpq_sub["depression_severity"] = dpq_sub["depression_score"].apply(phq9_severity)

    depression_features = dpq_sub[
        ["id", "depression_score", "depression_severity"]
    ].drop_duplicates("id")

    return depression_features


# ============================================================
# 4. Función principal para Kedro
#    (concatena ciclos I y J dentro del nodo)
# ============================================================

def build_features_rich(
    core_df: pd.DataFrame,
    smq_i: pd.DataFrame,
    smqfam_i: pd.DataFrame,
    alq_i: pd.DataFrame,
    slq_i: pd.DataFrame,
    dpq_i: pd.DataFrame,
    smq_j: pd.DataFrame,
    smqfam_j: pd.DataFrame,
    alq_j: pd.DataFrame,
    slq_j: pd.DataFrame,
    dpq_j: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye el dataset de features 'ricas' para riesgo de hipertensión
    usando ambos ciclos NHANES (I y J).
    Pensado como nodo de Kedro.
    """

    # 4.1 Concatenar ciclos I y J por módulo
    smq_df = pd.concat([smq_i, smq_j], ignore_index=True)
    smqfam_df = pd.concat([smqfam_i, smqfam_j], ignore_index=True)
    alq_df = pd.concat([alq_i, alq_j], ignore_index=True)
    slq_df = pd.concat([slq_i, slq_j], ignore_index=True)
    dpq_df = pd.concat([dpq_i, dpq_j], ignore_index=True)

    # 4.2 Filtro de adultos en el core
    if "age_years" in core_df.columns:
        core_f = core_df[core_df["age_years"] >= 18].copy()
    else:
        core_f = core_df.copy()

    # 4.3 Reconstruir features por módulo
    smoking_f = build_smoking_features(smq_df, smqfam_df, core_f)
    alcohol_f = build_alcohol_features(alq_df)
    sleep_f = build_sleep_features(slq_df)
    depression_f = build_depression_features(dpq_df)

    # 4.4 Columnas base
    base_cols_local = [
        "id",
        "age_years",
        "gender_code",
        "race_ethnicity_code",
        "education_level_code",
        "income_poverty_ratio",
        "bmi",
        "waist_cm",
        "sedentary_minutes",
        "cycle",
        "HTN_label",
    ]
    base_cols_local = [c for c in base_cols_local if c in core_f.columns]
    features = core_f[base_cols_local].copy()

    # 4.5 Integración final
    features = (
        features
        .merge(smoking_f, on="id", how="left")
        .merge(alcohol_f, on="id", how="left")
        .merge(sleep_f, on="id", how="left")
        .merge(depression_f, on="id", how="left")
    )

    return features
