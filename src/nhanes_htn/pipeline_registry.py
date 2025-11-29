from __future__ import annotations

from kedro.pipeline import Pipeline
from nhanes_htn.pipelines import data_engineering as de
from nhanes_htn.pipelines import supervised_models as sm


def register_pipelines():
    data_engineering = de.create_pipeline()
    supervised_models = sm.create_pipeline()

    return {
        "__default__": data_engineering + supervised_models,
        "data_engineering": data_engineering,
        "supervised_models": supervised_models,
    }
