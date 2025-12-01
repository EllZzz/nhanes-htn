from __future__ import annotations

from kedro.pipeline import Pipeline
from nhanes_htn.pipelines import data_engineering as de
from nhanes_htn.pipelines import supervised_models as sm
from nhanes_htn.pipelines import features_rich as fr


def register_pipelines() -> dict[str, Pipeline]:
    data_engineering = de.create_pipeline()
    supervised_models = sm.create_pipeline()
    features_rich = fr.create_pipeline()

    return {
        "__default__": data_engineering + features_rich + supervised_models,
        "data_engineering": data_engineering,
        "features_rich": features_rich,
        "supervised_models": supervised_models,
    }
