from __future__ import annotations

from kedro.pipeline import Pipeline
from nhanes_htn.pipelines import data_engineering


def register_pipelines() -> dict[str, Pipeline]:
    data_eng = data_engineering.create_pipeline()

    return {
        "data_engineering": data_eng,
        "__default__": data_eng,
    }
