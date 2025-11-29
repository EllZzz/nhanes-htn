from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline
from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.train_evaluate_regression,
                inputs=["nhanes_supervised_dataset", "params:regression"],
                outputs=["regression_model", "regression_metrics"],
                name="train_evaluate_regression",
            ),
            node(
                func=nodes.train_evaluate_classification,
                inputs=["nhanes_supervised_dataset", "params:classification"],
                outputs=["classification_model", "classification_metrics"],
                name="train_evaluate_classification",
            ),
            node(
                func=nodes.run_clustering,
                inputs=["nhanes_supervised_dataset", "params:clustering"],
                outputs=["clustering_dataset", "clustering_metrics"],
                name="run_clustering",
            ),
        ]
    )
