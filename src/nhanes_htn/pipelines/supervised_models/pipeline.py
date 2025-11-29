from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # 1) Split train/test
            node(
                func=nodes.split_data,
                inputs=dict(
                    features="nhanes_features_selected",
                    supervised="nhanes_supervised_dataset",
                    test_size="params:supervised_test_size",
                    random_state="params:supervised_random_state",
                ),
                outputs=[
                    "X_train",
                    "X_test",
                    "y_reg_train",
                    "y_reg_test",
                    "y_clf_train",
                    "y_clf_test",
                ],
                name="split_data",
            ),

            # 2) Regresión
            node(
                func=nodes.train_regression_models,
                inputs=["X_train", "y_reg_train"],
                outputs="regression_models_candidate",
                name="train_regression_models",
            ),
            node(
                func=nodes.evaluate_regression_models,
                inputs=[
                    "regression_models_candidate",
                    "X_train",
                    "X_test",
                    "y_reg_train",
                    "y_reg_test",
                ],
                outputs=["regression_model", "regression_metrics"],
                name="evaluate_regression_models",
            ),

            # 3) Clasificación
            node(
                func=nodes.train_classification_models,
                inputs=["X_train", "y_clf_train"],
                outputs="classification_models_candidate",
                name="train_classification_models",
            ),
            node(
                func=nodes.evaluate_classification_models,
                inputs=[
                    "classification_models_candidate",
                    "X_train",
                    "X_test",
                    "y_clf_train",
                    "y_clf_test",
                ],
                outputs=["classification_model", "classification_metrics"],
                name="evaluate_classification_models",
            ),
        ]
    )
