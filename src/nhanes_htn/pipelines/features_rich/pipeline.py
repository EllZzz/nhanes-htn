from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_features_rich


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=build_features_rich,
                inputs=[
                    "nhanes_supervised_dataset",  # core_df
                    "smq_i",
                    "smqfam_i",
                    "alq_i",
                    "slq_i",
                    "dpq_i",
                    "smq_j",
                    "smqfam_j",
                    "alq_j",
                    "slq_j",
                    "dpq_j",
                ],
                outputs="nhanes_features_rich",
                name="build_features_rich_node",
            )
        ]
    )
