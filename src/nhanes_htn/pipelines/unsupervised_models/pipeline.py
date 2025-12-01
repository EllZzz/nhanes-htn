from kedro.pipeline import Pipeline, node, pipeline

from .nodes import run_kmeans_clustering


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=run_kmeans_clustering,
                inputs=dict(
                    df_scaled="nhanes_model_input_scaled",          # entrada desde catalog
                    n_clusters="params:n_clusters_unsupervised",     # parámetro en parameters.yml
                ),
                outputs=["clustering_dataset", "clustering_metrics"],
                name="run_kmeans_clustering_node",
            ),
        ]
    )
