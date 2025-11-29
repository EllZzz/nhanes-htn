from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline
from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # 1) Merge por ciclo
            node(
                func=nodes.merge_cycle,
                inputs=dict(
                    demo="set1_demographic",
                    bmx="set1_body_measurements",
                    bpx="set1_blood_pressure",
                    paq="set1_questionnaire",
                    cycle_label="params:cycle_2015_2016_label",
                ),
                outputs="set1_merged",
                name="merge_cycle_2015_2016",
            ),
            node(
                func=nodes.merge_cycle,
                inputs=dict(
                    demo="set2_demographic",
                    bmx="set2_body_measurements",
                    bpx="set2_blood_pressure",
                    paq="set2_questionnaire",
                    cycle_label="params:cycle_2017_2018_label",
                ),
                outputs="set2_merged",
                name="merge_cycle_2017_2018",
            ),

            # 2) Concatenar ciclos
            node(
                func=nodes.concat_cycles,
                inputs=["set1_merged", "set2_merged"],
                outputs="nhanes_combined_raw",
                name="concat_cycles_2015_2018",
            ),

            # 3) Filtrar adultos
            node(
                func=nodes.filter_adults_examined,
                inputs=dict(
                    df="nhanes_combined_raw",
                    min_age="params:min_age",
                ),
                outputs="nhanes_adults_examined",
                name="filter_adults_examined",
            ),

            # 4) Renombrar columnas
            node(
                func=nodes.rename_columns,
                inputs="nhanes_adults_examined",
                outputs="nhanes_adults_renamed",
                name="rename_columns",
            ),

            # 5) Calcular SBP_mean, DBP_mean, HTN_label
            node(
                func=nodes.compute_bp_targets,
                inputs="nhanes_adults_renamed",
                outputs="nhanes_with_targets",
                name="compute_bp_targets",
            ),

            # 6) Limpiar códigos especiales de actividad
            node(
                func=nodes.clean_activity_codes,
                inputs="nhanes_with_targets",
                outputs="nhanes_activity_clean",
                name="clean_activity_codes",
            ),

            # 7) Imputar y ajustar tipos -> salida final
            node(
                func=nodes.impute_and_cast,
                inputs="nhanes_activity_clean",
                outputs="nhanes_supervised_dataset",
                name="impute_and_cast",
            ),

            # 8) Escalado de features
            node(
                func=nodes.scale_features,
                inputs=dict(
                    df="nhanes_supervised_dataset",
                    columns_to_scale="params:scale_columns",
                ),
                outputs=["nhanes_model_input_scaled", "feature_scaler"],
                name="scale_features",
            ),

            # 9) Selección de features finales
            node(
                func=nodes.select_features,
                inputs=dict(
                    df="nhanes_model_input_scaled",
                    feature_columns="params:selected_feature_columns",
                ),
                outputs="nhanes_features_selected",
                name="select_features",
            ),
        ]
    )
