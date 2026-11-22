import json
from typing import Optional

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.types.schema import ColSpec, DataType, Schema
from mlflow.types.type_hints import ModelSignature
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from constant import *

FEATURES = [
    "insurance_amount",
    "member_age",
    "provider_type",
    "diagnosis_code",
    "procedure_code",
    "previous_insurance_count",
]
NUMERIC_FEATURES = [
    "insurance_amount",
    "member_age",
    "previous_insurance_count",
]
CATEGORICAL_FEATURES = [
    "provider_type",
    "diagnosis_code",
    "procedure_code",
]
TARGET = "denied"


def load_training_dataframe(use_table: bool = False) -> pd.DataFrame:
    """Load training rows from a Delta/UC table if requested, else sample data."""
    if use_table:
        try:
            from pyspark.sql import SparkSession

            spark = SparkSession.getActiveSession()
            if spark is not None:
                return spark.table(TRAINING_TABLE).toPandas()
        except Exception as exc:
            print(f"Could not read table {TRAINING_TABLE} ({exc}); using sample data.")

    return pd.DataFrame(
        [
            [500.00, 45, "Hospital", "M54", "72100", 3, 0],
            [8500.00, 67, "Hospital", "I10", "99223", 12, 1],
            [250.00, 32, "PCP", "J20", "99213", 2, 0],
            [12500.00, 63, "Hospital", "I10", "99223", 15, 1],
            [310.00, 51, "Specialist", "E11", "99214", 7, 0],
            [9200.00, 72, "Hospital", "I25", "92928", 18, 1],
        ],
        columns=FEATURES + [TARGET],
    )


def prepare_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Cast feature and target columns to consistent types."""
    X = df[FEATURES].copy()
    X["insurance_amount"] = X["insurance_amount"].astype("float64")
    X["member_age"] = X["member_age"].astype("int64")
    X["previous_insurance_count"] = X["previous_insurance_count"].astype("int64")

    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("string")

    y = df[TARGET].astype("int64")
    return X, y


def build_pipeline(max_iter: int = 2000) -> Pipeline:
    """Build preprocessing + LogisticRegression pipeline."""
    preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
        ]
    )

    classifier = LogisticRegression(max_iter=max_iter)

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def build_explicit_signature() -> ModelSignature:
    """Build explicit MLflow schema matching your feature and prediction types."""
    input_schema = Schema(
        [
            ColSpec(DataType.double, "insurance_amount"),
            ColSpec(DataType.long, "member_age"),
            ColSpec(DataType.string, "provider_type"),
            ColSpec(DataType.string, "diagnosis_code"),
            ColSpec(DataType.string, "procedure_code"),
            ColSpec(DataType.long, "previous_insurance_count"),
        ]
    )
    output_schema = Schema([ColSpec(DataType.long)])
    return ModelSignature(inputs=input_schema, outputs=output_schema)


def main(max_iter: int = 2000, register_model: bool = True) -> None:
    print("Experiment:", EXPERIMENT_PATH)
    print("Registered model:", REGISTERED_MODEL_NAME)

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    # if specified as databrick-uc, then model will be saved in the catalog, can also specify  :  mlflow.set_registry_uri( "databricks-uc"
    mlflow.set_experiment(EXPERIMENT_PATH)

    df = load_training_dataframe(use_table=False)
    X, y = prepare_features_and_target(df)

    print("Feature types:")
    print(X.dtypes)
    print("\nTarget type:")
    print(y.dtype)
    print("\nTarget distribution:")
    print(y.value_counts())

    preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
        ]
    )
    print(preprocessor)

    pipeline = build_pipeline(max_iter=max_iter)
    print(pipeline)

    pipeline.fit(X, y)
    print("Model training completed successfully.")

    explicit_signature = build_explicit_signature()
    print("Explicit signature:", explicit_signature)

    input_example = X.head(2).copy()
    print(input_example)
    print("\nInput example types:")
    print(input_example.dtypes)

    inferred_signature = infer_signature(input_example, pipeline.predict(input_example))

    with mlflow.start_run(run_name="model-bigSky-demo"):
        mlflow.log_param("algorithm", "LogisticRegression")
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("training_records", len(df))
        mlflow.log_param("feature_count", len(FEATURES))

        mlflow.set_tags(
            {
                "business_domain": "insurances",
                "model_type": "binary_classification",
                "purpose": "learning_demo",
            }
        )

        mlflow.log_dict(
            {
                "features": FEATURES,
                "numeric_features": NUMERIC_FEATURES,
                "categorical_features": CATEGORICAL_FEATURES,
                "target": TARGET,
            },
            artifact_file="feature_definitions.json",
        )

        registered_name: Optional[str] = (
            REGISTERED_MODEL_NAME if register_model else None
        )
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model_artifact_path",
            registered_model_name=registered_name,
            signature=inferred_signature,
            input_example=input_example,
        )
        print("Logged model URI:", model_info.model_uri)

        # Quick smoke-test prediction inside the run
        test_insurance = pd.DataFrame(
            [
                {
                    "insurance_amount": 10000.00,
                    "member_age": 65,
                    "provider_type": "Hospital",
                    "diagnosis_code": "I10",
                    "procedure_code": "99223",
                    "previous_insurance_count": 13,
                }
            ]
        )
        test_insurance["insurance_amount"] = test_insurance["insurance_amount"].astype(
            "float64"
        )
        test_insurance["member_age"] = test_insurance["member_age"].astype("int64")
        test_insurance["previous_insurance_count"] = test_insurance[
            "previous_insurance_count"
        ].astype("int64")
        for col in CATEGORICAL_FEATURES:
            test_insurance[col] = test_insurance[col].astype("string")

        probability = pipeline.predict_proba(test_insurance)[0]
        print("Approval probability:", probability[0])
        print("Denial probability:", probability[1])
        print(
            "Test insurance payload:",
            json.dumps(test_insurance.to_dict(orient="records"), default=str),
        )


if __name__ == "__main__":
    main()
