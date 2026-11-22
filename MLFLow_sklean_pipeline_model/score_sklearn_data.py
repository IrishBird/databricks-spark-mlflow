import mlflow
import mlflow.sklearn
import pandas as pd
from constant import *

CATEGORICAL_FEATURES = [
    "provider_type",
    "diagnosis_code",
    "procedure_code",
]


def build_test_insurances() -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "insurance_amount": 10000.00,
                "member_age": 65,
                "provider_type": "Hospital",
                "diagnosis_code": "I10",
                "procedure_code": "99223",
                "previous_insurance_count": 13,
            },
            {
                "insurance_amount": 10000.00,
                "member_age": 66,
                "provider_type": "Hospital",
                "diagnosis_code": "I10",
                "procedure_code": "99223",
                "previous_insurance_count": 15,
            },
        ]
    )
    df["insurance_amount"] = df["insurance_amount"].astype("float64")
    df["member_age"] = df["member_age"].astype("int64")
    df["previous_insurance_count"] = df["previous_insurance_count"].astype("int64")
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("string")
    return df


def score_insurances(model_uri: str, insurances_df: pd.DataFrame) -> pd.DataFrame:
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    model = mlflow.sklearn.load_model(model_uri)
    predictions = model.predict(insurances_df)
    probabilities = model.predict_proba(insurances_df)

    scored = insurances_df.copy()
    scored["prediction"] = predictions
    scored["prob_not_denied"] = probabilities[:, 0]
    scored["prob_denied"] = probabilities[:, 1]
    return scored


def main() -> None:
    # Replace with a specific runs:/<run_id>/... or models:/<catalog.schema.model>/<version_or_alias> URI
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"
    insurances_df = build_test_insurances()

    try:
        scored = score_insurances(model_uri=model_uri, insurances_df=insurances_df)
        print(scored.to_string(index=False))
    except Exception as exc:
        print(
            f"Could not load model from '{model_uri}' ({exc}). "
            "Pass a valid runs:/... or models:/... URI to score_insurances()."
        )


if __name__ == "__main__":
    main()
