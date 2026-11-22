import os
import shutil
from typing import Optional

import mlflow
import mlflow.sklearn
from mlflow.client import MlflowClient
import pandas as pd
from constant import *


def log_and_register_pipeline(
    pipeline,
    registered_model_name: str,
    signature,
    input_example: pd.DataFrame,
    artifact_name: str = DEFAULT_ARTIFACT_NAME,
):
    """Log and register a fitted sklearn pipeline inside an active MLflow run."""
    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError(
            "log_and_register_pipeline() must be called inside an active mlflow.start_run() context."
        )

    model_info = mlflow.sklearn.log_model(
        sk_model=pipeline,
        name=artifact_name,
        signature=signature,
        input_example=input_example,
        registered_model_name=registered_model_name,
    )

    print("Run ID:", active_run.info.run_id)
    print("Model URI:", model_info.model_uri)
    print("Registered Model:", registered_model_name)
    return model_info


def inspect_model_versions_and_artifacts(
    registered_model_name: str,
    run_id: str = DEFAULT_RUN_ID,
    metadata_path: str = "model_metadata",
) -> None:
    """List registered model versions and artifacts for a given MLflow run."""
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{registered_model_name}'")
    for version in versions:
        print("Version:", version.version)
        print("Run ID:", version.run_id)
        print("--------------------")

    artifacts = client.list_artifacts(run_id)
    for artifact in artifacts:
        print("path:", artifact.path, "| is_dir:", artifact.is_dir)

    artifact_path = "model"
    model_files = client.list_artifacts(run_id, path=metadata_path)
    print(f"Found {len(model_files)} entries under '{artifact_path}/{metadata_path}':")
    for item in model_files:
        print(" ", item.path)


def export_model_pkl_to_volume(
    run_id: str = DEFAULT_RUN_ID,
    artifact_path: str = DEFAULT_ARTIFACT_NAME,
    destination: str = DEFAULT_PICKLE_DESTINATION,
) -> str:
    """Download logged MLflow artifacts and copy model.pkl into a UC Volume path."""
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=artifact_path,
    )
    print("Downloaded artifacts to:", local_path)

    source = os.path.join(local_path, "model.pkl")
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copy(source, destination)
    print("Copied pickle to:", destination)
    return destination


def main() -> None:
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    inspect_model_versions_and_artifacts(
        registered_model_name=REGISTERED_MODEL_NAME,
        run_id=DEFAULT_RUN_ID,
    )


if __name__ == "__main__":
    main()
