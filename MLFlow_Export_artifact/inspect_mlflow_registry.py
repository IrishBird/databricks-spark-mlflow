import re
from typing import Optional

import mlflow
from mlflow import MlflowClient
from constant import *


def get_registered_model_name(
    use_unity_catalog: bool = True,
    catalog_name: str = CATALOG_NAME,
    schema_name: str = SCHEMA_NAME,
    model_name: str = DEFAULT_MODEL_NAME,
) -> str:
    """Return 3-level UC model name or flat workspace registry name."""
    if use_unity_catalog:
        return f"{catalog_name}.{schema_name}.{model_name}"
    return model_name


def configure_mlflow(use_unity_catalog: bool = True) -> None:
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc" if use_unity_catalog else "databricks")


def resolve_source_run_id(version) -> tuple[Optional[str], Optional[str]]:
    """Resolve run_id directly or via MLflow 3 LoggedModel (m-<hex>) in version.source."""
    run_id = version.run_id
    match = re.search(r"logged_models/(m-[0-9a-f]+)", version.source or "")
    model_id = match.group(1) if match else None

    if run_id is None and model_id:
        run_id = mlflow.get_logged_model(model_id).source_run_id

    return model_id, run_id


def inspect_registered_versions(registered_model_name: str) -> list[dict]:
    """Search all versions of a registered model and resolve their MLflow 3 IDs."""
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{registered_model_name}'")

    resolved = []
    for version in versions:
        model_id, run_id = resolve_source_run_id(version)
        print("Version:", version.version)
        print("Model ID:", model_id)
        print("Run ID:", run_id)
        print("--------------------")
        resolved.append(
            {
                "version": version.version,
                "model_id": model_id,
                "run_id": run_id,
                "source": version.source,
            }
        )
    return resolved


def inspect_run_artifacts(
    run_id: str = DEFAULT_RUN_ID,
    metadata_subpath: str = "model_metadata",
) -> None:
    """List top-level run artifacts and files under model_metadata."""
    client = MlflowClient()

    artifacts = client.list_artifacts(run_id)
    for artifact in artifacts:
        print("path:", artifact.path, "| is_dir:", artifact.is_dir)

    model_files = client.list_artifacts(run_id, path=metadata_subpath)
    print(f"\nFiles in '{metadata_subpath}':")
    for f in model_files:
        print(f.path)


def main() -> None:
    use_uc = True
    configure_mlflow(use_unity_catalog=use_uc)

    registered_model_name = get_registered_model_name(
        use_unity_catalog=use_uc,
        model_name=DEFAULT_MODEL_NAME,
    )
    versions = inspect_registered_versions(registered_model_name)

    target_run_id = (
        versions[0]["run_id"] if versions and versions[0]["run_id"] else DEFAULT_RUN_ID
    )
    inspect_run_artifacts(run_id=target_run_id)


if __name__ == "__main__":
    main()
