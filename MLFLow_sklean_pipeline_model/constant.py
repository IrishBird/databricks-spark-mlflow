CATALOG_NAME = "<database_name>"
SCHEMA_NAME = "<schema_name>"
MODEL_NAME = "<model_name>"
REGISTERED_MODEL_NAME = f"{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_NAME}"
EXPERIMENT_PATH = "/Users/<username>/model_ml_experiement"
TRAINING_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.data_training"


# export ML model

DEFAULT_RUN_ID = "<run_id>"
DEFAULT_ARTIFACT_NAME = "insurance_bigSky_pipeline"
DEFAULT_PICKLE_DESTINATION = (
    "/Volumes/<team_name>/<project_name>/model-training/model.pkl"
)
