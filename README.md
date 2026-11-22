# Insurance Denial Prediction — Scripts Summary

This repository/workspace contains Python scripts and notebooks for training, logging, registering, inspecting, saving, and scoring binary insurance-denial classification models using **PySpark ML** and **scikit-learn** with **MLflow** on Databricks

---

## Project Overview

* **Domain & Task:** Binary classification (`denied`) predicting whether an insurance  will be denied
* **Input Features:**
  * **Numeric:** `insurance_amount`, `member_age`, `previous_insurance_count`
  * **Categorical:** `provider_type`, `diagnosis_code`, `procedure_code`
* **Target Column:** `denied` (binary label such as `0`/`1` or `0.0`/`1.0`)
* **Databricks / Unity Catalog Targets:**
  * **Catalog & Schema:** `<database_name>.<schema_name>`
  * **Registered Models:** `<database_name>.<schema_name>.<model_name>`
  * **Volumes Paths:** Unity Catalog Volume paths under `/Volumes/<team_name>/<project_name>/...` for Spark models, MLflow temp storage (`mlflow_tmp`), MLflow saved models, and exported `model.pkl` artifacts

---

## Scripts & Workflows Summary

### 1. PySpark ML Pipeline (`pyspark.ml`)
* **Pipeline Stages:** `StringIndexer` (`handleInvalid="keep"`) $\rightarrow$ `OneHotEncoder` $\rightarrow$ `VectorAssembler` (`features`) $\rightarrow$ `LogisticRegression` (`labelCol="denied"`, `predictionCol="prediction"`, `probabilityCol="probability"`)
* **Persistence & MLflow Spark:**
  * Saves/loads native `PipelineModel` directly to a Volume path (`write().overwrite().save(...)` / `PipelineModel.load(...)`
  * Logs and saves Spark models via `mlflow.spark.log_model(...)`, `mlflow.spark.save_model(...)`, and `mlflow.spark.load_model(...)` using `dfs_tmpdir` on a UC Volume
* **Scoring & Probability Extraction:** Transforms new insurance DataFrames, converts the `probability` vector via `vector_to_array`, extracts index `1` (`getItem(1)`), rounds, and casts to `string` (`score`)

### 2. scikit-learn + MLflow Training & Registration
* **Pipeline:** `ColumnTransformer` (`OneHotEncoder(handle_unknown="ignore")` on categorical features and `"passthrough"` on numeric features) paired with `LogisticRegression(max_iter=2000)` inside a scikit-learn `Pipeline`
* **MLflow Tracking & Signatures:** Configures `mlflow.set_tracking_uri("databricks")`, `mlflow.set_registry_uri("databricks-uc")`, and `mlflow.set_experiment(...)`, logs parameters/tags/artifacts inside `mlflow.start_run(...)`, and attaches `ModelSignature` / `infer_signature` and `input_example`
* **Type Alignment for Inference:** Casts incoming insurance DataFrames (`insurance_amount` to `float64`, `member_age` and `previous_insurance_count` to `int64`, and categorical columns to `string`) before calling `predict` and `predict_proba`

### 3. MLflow Registry Inspection, Artifact Export & Scoring
* **Version & `LoggedModel` Resolution:** Uses `MlflowClient.search_model_versions(...)` and parses `logged_models/(m-[0-9a-f]+)` from `version.source` with `mlflow.get_logged_model(model_id).source_run_id` when `version.run_id` is `None`
* **Artifact & Pickle Handling:** Lists run/metadata artifacts via `MlflowClient.list_artifacts(...)`, downloads artifacts with `mlflow.artifacts.download_artifacts(...)`, copies `model.pkl` to a UC Volume destination with `shutil.copy(...)`, and loads models either via `models:/<name>/<version>` (`mlflow.sklearn.load_model`) or via `pickle.load(...)`

---

# MLflow Model Lifecycle: `name` vs `registered_model_name`

## Question

What's the difference between `name` and `registered_model_name` in the following MLflow code?

```python
model_info = mlflow.sklearn.log_model(
    sk_model=pipeline,
    name="bigSky_pipeline",
    signature=signature,
    input_example=input_example,
    registered_model_name="registered_model_nm",
)
```

---

## Short Answer

Yes, they are **two different names for two different levels of the MLflow model lifecycle**.

| Parameter | Meaning | Think of it as |
|------------|----------|---------------|
| `name` | Name of the **logged model** associated with a specific MLflow run | The model artifact produced by this training run |
| `registered_model_name` | Name of the model in the **MLflow Model Registry / Unity Catalog** | The long-lived model identity that contains versions |

MLflow's own examples commonly use different values for these parameters, such as:

```python
name="sklearn-model"
registered_model_name="sk-learn-random-forest-reg-model"
```

---

## Understanding the Difference

### `name`

```python
name="bigSky_pipeline"
```

This tells MLflow:

> Log this trained pipeline as a **Logged Model** named `bigSky_pipeline`.

With MLflow 3, logged models are first-class objects that can store:

- Metrics
- Parameters
- Artifacts
- Signatures
- Input examples

## How `name` Is Used as the Artifact Path


## Example: Logging a Model

```python
mlflow.sklearn.log_model(
    sk_model=pipeline,
    name="bigSky_pipeline_uc",
    registered_model_name="prod_ml.ml_models.bigSky"
)
```

The value provided to `name` becomes the artifact directory within the MLflow run.

## Downloading Artifacts

When downloading artifacts, the `artifact_path` typically matches the model `name` used during logging.

```python
def export_model_pkl_to_volume(
    run_id: str = DEFAULT_RUN_ID,
    artifact_path: str = DEFAULT_ARTIFACT_NAME,
    destination: str = DEFAULT_PICKLE_DESTINATION,
) -> str:

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
```

## Example

If the model was logged as:

```python
name="bigSky_pipeline"
```

then download the artifacts using:

```python
local_path = mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path="bigSky_pipeline",
)
```

Therefore:

```text
artifact_path == name
```

because MLflow stores the model artifacts under the logged model name within the run.

## `name` vs `registered_model_name`

Suppose the model was logged as:

```python
name = "bigSky_pipeline_uc"

registered_model_name = (
    "prod_ml.claim_models.claim_bigSky"
)
```

Use the following when downloading artifacts:

```python
artifact_path="bigSky_pipeline_uc"
```

Do **not** use:

```python
artifact_path="prod_ml.claim_models.claim_bigSky"
```

`registered_model_name` belongs to the Model Registry / Unity Catalog layer, while `artifact_path` refers to the artifact location within a specific MLflow run.

##  Model

```text
MLflow Run
│
├── Artifacts
│   └── bigSky_pipeline_uc
│       ├── MLmodel
│       ├── model.pkl
│       └── conda.yaml
│
└── Registered Model
    prod_ml.claim_models.claim_bigSky
        ├── Version 1
        ├── Version 2
        └── Version N
```

## Rule of Thumb

| Task | Use |
|------|------|
| Download artifacts from a run | `name` (artifact path) |
| Access `model.pkl` from a run | `name` (artifact path) |
| Register a model | `registered_model_name` |
| Create model versions | `registered_model_name` |
| Load models using `models:/...` | `registered_model_name` |
| Deploy or serve models | `registered_model_name` |

> **Key takeaway:** Use the logged model `name` when working with artifacts inside an MLflow run. Use `registered_model_name` when working with Unity Catalog, model versions, aliases, deployment, or serving.