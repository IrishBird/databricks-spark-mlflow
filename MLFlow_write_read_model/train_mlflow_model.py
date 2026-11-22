import mlflow
import mlflow.spark
from mlflow.models import infer_signature
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession


def get_spark_session(app_name: str = "ModelMLflowTraining") -> SparkSession:
    """Return an active SparkSession (e.g., Databricks) or create one."""
    active = SparkSession.getActiveSession()
    if active is not None:
        return active
    return SparkSession.builder.appName(app_name).getOrCreate()


def build_training_df(spark: SparkSession):
    columns = [
        "insurance_amount",
        "member_age",
        "provider_type",
        "diagnosis_code",
        "procedure_code",
        "previous_insurance_count",
        "denied",
    ]

    training_data = [
        (500.00, 45, "Hospital", "M54", "72100", 3, 0.0),
        (8500.00, 67, "Hospital", "I10", "99223", 12, 1.0),
        (250.00, 32, "PCP", "J20", "99213", 2, 0.0),
        (12500.00, 63, "Hospital", "I10", "99223", 15, 1.0),
        (310.00, 51, "Specialist", "E11", "99214", 7, 0.0),
        (9200.00, 72, "Hospital", "I25", "92928", 18, 1.0),
    ]

    return spark.createDataFrame(training_data, columns)


def build_pipeline() -> Pipeline:
    provider_indexer = StringIndexer(
        inputCol="provider_type",
        outputCol="provider_type_index",
        handleInvalid="keep",
    )
    diagnosis_indexer = StringIndexer(
        inputCol="diagnosis_code",
        outputCol="diagnosis_code_index",
        handleInvalid="keep",
    )
    procedure_indexer = StringIndexer(
        inputCol="procedure_code",
        outputCol="procedure_code_index",
        handleInvalid="keep",
    )

    encoder = OneHotEncoder(
        inputCols=[
            "provider_type_index",
            "diagnosis_code_index",
            "procedure_code_index",
        ],
        outputCols=[
            "provider_type_vector",
            "diagnosis_code_vector",
            "procedure_code_vector",
        ],
    )

    assembler = VectorAssembler(
        inputCols=[
            "insurance_amount",
            "member_age",
            "previous_insurance_count",
            "provider_type_vector",
            "diagnosis_code_vector",
            "procedure_code_vector",
        ],
        outputCol="features",
    )

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="denied",
        predictionCol="prediction",
        probabilityCol="probability",
        maxIter=100,
    )

    return Pipeline(
        stages=[
            provider_indexer,
            diagnosis_indexer,
            procedure_indexer,
            encoder,
            assembler,
            lr,
        ]
    )


def main():
    catalog_name = "<database_name>"
    schema_name = "<schema_name>"
    model_name = "<model_name>"
    registered_model_name = f"{catalog_name}.{schema_name}.{model_name}"

    experiment_path = "/User/<username>/experiement_path"
    dfs_tmpdir = "/Volumes/<team_name>/<project_name>/mlflow_tmp"
    model_volume_path = "/Volumes/<team_name>/<project_name>/models/<model_name>"

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(experiment_path)

    spark = get_spark_session()
    spark_df = build_training_df(spark)
    spark_pipeline = build_pipeline()

    with mlflow.start_run():
        spark_model = spark_pipeline.fit(spark_df)
        print(f"Fitted model type: {type(spark_model)}")

        input_df = spark_df.drop("denied")
        sample_output = spark_model.transform(input_df).select("prediction")
        signature = infer_signature(input_df.toPandas(), sample_output.toPandas())

        model_info = mlflow.spark.log_model(
            spark_model=spark_model,
            artifact_path="insurance_predict_pipeline",
            registered_model_name=registered_model_name,
            dfs_tmpdir=dfs_tmpdir,
            signature=signature,
        )
        print(f"Logged model URI: {model_info.model_uri}")

        mlflow.spark.save_model(
            spark_model=spark_model,
            path=model_volume_path,
            dfs_tmpdir=dfs_tmpdir,
        )
        print(f"Saved MLflow Spark model to: {model_volume_path}")


if __name__ == "__main__":
    main()
