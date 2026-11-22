import mlflow.spark
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def get_spark_session(app_name: str = "PredictScoring") -> SparkSession:
    """Return an active SparkSession (e.g., Databricks) or create one."""
    active = SparkSession.getActiveSession()
    if active is not None:
        return active
    return SparkSession.builder.appName(app_name).getOrCreate()


def build_new_insurances_df(spark: SparkSession):
    columns = [
        "insurance_amount",
        "member_age",
        "provider_type",
        "diagnosis_code",
        "procedure_code",
        "previous_insurance_count",
    ]

    new_insurances_data = [
        (10000.00, 65, "Hospital", "I10", "99223", 13),
        (10000.00, 66, "Hospital", "I10", "99223", 15),
    ]

    return spark.createDataFrame(new_insurances_data, columns)


def add_score_column(predictions_df):
    return predictions_df.withColumn(
        "score",
        F.round(vector_to_array(F.col("probability")).getItem(1), 3).cast("string"),
    )


def score_with_mlflow_model(spark: SparkSession, model_uri: str, dfs_tmpdir: str):
    loaded_model = mlflow.spark.load_model(
        model_uri=model_uri,
        dfs_tmpdir=dfs_tmpdir,
    )
    new_insurances = build_new_insurances_df(spark)
    predictions = loaded_model.transform(new_insurances)
    return add_score_column(predictions)


def score_with_native_pipeline_model(spark: SparkSession, model_path: str):
    ml_model = PipelineModel.load(model_path)
    new_insurances = build_new_insurances_df(spark)
    predictions = ml_model.transform(new_insurances)
    return add_score_column(predictions)


def main():
    dfs_tmpdir = "/Volumes/<team_name>/<project_name>/mlflow_tmp"
    model_volume_path = "/Volumes/<team_name>/<project_name>/models/<model_name>"
    native_model_path = "/Volumes/<team_name>/<project_name>/insurance-training-1"

    spark = get_spark_session()

    # Option A: Load from MLflow volume path
    mlflow_scored = score_with_mlflow_model(
        spark=spark,
        model_uri=model_volume_path,
        dfs_tmpdir=dfs_tmpdir,
    )
    mlflow_scored.select(
        "insurance_amount",
        "member_age",
        "provider_type",
        "diagnosis_code",
        "procedure_code",
        "previous_insurance_count",
        "prediction",
        "probability",
        "score",
    ).show(truncate=False)

    # Option B: Load native Spark PipelineModel directly
    native_scored = score_with_native_pipeline_model(
        spark=spark,
        model_path=native_model_path,
    )
    native_scored.select(
        "insurance_amount",
        "provider_type",
        "diagnosis_code",
        "prediction",
        "probability",
        "score",
    ).show(truncate=False)


if __name__ == "__main__":
    main()
