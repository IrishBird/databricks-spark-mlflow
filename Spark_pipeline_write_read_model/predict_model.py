from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from constant import *


def get_spark_session(app_name: str = "dataScoring") -> SparkSession:
    """Return an active SparkSession (e.g., Databricks/Connect) or create one."""
    active = SparkSession.getActiveSession()
    if active is not None:
        return active
    return SparkSession.builder.appName(app_name).getOrCreate()


def build_new_insurances(spark: SparkSession):
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


def score_insurances(spark: SparkSession, model_path: str, insurances_df):
    ml_model = PipelineModel.load(model_path)
    print(f"Loaded model type: {type(ml_model)}")

    result = ml_model.transform(insurances_df)

    scored_df = result.withColumn(
        "score",
        F.round(vector_to_array(F.col("probability")).getItem(1), 3).cast("string"),
    )
    return scored_df


def main():

    spark = get_spark_session()
    new_insurances = build_new_insurances(spark)
    scored = score_insurances(spark, destination, new_insurances)

    scored.select(
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


if __name__ == "__main__":
    main()
