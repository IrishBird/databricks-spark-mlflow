from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession


def get_spark_session(app_name: str = "DBX ModelTraining") -> SparkSession:
    """Return an active SparkSession (e.g., Databricks/Connect) or create one."""
    active = SparkSession.getActiveSession()
    if active is not None:
        return active
    return SparkSession.builder.appName(app_name).getOrCreate()


def build_training_data(spark: SparkSession):
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
        (500.0, 45, "Hospital", "M54", "72100", 3, 0.0),
        (8500.0, 67, "Hospital", "I10", "99223", 12, 1.0),
        (250.0, 32, "PCP", "J20", "99213", 2, 0.0),
        (12500.0, 63, "Hospital", "I10", "99223", 15, 1.0),
        (310.0, 51, "Specialist", "E11", "99214", 7, 0.0),
        (9200.0, 72, "Hospital", "I25", "92928", 18, 1.0),
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
    """
    Save the model in the DBX volume
    """
    destination = "/Volumes/<team_name>/<project_name>/model-training-1"

    spark = get_spark_session()
    train_df = build_training_data(spark)
    train_df.show(truncate=False)

    pipeline = build_pipeline()
    spark_model = pipeline.fit(train_df)
    print(f"Fitted model type: {type(spark_model)}")

    spark_model.write().overwrite().save(destination)
    print(f"Saved model to: {destination}")


if __name__ == "__main__":
    main()
