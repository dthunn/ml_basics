# Import Spark session (already available in Databricks notebooks as 'spark')
from pyspark.sql import SparkSession

# Load CSV file from a URL or DBFS path
csv_url_or_path = "dbfs:/FileStore/mydata/data.csv"
# If loading from an HTTP URL, you might need to download it first to DBFS or mount an external storage

df = spark.read.option("header", "true").option("inferSchema", "true").csv(csv_url_or_path)

display(df)  # To preview data in Databricks notebook

# Define categorical and numeric columns (adjust based on your CSV)
categorical_cols = ['color', 'size']  # example categorical columns
numeric_cols = ['weight']  # example numeric columns

# Import required Spark ML classes
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

# Index categorical columns
indexers = [StringIndexer(inputCol=col, outputCol=col + "Index") for col in categorical_cols]

# One-hot encode indexed columns
encoders = [OneHotEncoder(inputCol=col + "Index", outputCol=col + "Vec") for col in categorical_cols]

# Assemble all features into one vector column "features"
assembler = VectorAssembler(
    inputCols=[col + "Vec" for col in categorical_cols] + numeric_cols,
    outputCol="features"
)

# Prepare pipeline for preprocessing
preprocessing_pipeline = Pipeline(stages=indexers + encoders + [assembler])

# Fit and transform the dataframe
preprocessed_df = preprocessing_pipeline.fit(df).transform(df)

# Import XGBoost for Spark
from sparkxgb import XGBoostClassifier  # Make sure this package is installed in your cluster!

# Initialize XGBoost classifier
xgb_classifier = XGBoostClassifier(
    featuresCol="features",
    labelCol="label",           # Adjust if your label column name differs
    predictionCol="prediction",
    objective="binary:logistic",
    numRound=100,
    maxDepth=5
)

# Split data into train and test
train_df, test_df = preprocessed_df.randomSplit([0.7, 0.3], seed=42)

# Train the XGBoost model
xgb_model = xgb_classifier.fit(train_df)

# Make predictions on test data
predictions = xgb_model.transform(test_df)

# Evaluate model performance
from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(labelCol="label")
auc = evaluator.evaluate(predictions)
print(f"Test AUC = {auc:.2f}")

# from sparkxgb import XGBoostClassifier  # Make sure xgboost4j-spark is installed

# # (Assuming df and preprocessing pipeline are already set up as before)

# xgb = XGBoostClassifier(
#     featuresCol="features",
#     labelCol="label",
#     predictionCol="prediction",
#     objective="binary:logistic"
# )

# pipeline = Pipeline(stages=indexers + encoders + [assembler, xgb])

# paramGrid = ParamGridBuilder() \
#     .addGrid(xgb.maxDepth, [3, 5]) \
#     .addGrid(xgb.numRound, [50, 100]) \
#     .build()

# evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

# crossval = CrossValidator(
#     estimator=pipeline,
#     estimatorParamMaps=paramGrid,
#     evaluator=evaluator,
#     numFolds=3,
#     parallelism=2
# )

# train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)

# cvModel = crossval.fit(train_df)

# predictions = cvModel.transform(test_df)

# auc = evaluator.evaluate(predictions)
# print(f"Test AUC = {auc:.3f}")

# best_xgb_model = cvModel.bestModel.stages[-1]
# print(f"Best maxDepth: {best_xgb_model.getMaxDepth()}")
# print(f"Best numRound: {best_xgb_model.getNumRound()}")