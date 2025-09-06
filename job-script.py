import logging
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
import time
from awsglue.utils import getResolvedOptions
import sys

# ------------------------
# Logging setup
# ------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

args = getResolvedOptions(sys.argv, ["JOB_NAME",])
test = args["JOB_NAME"]
handler = logging.StreamHandler()  # ensures logs go to stdout
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


logger.info("Starting Glue Spark job")

# ------------------------
# Spark setup
# ------------------------
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

# ------------------------
# Minimal Spark operation
# ------------------------
data = [("Alice", 34), ("Bob", 45), ("Charlie", 29)]
columns = ["name", "age"]

df = spark.createDataFrame(data, columns)

logger.info("Created DataFrame:")
df.show()  # this prints to stdout -> CloudWatch

# Example transformation
df_filtered = df.filter(df.age > 30)
logger.info("Filtered DataFrame:")
df_filtered.show()
output_path = f"s3://aws-glue-course-dthunn/{test}/df-output/"
df.write.mode("overwrite").parquet(output_path)

# Add a small sleep to ensure logs appear
time.sleep(5)

logger.info("Glue Spark job finished successfully")
