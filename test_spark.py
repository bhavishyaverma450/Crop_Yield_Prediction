
import sys
from pyspark.sql import SparkSession

print("Python version:", sys.version)

spark = (
    SparkSession.builder
    .appName("CropYieldPredictionSetupTest")
    .master("local[*]")
    .getOrCreate()
)

print("Spark version:", spark.version)

data = [
    ("Rice", 2023, 1200.0),
    ("Wheat", 2023, 950.0),
    ("Maize", 2023, 700.0),
]

df = spark.createDataFrame(
    data,
    ["crop", "year", "production"]
)

df.show()
print("PySpark setup test successful!")

spark.stop()
