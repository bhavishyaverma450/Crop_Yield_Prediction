
import os
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("CropYieldDatasetInspection")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

data_folder = os.path.join(os.getcwd(), "data")

csv_files = [
    file for file in os.listdir(data_folder)
    if file.lower().endswith(".csv")
]

if not csv_files:
    raise FileNotFoundError(
        "No CSV file found in the data folder. "
        "Download and extract the Kaggle dataset first."
    )

file_path = os.path.join(data_folder, csv_files[0])
print("\nDataset file:", csv_files[0])

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(file_path)
)

print("\nDataset schema:")
df.printSchema()

print("\nFirst 10 records:")
df.show(10, truncate=False)

print("\nNumber of rows:", df.count())
print("Number of columns:", len(df.columns))

print("\nColumn names:")
print(df.columns)

print("\nMissing values by column:")
from pyspark.sql.functions import col, sum as spark_sum, when

df.select([
    spark_sum(
        when(col(c).isNull(), 1).otherwise(0)
    ).alias(c)
    for c in df.columns
]).show(truncate=False)

spark.stop()
