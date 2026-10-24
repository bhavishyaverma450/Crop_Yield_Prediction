
import os
import sys
import json
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    Imputer,
    VectorAssembler
)
from pyspark.ml.regression import (
    LinearRegression,
    DecisionTreeRegressor
)
from pyspark.ml.evaluation import RegressionEvaluator


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "crop_yield.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place crop_yield.csv inside the data folder."
    )

spark = (
    SparkSession.builder
    .appName("INT315_Crop_Yield_Prediction")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("=" * 60)
print("INT315 - CROP YIELD PREDICTION USING SPARK MLLIB")
print("=" * 60)
print("Python version:", sys.version.split()[0])
print("Spark version:", spark.version)


# --------------------------------------------------
# 2. Load and inspect the dataset
# --------------------------------------------------

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .option("mode", "DROPMALFORMED")
    .csv(DATA_PATH)
)

# Standardize column names by removing extra spaces.
for old_name in df.columns:
    new_name = old_name.strip()
    if old_name != new_name:
        df = df.withColumnRenamed(old_name, new_name)

required_columns = [
    "Crop", "Crop_Year", "Season", "State",
    "Area", "Production", "Annual_Rainfall",
    "Fertilizer", "Pesticide", "Yield"
]

missing_columns = [
    name for name in required_columns
    if name not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing expected columns: {missing_columns}\n"
        f"Available columns: {df.columns}"
    )

print("\nDataset schema:")
df.printSchema()

print("\nFirst five rows:")
df.show(5, truncate=False)

print("\nOriginal row count:", df.count())


# --------------------------------------------------
# 3. Data cleaning and preprocessing
# --------------------------------------------------

numeric_columns = [
    "Crop_Year", "Area", "Production",
    "Annual_Rainfall", "Fertilizer",
    "Pesticide", "Yield"
]

for name in numeric_columns:
    df = df.withColumn(
        name,
        F.col(name).cast("double")
    )

# Exclude missing and invalid target values.
df = df.filter(
    F.col("Yield").isNotNull()
    & (~F.isnan("Yield"))
    & (F.abs(F.col("Yield")) < float("inf"))
)

# Exclude records with invalid or non-positive area.
df = df.filter(
    F.col("Area").isNotNull()
    & (F.col("Area") > 0)
)

# Replace missing categorical values.
categorical_columns = ["Crop", "Season", "State"]

for name in categorical_columns:
    df = df.withColumn(
        name,
        F.when(
            F.col(name).isNull()
            | (F.trim(F.col(name)) == ""),
            "Unknown"
        ).otherwise(F.trim(F.col(name)))
    )

print("\nRows after basic cleaning:", df.count())

print("\nMissing values after cleaning:")
df.select([
    F.sum(
        F.when(F.col(c).isNull(), 1).otherwise(0)
    ).alias(c)
    for c in df.columns
]).show(truncate=False)

print("\nDescriptive statistics:")
df.select(
    "Area", "Annual_Rainfall",
    "Fertilizer", "Pesticide", "Yield"
).describe().show()


# --------------------------------------------------
# 4. Correlation heatmap
# --------------------------------------------------

# Production is included for descriptive analysis only.
# It is excluded from the model because Yield is
# generally calculated using production and area.

heatmap_columns = [
    "Crop_Year", "Area", "Production",
    "Annual_Rainfall", "Fertilizer",
    "Pesticide", "Yield"
]

pandas_df = df.select(heatmap_columns).toPandas()
correlation = pandas_df.corr(numeric_only=True)

plt.figure(figsize=(10, 7))
image = plt.imshow(
    correlation,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    aspect="auto"
)

plt.colorbar(image, label="Correlation coefficient")
plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45,
    ha="right"
)
plt.yticks(range(len(correlation.columns)), correlation.columns)

for row in range(len(correlation.columns)):
    for col in range(len(correlation.columns)):
        value = correlation.iloc[row, col]
        plt.text(
            col, row, f"{value:.2f}",
            ha="center", va="center", fontsize=8
        )

plt.title("Correlation Heatmap - Crop Yield Dataset")
plt.tight_layout()

heatmap_path = os.path.join(
    OUTPUT_DIR, "correlation_heatmap.png"
)
plt.savefig(heatmap_path, dpi=200)
plt.close()

print("\nCorrelation heatmap saved:", heatmap_path)


# --------------------------------------------------
# 5. Prepare features and split the dataset
# --------------------------------------------------

# Production is deliberately not used as a feature
# to avoid target leakage when predicting Yield.
numeric_features = [
    "Crop_Year", "Area",
    "Annual_Rainfall", "Fertilizer", "Pesticide"
]

categorical_features = ["Crop", "Season", "State"]

# Remove rows that have missing target values and
# convert target to a numeric label.
model_df = df.select(
    *(categorical_features + numeric_features + ["Yield"])
)

train_df, test_df = model_df.randomSplit(
    [0.8, 0.2], seed=42
)

train_df = train_df.cache()
test_df = test_df.cache()

print("\nTraining rows:", train_df.count())
print("Testing rows:", test_df.count())

if train_df.count() == 0 or test_df.count() == 0:
    raise ValueError("Training or testing dataset is empty.")


# --------------------------------------------------
# 6. Build reusable preprocessing stages
# --------------------------------------------------

indexers = [
    StringIndexer(
        inputCol=name,
        outputCol=f"{name}_index",
        handleInvalid="keep"
    )
    for name in categorical_features
]

encoder = OneHotEncoder(
    inputCols=[f"{name}_index" for name in categorical_features],
    outputCols=[f"{name}_encoded" for name in categorical_features],
    handleInvalid="keep"
)

imputer = Imputer(
    strategy="median",
    inputCols=numeric_features,
    outputCols=[f"{name}_imputed" for name in numeric_features]
)

assembler = VectorAssembler(
    inputCols=(
        [f"{name}_encoded" for name in categorical_features]
        + [f"{name}_imputed" for name in numeric_features]
    ),
    outputCol="features",
    handleInvalid="keep"
)

# Use the same split and preprocessing for both models.
common_stages = indexers + [encoder, imputer, assembler]


# --------------------------------------------------
# 7. Define evaluation metrics
# --------------------------------------------------

rmse_evaluator = RegressionEvaluator(
    labelCol="Yield",
    predictionCol="prediction",
    metricName="rmse"
)

mae_evaluator = RegressionEvaluator(
    labelCol="Yield",
    predictionCol="prediction",
    metricName="mae"
)

r2_evaluator = RegressionEvaluator(
    labelCol="Yield",
    predictionCol="prediction",
    metricName="r2"
)


# --------------------------------------------------
# 8. Train and evaluate Linear Regression
# --------------------------------------------------

print("\nTraining Linear Regression...")

linear_regression = LinearRegression(
    featuresCol="features",
    labelCol="Yield",
    predictionCol="prediction",
    maxIter=50,
    regParam=0.01
)

linear_pipeline = Pipeline(
    stages=common_stages + [linear_regression]
)

linear_model = linear_pipeline.fit(train_df)
linear_predictions = linear_model.transform(test_df).cache()

linear_rmse = rmse_evaluator.evaluate(linear_predictions)
linear_mae = mae_evaluator.evaluate(linear_predictions)
linear_r2 = r2_evaluator.evaluate(linear_predictions)

print("\nLINEAR REGRESSION RESULTS")
print(f"RMSE: {linear_rmse:.4f}")
print(f"MAE : {linear_mae:.4f}")
print(f"R2  : {linear_r2:.4f}")


# --------------------------------------------------
# 9. Train and evaluate Decision Tree Regression
# --------------------------------------------------

print("\nTraining Decision Tree Regression...")

decision_tree = DecisionTreeRegressor(
    featuresCol="features",
    labelCol="Yield",
    predictionCol="prediction",
    maxDepth=8,
    seed=42
)

tree_pipeline = Pipeline(
    stages=common_stages + [decision_tree]
)

tree_model = tree_pipeline.fit(train_df)
tree_predictions = tree_model.transform(test_df).cache()

tree_rmse = rmse_evaluator.evaluate(tree_predictions)
tree_mae = mae_evaluator.evaluate(tree_predictions)
tree_r2 = r2_evaluator.evaluate(tree_predictions)

print("\nDECISION TREE RESULTS")
print(f"RMSE: {tree_rmse:.4f}")
print(f"MAE : {tree_mae:.4f}")
print(f"R2  : {tree_r2:.4f}")


# --------------------------------------------------
# 10. Compare models and save results
# --------------------------------------------------

metrics = [
    ("Linear Regression", linear_rmse, linear_mae, linear_r2),
    ("Decision Tree Regression", tree_rmse, tree_mae, tree_r2)
]

print("\nMODEL COMPARISON")
print("-" * 65)
print(f"{'Model':<28}{'RMSE':>12}{'MAE':>12}{'R2':>12}")
print("-" * 65)

for name, rmse, mae, r2 in metrics:
    print(f"{name:<28}{rmse:>12.4f}{mae:>12.4f}{r2:>12.4f}")

# Lower RMSE is the primary selection criterion.
best_name, best_rmse, best_mae, best_r2 = min(
    metrics, key=lambda item: item[1]
)

print("\nBest model by test RMSE:", best_name)

metrics_path = os.path.join(OUTPUT_DIR, "model_metrics.csv")

with open(metrics_path, "w", encoding="utf-8") as file:
    file.write("Model,RMSE,MAE,R2\n")
    for name, rmse, mae, r2 in metrics:
        file.write(f'"{name}",{rmse},{mae},{r2}\n')

# Save test predictions from the better model.
best_predictions = (
    linear_predictions
    if best_name == "Linear Regression"
    else tree_predictions
)

prediction_path = os.path.join(OUTPUT_DIR, "predictions")

(
    best_predictions
    .select("Yield", "prediction")
    .withColumnRenamed("Yield", "actual_yield")
    .withColumnRenamed("prediction", "predicted_yield")
    .write.mode("overwrite")
    .option("header", True)
    .csv(prediction_path)
)

# Save a small preview for the report.
best_predictions.select(
    "Crop", "Season", "State",
    "Yield", "prediction"
).show(20, truncate=False)

# Save a compact summary for the report.
summary = {
    "spark_version": spark.version,
    "training_rows": train_df.count(),
    "testing_rows": test_df.count(),
    "linear_regression": {
        "RMSE": linear_rmse,
        "MAE": linear_mae,
        "R2": linear_r2
    },
    "decision_tree": {
        "RMSE": tree_rmse,
        "MAE": tree_mae,
        "R2": tree_r2
    },
    "best_model_by_test_RMSE": best_name
}

summary_path = os.path.join(OUTPUT_DIR, "result_summary.json")
with open(summary_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4)

print("\nOUTPUT FILES")
print("Metrics:", metrics_path)
print("Predictions folder:", prediction_path)
print("Heatmap:", heatmap_path)
print("Summary:", summary_path)
print("\nPROJECT EXECUTION COMPLETED SUCCESSFULLY!")

# --------------------------------------------------
# 11. Additional visualizations for the project report
# --------------------------------------------------

print("\nGenerating additional result graphs...")

# Graph 2: Actual vs Predicted Yield
plot_df = (
    best_predictions
    .select("Yield", "prediction")
    .orderBy(F.rand(seed=42))
    .limit(2000)
    .toPandas()
    .dropna()
)

plt.figure(figsize=(8, 6))
plt.scatter(
    plot_df["Yield"],
    plot_df["prediction"],
    alpha=0.5
)

lower = min(
    plot_df["Yield"].min(),
    plot_df["prediction"].min()
)
upper = max(
    plot_df["Yield"].max(),
    plot_df["prediction"].max()
)

plt.plot(
    [lower, upper],
    [lower, upper],
    linestyle="--"
)
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title(f"Actual vs Predicted Yield - {best_name}")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"),
    dpi=200
)
plt.close()


# Graph 3: RMSE comparison
model_names = [item[0] for item in metrics]
rmse_values = [item[1] for item in metrics]

plt.figure(figsize=(8, 5))
bars = plt.bar(model_names, rmse_values)
plt.bar_label(bars, fmt="%.3f")
plt.ylabel("RMSE")
plt.title("RMSE Comparison of Regression Models")
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "rmse_comparison.png"),
    dpi=200
)
plt.close()


# Graph 4: MAE comparison
mae_values = [item[2] for item in metrics]

plt.figure(figsize=(8, 5))
bars = plt.bar(model_names, mae_values)
plt.bar_label(bars, fmt="%.3f")
plt.ylabel("MAE")
plt.title("MAE Comparison of Regression Models")
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "mae_comparison.png"),
    dpi=200
)
plt.close()


# Graph 5: R-squared comparison
r2_values = [item[3] for item in metrics]

plt.figure(figsize=(8, 5))
bars = plt.bar(model_names, r2_values)
plt.bar_label(bars, fmt="%.3f")
plt.axhline(0, linewidth=0.8)
plt.ylabel("R-squared (R²)")
plt.title("R² Comparison of Regression Models")
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "r2_comparison.png"),
    dpi=200
)
plt.close()


# Graph 6: Top 10 crops by average yield
crop_yield = (
    df.groupBy("Crop")
    .agg(F.avg("Yield").alias("average_yield"))
    .orderBy(F.desc("average_yield"))
    .limit(10)
    .toPandas()
    .sort_values("average_yield")
)

plt.figure(figsize=(10, 6))
plt.barh(
    crop_yield["Crop"],
    crop_yield["average_yield"]
)
plt.xlabel("Average Yield")
plt.ylabel("Crop")
plt.title("Top 10 Crops by Average Yield")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "top_crops_by_yield.png"),
    dpi=200
)
plt.close()


# Graph 7: Year-wise average yield
year_yield = (
    df.groupBy("Crop_Year")
    .agg(F.avg("Yield").alias("average_yield"))
    .orderBy("Crop_Year")
    .toPandas()
    .dropna()
)

plt.figure(figsize=(10, 5))
plt.plot(
    year_yield["Crop_Year"],
    year_yield["average_yield"],
    marker="o"
)
plt.xlabel("Crop Year")
plt.ylabel("Average Yield")
plt.title("Year-wise Average Crop Yield")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "yearwise_average_yield.png"),
    dpi=200
)
plt.close()


# Graph 8: Top 10 states by average yield
state_yield = (
    df.groupBy("State")
    .agg(F.avg("Yield").alias("average_yield"))
    .orderBy(F.desc("average_yield"))
    .limit(10)
    .toPandas()
    .sort_values("average_yield")
)

plt.figure(figsize=(10, 6))
plt.barh(
    state_yield["State"],
    state_yield["average_yield"]
)
plt.xlabel("Average Yield")
plt.ylabel("State")
plt.title("Top 10 States by Average Crop Yield")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "top_states_by_yield.png"),
    dpi=200
)
plt.close()


# Graph 9: Residual distribution
# Residual = actual yield - predicted yield
residuals = (
    best_predictions
    .select(
        (F.col("Yield") - F.col("prediction"))
        .alias("residual")
    )
    .orderBy(F.rand(seed=42))
    .limit(10000)
    .toPandas()
    .dropna()
)

plt.figure(figsize=(8, 5))
plt.hist(
    residuals["residual"],
    bins=40,
    edgecolor="black"
)
plt.axvline(0, linestyle="--")
plt.xlabel("Prediction Error (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title(f"Residual Distribution - {best_name}")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "residual_distribution.png"),
    dpi=200
)
plt.close()


print("\nAll additional graphs generated successfully.")
print("Check the output folder for PNG files.")


linear_predictions.unpersist()
tree_predictions.unpersist()
train_df.unpersist()
test_df.unpersist()
spark.stop()
