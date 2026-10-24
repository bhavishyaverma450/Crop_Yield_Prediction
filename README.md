# Crop Yield Prediction Using Apache Spark MLlib

A cluster computing project for predicting agricultural crop yield in India using **Apache Spark, PySpark, and Spark MLlib**. This project compares Linear Regression and Decision Tree Regression and evaluates their performance using RMSE, MAE, and R².

## Project Overview

Agriculture plays an important role in India's economy. Crop yield can vary with factors such as crop type, cultivated area, rainfall, fertilizer usage, pesticide usage, location, and year.

This project uses historical agricultural data to explore these relationships and build machine learning models that predict crop yield.

The implementation uses PySpark DataFrames and Spark MLlib pipelines for data preprocessing, model training, prediction, and evaluation.

## Objectives

- Load and process agricultural data using Apache Spark.
- Clean and prepare data for machine learning.
- Analyze relationships among numerical variables using a correlation heatmap.
- Train Linear Regression and Decision Tree Regression models.
- Evaluate models using RMSE, MAE, and R².
- Compare model performance and identify the better-performing model based on test RMSE.
- Generate visualizations and prediction outputs for result analysis.

## Dataset

The project is designed for the **Crop Yield in Indian States** dataset from Kaggle.

**Dataset link:**  
https://www.kaggle.com/datasets/akshatgupta7/crop-yield-in-indian-states-dataset

The expected dataset fields are:

| Column | Description |
|---|---|
| `Crop` | Crop type |
| `Crop_Year` | Year of cultivation |
| `Season` | Agricultural season |
| `State` | Indian state |
| `Area` | Cultivated area |
| `Production` | Crop production |
| `Annual_Rainfall` | Annual rainfall |
| `Fertilizer` | Fertilizer usage |
| `Pesticide` | Pesticide usage |
| `Yield` | Target variable to predict |

The actual dataset schema and units should be verified against the downloaded CSV.

### Dataset Setup

1. Download the dataset from Kaggle.
2. Extract the downloaded archive.
3. Place the CSV file in the project's `data` directory.
4. Rename the file to `crop_yield.csv` if necessary.

Expected location:

```text
data/crop_yield.csv
```

The dataset is not included in this repository by default. Download it from the original source before running the program.

## Technologies Used

- **Programming language:** Python
- **Cluster computing framework:** Apache Spark 3.5.9
- **Machine learning:** PySpark MLlib
- **Data processing:** Spark DataFrames
- **Data analysis:** Pandas
- **Visualization:** Matplotlib
- **Development environment:** VS Code and Windows Command Prompt
- **Execution mode:** Spark local mode (`local[*]`)

## Machine Learning Algorithms

### 1. Linear Regression

Linear Regression models the relationship between the input features and crop yield. It provides a baseline for evaluating the prediction task.

### 2. Decision Tree Regression

Decision Tree Regression learns nonlinear relationships by splitting observations according to feature values. It is compared with Linear Regression using the same training and testing datasets.

### Feature Preparation

The implementation uses:

- `StringIndexer` to convert categorical variables into numerical indices.
- `OneHotEncoder` to encode categorical features.
- `Imputer` to handle missing numerical feature values.
- `VectorAssembler` to combine features into a vector for Spark MLlib.

`Production` is excluded from the model input features to reduce target leakage when predicting yield.

## Methodology

1. Load the agricultural CSV file using Spark.
2. Inspect the dataset schema and sample records.
3. Convert relevant numerical columns into numeric data types.
4. Handle missing categorical values and filter invalid target and area records.
5. Calculate descriptive statistics and generate a correlation heatmap.
6. Prepare categorical and numerical features.
7. Split the data into training and testing subsets.
8. Train a Linear Regression model.
9. Train a Decision Tree Regression model.
10. Generate predictions on the testing data.
11. Calculate RMSE, MAE, and R² for both models.
12. Compare results and save predictions, metrics, and visualizations.

## Evaluation Metrics

### Root Mean Squared Error (RMSE)

RMSE measures the typical magnitude of prediction errors, with larger errors receiving greater weight. Lower values generally indicate better predictive accuracy.

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted yield. Lower values indicate smaller average errors.

### Coefficient of Determination (R²)

R² measures how well the model explains variation in the target variable. A value closer to 1 generally indicates a better fit, while negative values indicate performance worse than a constant-mean baseline on the evaluated data.

The metrics must be interpreted together rather than using a single metric alone.

## Visualizations

The project generates the following visualizations:

1. **Correlation Heatmap** — relationships among numerical dataset variables.
2. **Actual vs Predicted Yield** — compares observed and predicted yield values.
3. **RMSE Comparison** — compares the models' root mean squared errors.
4. **MAE Comparison** — compares the models' mean absolute errors.
5. **R² Comparison** — compares goodness of fit.
6. **Top Crops by Average Yield** — shows crops with the highest observed average yield.
7. **Year-wise Average Yield** — displays historical variation in average yield.
8. **Top States by Average Yield** — compares observed average yields across states.
9. **Residual Distribution** — visualizes actual-minus-predicted errors.

These plots are descriptive or evaluation visualizations; they do not by themselves establish that a particular agricultural factor causes changes in yield.

## Project Structure

```text
CropYieldPrediction/
├── data/
│   └── crop_yield.csv
├── output/
│   ├── correlation_heatmap.png
│   ├── actual_vs_predicted.png
│   ├── rmse_comparison.png
│   ├── mae_comparison.png
│   ├── r2_comparison.png
│   ├── top_crops_by_yield.png
│   ├── yearwise_average_yield.png
│   ├── top_states_by_yield.png
│   ├── residual_distribution.png
│   ├── model_metrics.csv
│   ├── result_summary.json
│   └── predictions/
├── screenshots/
├── test_spark.py
├── inspect_dataset.py
├── crop_yield_project.py
├── .gitignore
└── README.md
```

The generated files may differ depending on which execution steps have completed successfully.

## Installation and Setup

### Prerequisites

- Windows
- Python installed
- Java Development Kit compatible with Spark
- Apache Spark 3.5.9
- PySpark 3.5.9
- VS Code or another Python editor

### Install Python Libraries

Open Command Prompt and navigate to the project directory:

```cmd
cd /d D:\clustercA\CropYieldPrediction
```

Install the required packages:

```cmd
python -m pip install pyspark==3.5.9 pandas matplotlib
```

### Configure Python for Spark

In Windows Command Prompt, set the Python executable for the current terminal session:

```cmd
set PYSPARK_PYTHON=python
set PYSPARK_DRIVER_PYTHON=python
```

### Verify Spark

```cmd
spark-submit --version
```

### Run the Setup Test

```cmd
python test_spark.py
```

### Inspect the Dataset

```cmd
python inspect_dataset.py
```

### Run the Main Project

Make sure `data/crop_yield.csv` exists, then run:

```cmd
python crop_yield_project.py
```

The script trains the models, calculates evaluation metrics, saves predictions, and generates the heatmap and model comparison visualizations.

## Output Files

The `output` directory contains:

- `correlation_heatmap.png` — correlation visualization.
- `actual_vs_predicted.png` — prediction accuracy visualization.
- `rmse_comparison.png`, `mae_comparison.png`, and `r2_comparison.png` — model evaluation charts.
- `top_crops_by_yield.png` and `top_states_by_yield.png` — crop and state comparisons.
- `yearwise_average_yield.png` — historical yield trend.
- `residual_distribution.png` — distribution of prediction errors.
- `model_metrics.csv` — evaluation metrics for both models.
- `result_summary.json` — summary of model results and dataset split sizes.
- `predictions/` — Spark-generated CSV part files containing actual and predicted yield.

## Results and Analysis

The program reports the test RMSE, MAE, and R² for both regression models and selects the model with the lower test RMSE as the preferred model under that criterion.

**Insert the actual results from `output/model_metrics.csv` here after a successful execution.**

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Linear Regression | Run to obtain | Run to obtain | Run to obtain |
| Decision Tree Regression | Run to obtain | Run to obtain | Run to obtain |

No model performance values are prefilled because the results must come from running the program on the downloaded dataset.

## Limitations

- Results depend on the dataset's completeness, quality, coverage, and units.
- The random train/test split may place records from the same state, crop, or year in both subsets.
- A random split does not measure performance on future years or completely unseen regions.
- Average yield comparisons are descriptive and do not establish causal relationships.
- The program runs in Spark local mode; it demonstrates Spark-based processing but does not demonstrate execution across multiple cluster machines.
- Model performance may be improved through feature engineering, hyperparameter tuning, and more rigorous validation.

## Future Scope

- Evaluate models using time-based train/test splits.
- Apply cross-validation and hyperparameter tuning.
- Explore additional Spark MLlib algorithms where appropriate.
- Incorporate verified soil, weather, and irrigation data.
- Build a user interface for interactive crop-yield prediction.
- Deploy the model as a prediction service.

## Conclusion

This project demonstrates an end-to-end machine learning workflow using Apache Spark and PySpark MLlib for agricultural crop-yield prediction. It combines data preprocessing, feature engineering, regression model training, evaluation metrics, and visualization.

Comparing Linear Regression with Decision Tree Regression helps assess the predictive value of different modeling approaches. The resulting analysis provides a practical demonstration of Spark-based data processing and machine learning for agricultural data.

## Author

**Project:** INT315 – Cluster Computing CA-2  
**Topic:** Crop Yield Prediction Using Apache Spark MLlib  
**Language:** Python  
**Framework:** Apache Spark / PySpark
