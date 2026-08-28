# 🏠 Real Estate Market Analyzer

> **An end-to-end data analytics, machine learning, and REST API project for analyzing and predicting the Egyptian real estate market.**

![Python](https://img.shields.io/badge/Python-3.14+-3776AB?style=for-the-badge\&logo=python\&logoColor=white)

![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge\&logo=pandas\&logoColor=white)

![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge\&logo=numpy\&logoColor=white)

![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge\&logo=scikit-learn\&logoColor=white)

![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?style=for-the-badge\&logo=fastapi\&logoColor=white)

![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge\&logo=jupyter\&logoColor=white)

---

## 📌 Project Overview

**Real Estate Market Analyzer** is an end-to-end data project designed to analyze the Egyptian real estate market using real-world property listing data.

The project was built as a complete data workflow rather than as a single machine learning model.

The pipeline covers:

```text
Raw Real Estate Data
        ↓
Data Loading & Understanding
        ↓
Data Cleaning
        ↓
Data Transformation
        ↓
Feature Engineering
        ↓
Exploratory Data Analysis (EDA)
        ↓
Statistical Analysis
        ↓
Data Visualization
        ↓
Market Insights
        ↓
REST API Integration
        ↓
Machine Learning
        ↓
Model Evaluation & Comparison
        ↓
Final Project Cleanup
        ↓
Professional Documentation
```

The project separates the **Sale** and **Rent** markets because they represent different pricing behaviors and therefore require different analytical and machine-learning approaches.

---

# 🎯 Project Objectives

The main objectives were to:

* Understand the Egyptian real estate listing market.
* Clean and validate a real-world property dataset.
* Analyze Sale and Rent listings independently.
* Identify important market patterns and relationships.
* Engineer meaningful real-estate features.
* Build a reusable data-processing layer.
* Produce validated market insights.
* Expose the processed market data through a REST API.
* Build machine-learning models for property price prediction.
* Compare multiple ML algorithms.
* Improve model quality through iterative feature engineering.
* Prevent data leakage during model development.
* Build a clean and maintainable project structure.
* Prepare the project for professional GitHub publication.

---

# 📊 Dataset

The project uses the **Real Estate Listings** dataset from Kaggle.

**Dataset:** `waddahali/real-estate-listings`

The main raw file used during development was:

```text
propertyfinder.csv
```

The dataset contains real-estate listings with information related to:

* Property type
* Transaction category
* City
* Town
* District
* Subdistrict
* Bedrooms
* Bathrooms
* Area
* Furnishing status
* Listing level
* New-construction status
* Developer/direct-listing information
* Price
* Price period
* Area unit
* Other listing attributes

## Dataset Availability

The original and processed CSV datasets are **not included in the GitHub repository** because of their file size.

The following files are kept locally and excluded from version control:

```text
data/raw/propertyfinder.csv
data/processed/cleaned_propertyfinder.csv
```

The dataset source is documented above so the analysis can be reproduced by obtaining the original dataset from Kaggle.

---

# 📈 Dataset Overview

After validation, the project contained:

| Metric         |  Value |
| -------------- | -----: |
| Total Listings | 39,712 |
| Sale Listings  | 19,802 |
| Rent Listings  | 19,910 |

The dataset validation confirmed the expected market split and the consistency of the major transaction-related fields.

### Important categorical values

```text
category
├── buy
└── rent
```

The dataset also contains:

```text
listing_type = property
```

For sale listings:

```text
price_period = sell
```

For rental listings:

```text
price_period = monthly
```

The area unit used in the dataset is:

```text
sqm
```

---

# 🧰 Technology Stack

## Programming Language

### Python

Python was the primary programming language used throughout the project.

It was used for:

* Data loading
* Data cleaning
* Data transformation
* Feature engineering
* Exploratory analysis
* Statistical analysis
* Visualization
* Machine learning
* API development
* Validation
* Testing utilities

The development environment used Python **3.14.6**.

---

# 📚 Python Libraries

## Pandas

Used as the primary data-analysis library.

Main responsibilities:

* Loading CSV data
* DataFrame manipulation
* Filtering
* Grouping
* Aggregation
* Missing-value analysis
* Data transformation
* Feature creation
* Statistical summaries
* Preparing data for ML

---

## NumPy

Used for numerical operations and mathematical transformations.

It was particularly useful for:

* Numerical feature processing
* Mathematical transformations
* Log transformation of the target
* Numerical calculations
* Handling numerical arrays

---

## Matplotlib

Used for data visualization and communicating analytical findings through charts.

Visualizations were used to investigate:

* Price distributions
* Property characteristics
* Market differences
* Relationships between numerical variables
* Sale vs. Rent behavior
* Geographic patterns
* Model-related analysis

---

## Scikit-learn

Used as the main machine-learning framework.

It supported:

* Train/test splitting
* Preprocessing
* Encoding categorical variables
* Scaling/transforming numerical features where appropriate
* Linear Regression
* Decision Tree
* Random Forest
* Model evaluation
* Feature importance analysis
* ML pipelines

Important evaluation metrics included:

```text
MAE
MSE
RMSE
R²
```

---

## FastAPI

FastAPI was used to build the project's REST API layer.

The API provides programmatic access to the analyzed real-estate data and market information.

The API was designed with:

* FastAPI application structure
* Request validation
* Query parameters
* HTTP error handling
* JSON responses
* Reusable data-loading logic

---

## Pydantic

Pydantic was used through the FastAPI stack for request/response validation and structured API data handling.

The project environment used:

```text
Pydantic 2.13.4
```

---

## OpenPyXL

OpenPyXL was installed and used as the Excel-processing dependency for Python-based spreadsheet handling where required.

Version used:

```text
openpyxl 3.1.5
```

---

# 🖥️ Development Tools

The project workflow used a professional Python development environment consisting of:

### Visual Studio Code

Primary code editor used for:

* Python development
* Project structure management
* Source-code editing
* Terminal operations
* Git workflow
* API development

### Jupyter Notebook

Used for the analytical and machine-learning workflow.

The main notebook was:

```text
notebooks/real_estate_analysis.ipynb
```

The notebook contains the analytical workflow and experimentation performed during the project.

### Postman

Used during API development and testing to send requests to the FastAPI application and inspect responses.

### Git

Used for source-control management.

### GitHub / GitHub Desktop

Used to prepare the project for professional version control and GitHub publication.

---

# 🗂️ Project Structure

The project was organized into a modular structure instead of keeping all logic in a single script.

```text
REAL_ESTATE_ANALYZER/

│
├── data/
│   ├── raw/
│   │   └── propertyfinder.csv
│   │
│   └── processed/
│
├── models/
│   ├── final_rent_model.pkl
│   └── final_sale_model.pkl
│
├── notebooks/
│   └── real_estate_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── data_processing.py
│   ├── analysis.py
│   └── visualization.py
│
├── .gitignore
├── LICENSE
├── requirements.txt
├── README.md
└── ...
```

> **Note:** Large datasets and trained model artifacts shown above are local project files and are intentionally excluded from the GitHub repository.

### `data/raw/`

Contains the original/raw dataset.

The raw data is kept separate from processed data to preserve the original source and make the pipeline reproducible.

The raw dataset is intentionally excluded from GitHub because of its size.

### `data/processed/`

Used for processed datasets and intermediate data products.

These files are also excluded from GitHub because of their size.

### `models/`

Contains trained machine-learning model artifacts generated during development.

The final trained models are kept locally and excluded from GitHub because the serialized model files are too large for a standard GitHub repository.

### `notebooks/`

Contains the Jupyter analytical workflow.

### `src/`

Contains reusable application code.

---

# 🧩 Source Code Architecture

## `src/data_processing.py`

Responsible for reusable data-loading and processing functionality.

The objective was to avoid duplicating data-loading logic across different parts of the project.

The API also imports the reusable data-loading functionality from this module.

---

## `src/analysis.py`

Contains reusable analysis-related functionality used to keep analytical logic separate from the API layer.

This separation improves:

* Maintainability
* Reusability
* Readability
* Testing
* Future extension

---

## `src/visualization.py`

Contains reusable visualization functionality used to separate chart generation and visualization logic from the main analytical workflow.

This improves the organization and reusability of the project's visualization layer.

---

## `src/api.py`

Contains the FastAPI application.

The application is initialized with:

```text
Real Estate Market Analyzer API
```

and exposes the real-estate data and market analysis through HTTP endpoints.

---

# 🔬 Phase 1 — Data Loading & Understanding

The first stage focused on understanding the raw dataset before making any transformations.

The analysis examined:

* Dataset shape
* Column names
* Data types
* Missing values
* Unique categorical values
* Numerical distributions
* Transaction categories
* Price-related fields
* Area-related fields
* Geographic hierarchy

The initial dataset contained:

```text
39,712 rows
78 columns
```

This stage established the foundation for all subsequent analysis.

---

# 🧹 Phase 2 — Data Cleaning

The dataset contained real-world imperfections, including missing values and inconsistent information.

The cleaning stage focused on:

* Detecting missing values
* Inspecting invalid values
* Understanding categorical inconsistencies
* Handling numerical fields
* Validating price information
* Validating area information
* Checking transaction categories
* Preserving useful observations
* Removing or handling problematic records when necessary

Special attention was given to avoiding aggressive deletion of data merely because a field was missing.

Instead, each feature was evaluated based on its analytical and ML importance.

---

# 🔄 Phase 3 — Data Transformation

The data was transformed into a form suitable for statistical analysis and machine learning.

Major transformation tasks included:

* Converting numerical fields to appropriate numeric types
* Preparing categorical variables
* Separating transaction categories
* Preparing price variables
* Preparing area variables
* Handling missing categorical values
* Creating analysis-ready datasets
* Preparing ML-compatible features

---

# 🧠 Phase 4 — Feature Engineering

Feature engineering was one of the most important stages of the project.

Instead of relying only on the original dataset columns, additional information was derived from existing variables.

The goal was to represent the real-estate market more meaningfully for machine learning.

---

# 🗺️ Location Hierarchy Features — V13

A dedicated feature-engineering stage focused on the geographic hierarchy.

The project considered:

```text
city
   ↓
town
   ↓
district
   ↓
subdistrict
```

This allowed the ML models to distinguish between properties that may have similar physical characteristics but belong to very different markets.

The V13 stage specifically improved the representation of location-related information.

---

# 💰 Economic Feature Engineering — V14

V14 introduced additional economic/pricing-oriented feature engineering.

The purpose was to make the relationship between property characteristics and price more meaningful.

This stage was part of the iterative process used to improve the rental model and evaluate whether engineered economic variables could improve predictive performance.

---

# 🔍 Phase 5 — Exploratory Data Analysis

EDA was performed to understand the market before modeling.

The analysis investigated:

### Property characteristics

* Property type
* Bedrooms
* Bathrooms
* Area
* Furnishing status
* New construction
* Listing level

### Geographic characteristics

* City
* Town
* District
* Subdistrict

### Market characteristics

* Sale vs. Rent
* Price distribution
* Monthly rental behavior
* Property composition
* Geographic price differences

EDA was not treated as a cosmetic visualization step.

It was used to answer business questions and guide feature engineering and modeling decisions.

---

# 📊 Phase 6 — Statistical Analysis

Statistical analysis was used to quantify relationships within the dataset.

The project examined:

* Descriptive statistics
* Distribution characteristics
* Numerical relationships
* Price behavior
* Area/price relationships
* Differences between market segments
* Potential outliers
* Feature relevance

The statistical stage helped determine which variables were likely to carry predictive information.

---

# 📉 Phase 7 — Data Visualization

Visualization was used to transform numerical analysis into understandable market information.

The visualization workflow supported analysis of:

* Price distributions
* Property-size distributions
* Bedroom/bathroom distributions
* Geographic differences
* Furnishing patterns
* Sale vs. Rent differences
* Feature relationships

The objective was not simply to generate charts, but to use visual evidence to support market conclusions.

---

# 💡 Phase 8 — Market Insights

After cleaning, EDA, statistical analysis, and visualization, the project produced validated market insights.

The analysis focused on questions such as:

* What types of properties dominate the market?
* How does property size relate to price?
* How do locations influence prices?
* How do Sale and Rent markets differ?
* Which property characteristics appear most important?
* How does furnishing affect rental pricing?
* How do property categories differ economically?

The final insights were validated against the underlying dataset before being treated as market findings.

---

# 🔀 Sale vs. Rent Separation

One of the major architectural decisions was to treat:

```text
Sale Market
```

and

```text
Rental Market
```

as separate analytical and ML problems.

This decision was important because sale prices and rental prices have fundamentally different distributions and economic behavior.

---

# 🏷️ Sale Machine Learning Model

The first major ML workflow focused on predicting:

```text
price_egp
```

for Sale properties.

An initial feature set included:

```text
property_type
city
district
bedrooms
bathrooms
area_value
furnished
listing_level
is_new_construction
is_direct_from_dev
```

The initial Sale dataset contained:

```text
19,802 listings
```

---

# 🧪 Sale Train/Test Split

The Sale dataset was divided into:

```text
Training: 15,841
Testing:   3,961
```

The split represented an approximately:

```text
80% Training
20% Testing
```

strategy.

---

# ⚙️ Sale Preprocessing

Categorical and numerical variables were processed using a machine-learning preprocessing pipeline.

After preprocessing:

```text
Training processed shape: (15,841, 466)
Testing processed shape:  (3,961, 466)
```

The large number of resulting features came primarily from categorical encoding.

---

# 🤖 Sale Model 1 — Linear Regression

The first baseline model was **Linear Regression**.

Results:

| Metric |            Result |
| ------ | ----------------: |
| MAE    |  8,611,701.36 EGP |
| RMSE   | 28,963,659.56 EGP |
| R²     |            0.3724 |

This established a baseline for comparison.

---

# 🌳 Sale Model 2 — Decision Tree

A Decision Tree Regressor was then evaluated.

Results:

| Metric |            Result |
| ------ | ----------------: |
| MAE    |  6,600,841.99 EGP |
| RMSE   | 26,368,407.95 EGP |
| R²     |            0.4799 |

The Decision Tree significantly improved upon the Linear Regression baseline.

---

# 🌲 Sale Model 3 — Random Forest

Random Forest was evaluated as a stronger nonlinear ensemble model.

Results:

| Metric |            Result |
| ------ | ----------------: |
| MAE    |  6,036,579.11 EGP |
| RMSE   | 25,896,814.59 EGP |
| R²     |            0.4983 |

### Best Sale model at this stage

**Random Forest**

It achieved:

* Lowest MAE
* Lowest RMSE
* Highest R²

among the three compared models.

---

# 🔐 Machine Learning V2 — Leakage-Safe Modeling

A second major ML version was developed to make the modeling workflow more robust.

The complete dataset contained:

```text
39,712 listings
78 original columns
```

The ML feature matrix contained:

```text
39,712 rows
12 features
```

The features included:

```text
property_type
city
town
district
subdistrict
bedrooms
bathrooms
area_value
furnished
listing_level
is_new_construction
is_direct_from_dev
```

---

# 🎯 Target Transformation

Because real-estate prices are highly skewed, a logarithmic target transformation was introduced:

```python
log_price = log(1 + price_egp)
```

This was designed to reduce the influence of extreme prices and provide a more stable target distribution for modeling.

---

# ✂️ V2 Train/Test Split

The V2 dataset was divided into:

```text
X_train_v2: (31,769, 12)

X_test_v2:  (7,943, 12)
```

representing an approximately:

```text
80% / 20%
```

train/test split.

---

# ⚙️ V2 Preprocessing

The preprocessing stage generated:

```text
Processed Training:

(31,769, 1,647)

Processed Testing:

(7,943, 1,647)
```

The numerical features included:

```text
bedrooms
bathrooms
area_value
```

Categorical variables were encoded into ML-compatible numerical representations.

---

# 🌲 V2 Random Forest

The V2 Random Forest model achieved:

| Metric |            Result |
| ------ | ----------------: |
| MAE    |  5,345,537.04 EGP |
| RMSE   | 17,017,562.73 EGP |
| R²     |            0.2199 |

The model also provided feature-importance information.

Important features included:

| Feature                 | Importance |
| ----------------------- | ---------: |
| area_value              |   0.214890 |
| property_type_Apartment |   0.167538 |
| furnished_NO            |   0.069798 |

This analysis demonstrated the importance of property size, property type, and furnishing status in the model's predictive behavior.

---

# 🏢 Rental Price Model

After completing the Sale modeling workflow, the project moved to the **Rent Model**.

The rental dataset contained:

```text
19,910 listings
```

The rental workflow was intentionally analyzed separately from Sale.

---

# 🔎 Rent Data Validation

The Rent dataset was independently validated before modeling.

Important missing-value counts included:

| Feature     | Missing Values |
| ----------- | -------------: |
| furnished   |          8,384 |
| subdistrict |          6,758 |
| bedrooms    |          1,343 |
| district    |            346 |
| bathrooms   |            297 |

These values were considered during preprocessing and feature engineering.

---

# 🧠 Rental Model Iterations

The rental model was developed iteratively rather than as a single experiment.

The workflow progressed through multiple feature-engineering versions.

Important stages included:

```text
V12

↓

V13 — Location Hierarchy Features

↓

V14 — Economic Feature Engineering
```

Each version was evaluated against previous versions to determine whether additional engineered information improved the model.

---

# 📈 V12 Rental Benchmark

The V12 results were retained as an important benchmark for comparing subsequent rental-model versions.

The recorded Random Forest benchmark included approximately:

```text
MAE  = 55,787.67 EGP
RMSE = 662,114.38 EGP
```

This benchmark was used to evaluate the effect of subsequent location and economic feature engineering.

---

# 🗺️ V13 — Location-Aware Rental Modeling

V13 focused on strengthening the geographic representation of rental properties.

The model incorporated the hierarchy:

```text
City
  ↓
Town
  ↓
District
  ↓
Subdistrict
```

The objective was to allow the model to capture localized rental-market behavior rather than treating geographic variables as isolated categorical columns.

---

# 💵 V14 — Economic Feature Engineering

V14 extended the rental feature space with economic/pricing-oriented engineered features.

The purpose was to test whether derived economic relationships could provide additional predictive signal beyond the original listing attributes.

This represented the latest stage of the rental-model development workflow.

---

# 🧪 Model Evaluation Strategy

The project used multiple regression metrics.

## MAE — Mean Absolute Error

Measures the average absolute prediction error.

```text
MAE = average(|actual - predicted|)
```

It is especially useful because it is expressed in the same unit as the target.

For this project:

```text
EGP
```

---

## MSE — Mean Squared Error

Penalizes larger errors more heavily because the errors are squared.

```text
MSE = average((actual - predicted)²)
```

---

## RMSE — Root Mean Squared Error

The square root of MSE.

```text
RMSE = sqrt(MSE)
```

RMSE remains in the target's unit and gives more weight to large prediction errors.

---

## R² — Coefficient of Determination

Measures how much of the target variance is explained by the model.

```text
R² = 1 - SS_res / SS_tot
```

Higher values generally indicate better explanatory performance.

---

# 🧠 Why Multiple Models Were Compared

The project did not assume that one algorithm would automatically be the best.

Instead, different model families were compared:

```text
Linear Regression
        ↓
Decision Tree
        ↓
Random Forest
```

This provided a baseline-to-advanced progression.

### Linear Regression

Useful as a simple baseline.

### Decision Tree

Captures nonlinear relationships and feature interactions.

### Random Forest

Combines multiple decision trees to improve generalization and capture complex nonlinear relationships.

---

# 🔍 Feature Importance

Tree-based models were also used to investigate which variables contributed most strongly to predictions.

For example, the V2 Random Forest identified:

```text
area_value
property_type_Apartment
furnished_NO
```

as highly important features.

This provides interpretability in addition to predictive performance.

---

# 🌐 REST API

The project also includes a FastAPI-based REST API.

The API layer was designed to expose the real-estate dataset and market-analysis functionality through HTTP endpoints.

Application metadata:

```text
Title:

Real Estate Market Analyzer API

Description:

API for analyzing the Egyptian real estate market

Version:

1.0.0
```

---

# 🔌 API Capabilities

The API architecture includes functionality for accessing:

* Property listings
* Market overview information
* Filtering/querying property data
* Structured JSON responses
* Error handling
* Request validation

The API uses reusable processing functionality rather than duplicating the data-loading implementation.

---

# 🧪 API Testing

The API was tested during development using HTTP requests and API-testing workflows.

The testing process verified:

* API startup
* Endpoint availability
* HTTP responses
* JSON response structure
* Query behavior
* Error handling

Postman was used as part of the API testing workflow.

---

# 💾 Data & Model Files

Large generated artifacts are intentionally excluded from version control.

The local project may contain:

```text
data/raw/propertyfinder.csv

data/processed/cleaned_propertyfinder.csv

models/final_rent_model.pkl

models/final_sale_model.pkl
```

These files are excluded through `.gitignore` to keep the GitHub repository lightweight and avoid committing large binary/model artifacts.

The repository therefore focuses on:

* Source code
* Analytical workflow
* Documentation
* Configuration
* Reproducible project structure

while large datasets and trained model artifacts remain local.

### Why are these files excluded?

The project contains large CSV datasets and serialized machine-learning models.

Their approximate sizes are:

```text
final_rent_model.pkl       ≈ 123 MB
final_sale_model.pkl       ≈ 307 MB
cleaned_propertyfinder.csv ≈ 76 MB
propertyfinder.csv         ≈ 69 MB
```

These files are intentionally excluded from the standard Git repository because of their size.

The trained models can be regenerated using the documented machine-learning workflow when the required dataset and dependencies are available.

---

# 🧹 Final Project Cleanup

After completing the analytical, API, and ML workflows, the project entered a final cleanup stage.

The cleanup focused on:

* Removing unnecessary files
* Removing temporary artifacts
* Removing backup notebooks
* Removing Python cache files
* Separating raw and processed data
* Keeping reusable source code inside `src/`
* Organizing notebooks
* Reviewing dependencies
* Reviewing project structure
* Removing unnecessary generated files
* Excluding large datasets and trained model artifacts from Git
* Protecting environment configuration files
* Creating a professional `LICENSE`
* Preparing the repository for GitHub
* Creating professional documentation

The final project structure was intentionally kept clean and understandable for another developer reviewing the repository for the first time.

---

# 📦 Requirements

The project dependencies are documented in:

```text
requirements.txt
```

The environment included the core packages required for:

```text
Data Analysis

Numerical Computing

Machine Learning

API Development

Data Validation

Spreadsheet/Data File Handling
```

Key package versions used during development included:

```text
Python       3.14.6

pandas       3.0.5

numpy        2.5.2

fastapi      0.141.1

pydantic     2.13.4

openpyxl     3.1.5

pip          26.2.1
```

---

# 🚀 How to Run the Project

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>

cd REAL_ESTATE_ANALYZER
```

---

## 2. Obtain the Dataset

Download the **Real Estate Listings** dataset from Kaggle:

```text
waddahali/real-estate-listings
```

Place the original CSV file in:

```text
data/raw/propertyfinder.csv
```

The dataset is intentionally not included in the repository because of its size.

---

## 3. Create a virtual environment

Windows:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

If PowerShell execution policy prevents activation, the environment can also be activated through another supported terminal configuration.

---

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Run the analytical notebook

Open:

```text
notebooks/real_estate_analysis.ipynb
```

and execute the cells sequentially.

The notebook contains the main analytical and ML workflow.

> The trained model files used during development are not included in the repository because of their size. They can be regenerated by running the corresponding machine-learning workflow in the notebook.

---

# ▶️ Running the FastAPI Application

From the project root:

```bash
uvicorn src.api:app --reload
```

The API can then be accessed locally through the development server.

FastAPI also provides interactive API documentation through its automatic documentation interface.

---

# 🔄 End-to-End Workflow

The complete project lifecycle can be summarized as:

```text
1. Collect Dataset
       ↓
2. Load Raw Data
       ↓
3. Understand Structure
       ↓
4. Validate Data
       ↓
5. Clean Data
       ↓
6. Transform Data
       ↓
7. Engineer Features
       ↓
8. Perform EDA
       ↓
9. Statistical Analysis
       ↓
10. Visualization
       ↓
11. Generate Market Insights
       ↓
12. Build Reusable Processing Code
       ↓
13. Build FastAPI
       ↓
14. Test API
       ↓
15. Split Sale / Rent
       ↓
16. Build ML Baselines
       ↓
17. Evaluate Models
       ↓
18. Improve Features
       ↓
19. Prevent Leakage
       ↓
20. Compare Versions
       ↓
21. Final Cleanup
       ↓
22. Documentation
       ↓
23. GitHub Publication
```

---

# 🧱 Engineering Principles Used

The project followed several important software and data-science principles.

### Separation of Concerns

Data processing, analysis, visualization, and API functionality were separated into different modules.

### Reproducibility

The raw dataset and processing workflow were kept logically separated from processed data.

### Validation Before Modeling

Data quality was examined before machine learning.

### Leakage Awareness

The ML workflow was improved to ensure preprocessing and feature engineering did not improperly expose test-set information to the training process.

### Iterative Modeling

Models were improved through controlled versions rather than arbitrary changes.

### Model Comparison

Multiple algorithms were evaluated using consistent metrics.

### Explainability

Feature importance was inspected to understand model behavior.

### Documentation

The final repository was prepared with technical documentation explaining both the project and its development workflow.

### Version-Control Hygiene

Large datasets, trained model artifacts, Python cache files, environment files, and other unnecessary generated files were excluded from version control where appropriate.

---

# 📌 Key Results

## Sale Models

| Model             |        MAE (EGP) |        RMSE (EGP) |         R² |
| ----------------- | ---------------: | ----------------: | ---------: |
| Linear Regression |     8,611,701.36 |     28,963,659.56 |     0.3724 |
| Decision Tree     |     6,600,841.99 |     26,368,407.95 |     0.4799 |
| Random Forest     | **6,036,579.11** | **25,896,814.59** | **0.4983** |

**Random Forest** was the best-performing model in the initial Sale comparison.

---

# Leakage-Safe V2 Random Forest

| Metric |            Result |
| ------ | ----------------: |
| MAE    |  5,345,537.04 EGP |
| RMSE   | 17,017,562.73 EGP |
| R²     |            0.2199 |

Top recorded feature importances included:

```text
area_value                 0.214890

property_type_Apartment    0.167538

furnished_NO               0.069798
```

---

# Rental Benchmark

V12 Random Forest benchmark:

```text
MAE  ≈ 55,787.67 EGP

RMSE ≈ 662,114.38 EGP
```

Later versions focused on:

```text
V13 → Location Hierarchy Features

V14 → Economic Feature Engineering
```

---

# 🏆 What This Project Demonstrates

This project demonstrates practical experience across the complete data-science lifecycle.

### Data Engineering

* Raw data ingestion
* Data validation
* Cleaning
* Transformation
* Feature preparation

### Data Analysis

* Exploratory Data Analysis
* Descriptive statistics
* Relationship analysis
* Market segmentation
* Geographic analysis

### Data Visualization

* Distribution analysis
* Market comparisons
* Feature relationships
* Business-oriented visual communication

### Machine Learning

* Feature/target selection
* Train/test splitting
* Preprocessing
* Categorical encoding
* Regression modeling
* Model comparison
* Evaluation metrics
* Feature importance
* Log-target transformation
* Leakage-safe modeling
* Iterative feature engineering

### Backend/API Development

* FastAPI
* Pydantic validation
* REST endpoints
* Query parameters
* JSON responses
* HTTP error handling
* API testing

### Software Engineering

* Modular source code
* Project organization
* Dependency management
* Testing
* Cleanup
* Documentation
* Version control
* Repository hygiene

---

# 📚 Learning Journey

The project was intentionally developed as an end-to-end practical learning experience.

Instead of studying each technology independently and stopping at theoretical examples, the technologies were integrated into one realistic business problem.

The progression was:

```text
Python
  ↓
Pandas / NumPy
  ↓
Data Cleaning
  ↓
EDA
  ↓
Statistics
  ↓
Visualization
  ↓
Feature Engineering
  ↓
FastAPI
  ↓
Machine Learning
  ↓
Model Evaluation
  ↓
Feature Optimization
  ↓
Software Refactoring
  ↓
Testing
  ↓
Professional Documentation
```

---

# 🔮 Future Improvements

Possible future extensions include:

* Advanced hyperparameter tuning
* Cross-validation
* Gradient Boosting models
* XGBoost / LightGBM experimentation
* More sophisticated geographic features
* Price-per-square-meter modeling
* Separate models by property type
* Advanced outlier treatment
* Model persistence improvements
* Production inference endpoints
* API authentication
* Automated testing pipelines
* Dockerized deployment
* Cloud deployment
* Interactive dashboard
* Model monitoring
* Automated retraining pipeline

These are considered future extensions rather than claims about functionality already implemented in the current version.

---

# 👩‍💻 Author

**Nouran Talaat**

Data Analytics / Python / Machine Learning Project

---

# ⭐ Project Highlights

```text
✓ Real-world Egyptian real estate dataset

✓ 39,712 validated property listings

✓ Separate Sale and Rent analysis

✓ Complete data-cleaning workflow

✓ Feature engineering

✓ Location hierarchy engineering

✓ Economic feature engineering

✓ Exploratory Data Analysis

✓ Statistical analysis

✓ Data visualization

✓ Market insights

✓ FastAPI REST API

✓ API testing

✓ Machine learning

✓ Linear Regression

✓ Decision Tree

✓ Random Forest

✓ Log-target transformation

✓ Leakage-safe ML workflow

✓ Model evaluation

✓ Feature importance

✓ Iterative model development

✓ Modular Python architecture

✓ Final project cleanup

✓ Large-file handling

✓ Professional GitHub documentation
```

---

# 📜 Project Status

**Current status: Final Project / GitHub Publication Preparation**

The project has completed the major stages of:

```text
Data Analysis

        +

API Integration

        +

Machine Learning

        +

Model Evaluation

        +

Final Cleanup

        +

Documentation

        +

GitHub Preparation
```

Large datasets and trained model artifacts remain excluded from version control to maintain a clean and lightweight repository.

The repository is being prepared as a professional portfolio project demonstrating an end-to-end approach to real-world data analysis, machine learning, and API development.

---

# ❤️ Final Note

This project was built to demonstrate that a data project is more than training a model.

A complete real-world solution requires:

```text
Good Data

   +

Good Analysis

   +

Good Features

   +

Good Models

   +

Good Validation

   +

Good Software Structure

   +

Good Documentation
```

**Real Estate Market Analyzer** combines these components into one complete workflow for understanding and modeling the Egyptian real estate market.
