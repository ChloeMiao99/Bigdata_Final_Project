# Bigdata_Final_Project

## Project Overview

This project leverages linear regression, logistic regression, and tree-based models to analyze H-1B visa-eligible job positions in the U.S. It includes two primary machine learning tasks:

- **Regression**: Predict the exact wage value for each application  
- **Classification**: Predict the salary range by categorizing wages into defined buckets  

The H-1B dataset provides detailed information on employment and wage patterns across a broad spectrum of industries and job titles. Accurately modeling these patterns can:

- Support policy-making and ensure regulatory compliance  
- Serve as benchmarks for evaluating job offers  
- Detect anomalies or inconsistencies in reported wage data  

---

## Problem Statement

The task involves both **regression** (predicting numeric wages) and **classification** (categorizing wages into buckets). Challenges include:

- Wage inflation over time  
- Outliers and reporting inconsistencies  
- High-cardinality job titles and geographic identifiers  

---


## Data Source

- **Dataset**: [H-1B LCA Disclosure Data 2020–2024 (Kaggle)](https://www.kaggle.com/datasets/zongaobian/h1b-lca-disclosure-data-2020-2024)  
- **Target Variable**: `WAGE_RATE_OF_PAY_FROM`  
- **Engineered Features**:
  - `WAGE_DETRENDED`: Inflation-adjusted wages normalized to 2019  
  - `WAGE_BUCKET`: Binned version for classification tasks  
  - `Job_Group`: Aggregated SOC titles based on keyword clustering  
  - `yrs_of_experience`: Derived from `END_DATE - BEGIN_DATE`
  - `YEAR` (derived from `RECEIVED_DATE`)
  - Geographic features:
    - `WORKSITE_STATE`
    - `WORKSITE_COUNTY`
    - `WORKSITE_CITY`
    - `WORKSITE_POSTAL_CODE` (ZIP)


---


## Repository Structure

```
├── data/                  # Raw and cleaned datasets
├── notebooks/             # Jupyter notebooks for each milestone
├── models/                # Saved model files and tuning results
├── results/               # Evaluation metrics and plots
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## Approach and Setup

### Timeline / Deliverables

- **Milestone 3**: Feature engineering + hyperparameter tuning  
  - Tasks completed:
    - Created detrended wage and wage bucket features  
    - Tuned models using grid search with cross-validation  
    - Evaluated model performance for both regression and classification  

- **Final Deliverable**:
  - Deploy best-performing models  
  - Communicate results and business insights  
  - Submit comprehensive final report  

### Resources

- PySpark MLlib documentation for scalable pipelines  
- ANOVA and IQR theory for outlier detection and variance analysis  
- GitHub and Jupyter notebooks for version control and experimentation  

### How to Contribute

- Fork the repo  
- Create a feature or bugfix branch  
- Submit a pull request with description and comments  
- Follow project style guidelines  

---

## Setup

Install dependencies and run via PySpark:

```bash
pip install -r requirements.txt
```

---

## Feature Engineering

- **WAGE_DETRENDED**: Normalized wages by job group and year  
- **Outlier Removal**: IQR-based filtering within job groups  
- **Categorical Encoding**: Encoded job titles and locations  
- **yrs_of_experience**: Derived from contract dates  
- **Filtering**: Removed hourly/monthly wages, kept only annual above $60,000  

---

## Modeling

### Linear Regression Results

| Model Description                                | R² (Test) | RMSE (Test) | MAE (Test) |
|--------------------------------------------------|-----------|-------------|------------|
| Baseline (Raw Wage)                              | 0.211     | 30,256.78   | 23,812.02  |
| With `WAGE_DETRENDED`                            | 0.247     | 27,190.82   | 20,616.71  |
| Normalized Detrended Wage + State                | 0.244     | 5,505.04    | 4,347.52   |
| Normalized Detrended Wage + County               | 0.277     | 5,392.56    | 4,218.87   |
| Normalized Detrended Wage + City                 | 0.304     | 5,346.20    | 4,145.19   |
| Normalized Detrended Wage + ZIP Code             | **0.330** | **5,198.04**| **4,015.91**|
| ZIP Code + Hyperparameter Tuning                 | 0.324     | 5,223.33    | 4,111.65   |

### Tree-Based Regression Results

#### Initial Regression (Original Wages)

| Model                     | R² (Train/Test) | RMSE (Test) | MAE (Test) |
|--------------------------|-----------------|-------------|------------|
| Decision Tree            | 0.22 / 0.22      | 35,626.89   | 27,422.98  |
| Random Forest            | 0.22 / 0.22      | 35,564.59   | 27,295.29  |
| Gradient-Boosted Trees   | 0.26 / 0.26      | 34,507.07   | 26,235.76  |

#### Regression After Detrending

| Model                     | RMSE (Test)     |
|--------------------------|-----------------|
| Decision Tree            | 31,667.25       |
| Random Forest            | 31,598.36       |
| Gradient-Boosted Trees   | **30,632.95**   |

---

### Fine-Tuned Decision Tree

#### Regression (Tuned)

| Metric       | Value      |
|--------------|------------|
| RMSE (Train) | 30,794.87  |
| RMSE (Test)  | 30,857.26  |
| Best Param   | maxDepth=10 |

#### Classification (Tuned)

| Dataset | Accuracy |
|---------|----------|
| Train   | 30.74%   |
| Test    | 30.59%   |

- Accuracy nearly doubled from the original (16%) baseline.
- The tuned model captured better decision boundaries and improved generalization.

---

## Multinomial Logistic Regression Results

### Baseline Performance (Original Wages)

| Model                           | Accuracy (Test) |
|--------------------------------|------------------|
| Multinomial Logistic Regression | **34.57%**       |
| Decision Tree                   | 16%              |
| Random Forest                   | 15%              |

### Improved Classification Performance

| Model Description                     | Accuracy (Test) |
|--------------------------------------|------------------|
| Cleaned Data (w/ IQR Outlier Removal) | 33.04%           |
| + ZIP Code                            | **37.59%**       |
| + Hyperparameter Tuning               | 36.31%           |

### Tuning Parameters

- `regParam`: [0.01, 0.1]  
- `elasticNetParam`: [0.0, 1.0]  
- `maxIter`: [50, 100]

---


## Final Reflections

### 📌 Feature Engineering Matters
- **Detrending and normalization of wages** helped reduce inflation noise and made salaries comparable across years.  
- **Normalizing detrended wages** led to a substantial improvement in regression performance.  
- **Categorical mapping** (e.g., `Job_Group`) and **location encoding** (ZIP, city, county, state) added meaningful signal to the models.

### 📍 Location Granularity
- **ZIP code** consistently outperformed broader geographic identifiers like city, county, or state in both regression and classification tasks.  
- This granularity helped models better capture regional wage variations.

### 🔧 Model Performance & Tuning
- **Tree-based models** (especially Gradient-Boosted Trees) showed strong performance on detrended wage regression.  
- **Hyperparameter tuning** helped reduce overfitting and increased generalization across all model types.

### 🎯 Key Takeaways
- ✅ **Linear Regression** using **normalized detrended wages + ZIP code** achieved the best performance in regression (R² = 0.330).  
- ✅ **Multinomial Logistic Regression** outperformed all tree-based classifiers, achieving the **highest classification accuracy of 37.59%**.  
- ✅ The combination of **normalized detrended wages** and **ZIP code granularity** was the most effective feature set across tasks.

---



