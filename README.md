# Bigdata_Final_Project

## Project Overview

This project leverages multiple machine learning techniques—including linear regression, logistic regression, and tree-based models—to analyze wage patterns for H-1B visa-eligible positions in the U.S. It addresses two core predictive tasks:

- **Regression**: Predict the exact wage (`WAGE_RATE_OF_PAY_FROM`) for each LCA application  
- **Classification**: Categorize wages into defined salary buckets for classification analysis  

The H-1B dataset offers a comprehensive look into employment trends, salary standards, and job types across various industries. Building accurate models can:

- Provide insights of job categories and salary for international job seekers and employers  
- Aligns career planning with realistic salary expectations

---

## Problem Statement

The goal is to perform both **regression** (continuous salary prediction) and **classification** (wage bucket prediction), while overcoming several real-world data challenges:

- **Wage Inflation Over Time**: Wages shift across years due to inflation and market forces  
- **Outliers & Reporting Inconsistencies**: Wage anomalies and missing values distort model training  
- **High-Cardinality Features**: Thousands of unique job titles and worksite locations complicate generalization  

This dual modeling approach will yield insights that are valuable for stakeholders including job seekers, employers, and policy analysts. 

---


## Data Source

**Dataset**: [H-1B LCA Disclosure Data (2020–2024) – Kaggle](https://www.kaggle.com/datasets/zongaobian/h1b-lca-disclosure-data-2020-2024)  
**Overview**: This dataset contains detailed records of Labor Condition Applications (LCAs) for H-1B visa petitions filed between fiscal years 2020 and 2024. LCAs are essential to the H-1B process, serving as an employer’s attestation to the U.S. Department of Labor regarding wage standards and labor protections for U.S. workers.


### Target Variable
- `WAGE_RATE_OF_PAY_FROM`: The offered wage rate, used as the target variable for regression and classification tasks.


### Dataset Structure

#### 1. **Case Information**
- `CASE_NUMBER`, `CASE_STATUS`, `RECEIVED_DATE`, `DECISION_DATE`  
  *Indicates the status and timeline of each LCA filing.*

#### 2. **Employment Details**
- `JOB_TITLE`, `SOC_CODE`, `SOC_TITLE`, `FULL_TIME_POSITION`  
  *Describes the job role, classification, and full-time status.*

#### 3. **Employer Information**
- `EMPLOYER_NAME`, `EMPLOYER_ADDRESS`, `EMPLOYER_COUNTRY`  
  *Contains demographic and location-based employer data.*

#### 4. **Wage Information**
- `WAGE_RATE_OF_PAY_FROM`, `WAGE_RATE_OF_PAY_TO`, `PREVAILING_WAGE`  
  *Outlines the offered and prevailing wages for the position.*

#### 5. **Legal and Agent Details**
- `AGENT_REPRESENTING_EMPLOYER`, `AGENT_ATTORNEY_NAME`, `LAWFIRM_NAME_BUSINESS_NAME`  
  *Provides insights into legal representation during the petition.*

#### 6. **Worksite Information**
- `WORKSITE_ADDRESS`, `WORKSITE_CITY`, `WORKSITE_STATE`, `WORKSITE_POSTAL_CODE`  
  *Specifies the intended work location of the H-1B employee.*

#### 7. **Compliance and Conditions**
- `H_1B_DEPENDENT`, `WILLFUL_VIOLATOR`, `PUBLIC_DISCLOSURE`  
  *Indicates regulatory compliance and disclosure metrics.*


### Feature Engineering

- **`YEAR`**: Extracted from `RECEIVED_DATE` to capture the calendar year of filing.  
- **`yrs_of_experience`**: Calculated from `END_DATE - BEGIN_DATE` to reflect intended employment duration.  
- **`Job_Group`**: SOC titles were grouped into broader job families using keyword-based matching (e.g., *Data*, *Healthcare*, *Business & Accounting*).  
- **`Wage`**: Rows with salaries below $60,000 were removed to comply with H-1B legal wage requirements.  
- **`WAGE_DETRENDED`**: Wages were normalized to 2019 levels using job group and year-specific inflation multipliers.  
- **Outlier Removal**: Used the IQR method within each job group to remove extreme wage values after visualizing the distribution.  
- **`WAGE_BUCKET`**: Created categorical wage tiers from `WAGE_DETRENDED` for classification modeling.




---


## Repository Structure

```
├── data/                  # Raw and cleaned datasets
├── notebooks/             # Jupyter notebooks for each milestone
├── results/               # Evaluation metrics and plots
└── README.md              # Project documentation
```

---

## Approach and Setup

---

## Timeline / Deliverables

### Milestone 1: Project Plan & Data Understanding
- Outlined project goals, tasks, and evaluation metrics  
- Described datasets, join logic, and handling of missing values  
- Applied normalization strategies and explored distributions  
- Conducted initial exploratory data analysis (EDA), outlier detection, and feature type classification  
- Defined train/validation/test split strategy to avoid data leakage  
- **Deliverables**: Data dictionary, planning notebook, team credit assignment plan

---

### Milestone 2: EDA & Baseline Modeling
- Completed full-scale EDA and handled missing values  
- Developed baseline models including linear/logistic regression and tree-based methods  
- Created and evaluated derived features (e.g., job group, experience)  
- Justified transformations and visualized key variable relationships  
- Reported model evaluation metrics on training and test sets  
- **Deliverables**: Project notebook, 3-minute milestone presentation

---

### Milestone 3: Feature Engineering & Tuning
- Engineered key features: `WAGE_DETRENDED`, `WAGE_BUCKET`, and ZIP-based location attributes  
- Performed grid search and cross-validation for hyperparameter tuning  
- Compared performance across model families (linear, tree-based, logistic)  
- Evaluated generalization performance and addressed overfitting concerns  
- **Deliverables**: Updated modeling notebook, model performance tables

---

## Resources

- PySpark MLlib documentation for scalable pipelines  
- ANOVA and IQR theory for outlier detection and variance analysis  
- GitHub and Jupyter notebooks for version control and experimentation  

---

## How to Contribute

- Fork the repo  
- Create a feature or bugfix branch  
- Submit a pull request with description and comments  

---

## Sample outputs

### Modeling

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

### Feature Engineering Matters
- **Detrending and normalization of wages** helped reduce inflation noise and made salaries comparable across years.  
- **Normalizing detrended wages** led to a substantial improvement in regression performance.  
- **Categorical mapping** (e.g., `Job_Group`) and **location encoding** (ZIP, city, county, state) added meaningful signal to the models.

### Location Granularity
- **ZIP code** consistently outperformed broader geographic identifiers like city, county, or state in both regression and classification tasks.  
- This granularity helped models better capture regional wage variations.

### Model Performance & Tuning
- **Tree-based models** (especially Gradient-Boosted Trees) showed strong performance on detrended wage regression.  
- **Hyperparameter tuning** helped reduce overfitting and increased generalization across all model types.

### Key Takeaways
- **Linear Regression** using **normalized detrended wages + ZIP code** achieved the best performance in regression (R² = 0.330).  
- **Multinomial Logistic Regression** outperformed all tree-based classifiers, achieving the **highest classification accuracy of 37.59%**.  
- The combination of **normalized detrended wages** and **ZIP code granularity** was the most effective feature set across tasks.

---



