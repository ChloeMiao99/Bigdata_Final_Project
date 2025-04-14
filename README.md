# Bigdata_Final_Project

## Project Overview

The H-1B visa program plays a vital role in enabling highly skilled international professionals to work in the United States, particularly in fields such as technology, healthcare, engineering, and finance. U.S. employers file Labor Condition Applications (LCAs) as part of the H-1B process to attest that they will pay foreign workers fair wages comparable to those paid to U.S. employees. Given the program’s competitive nature and the significant number of applications filed each year, predicting wage has both practical and policy implications.

This project applies multiple machine learning techniques—including linear regression, logistic regression, and tree-based models—to predict wages for H-1B-eligible positions. Two main tasks are addressed: (1) **regression**, to predict exact wage values for each application, and (2) **classification**, to group wages into defined salary buckets for comparative analysis. By leveraging a comprehensive dataset of LCA filings from 2020 to 2024, the project aims to uncover patterns in salary levels across job categories, geographic regions, and years of experience.

Ultimately, these models can help inform data-driven decision-making for various stakeholders. Accurate wage predictions support regulatory compliance, allow international job seekers to benchmark job offers more effectively and align career planning with realistic salary expectations.

---

## Problem Statement

The central challenge of this project is to build predictive models for H-1B wages using both regression and classification techniques, while accounting for the complexities inherent in real-world labor data. One key difficulty is **wage inflation over time**, which can distort comparisons across years unless adjusted for economic factors. Additionally, the dataset contains **outliers and inconsistencies** in reported wages—some due to manual entry errors, others due to differences in compensation structures across industries.

Another major hurdle is the presence of **high-cardinality features**, such as thousands of unique job titles and geographic identifiers. These can lead to sparse representations and overfitting if not handled properly. Addressing these issues requires thoughtful feature engineering, normalization strategies, and robust model selection.


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

#### Feature Engineering
- Selected core features: `Job_Group`, `yrs_of_experience`, and geographic variables (`STATE`, `COUNTY`, `CITY`, `ZIP`)  
- Engineered:
  - `WAGE_DETRENDED`: Inflation-adjusted to 2019  
  - `WAGE_BUCKET`: Wage bins for classification  
- Found **ZIP code** to be the strongest location feature  
- Encoded categoricals with **StringIndexer** and **OneHotEncoder**

---

#### Data Splitting & Setup
- Split into **80% training / 20% test** sets  
- Ensured no data leakage during preprocessing  
- Used **PySpark MLlib** for pipeline integration

---

### Model Development

#### Linear Regression
- Predicted exact wages
  
#### Multinomial Logistic Regression
- Classified wages into buckets  

#### Tree-Based Models
- Applied Decision Tree, Random Forest, GBT  

---

### Optimization & Evaluation
- Tuned models via **grid search + cross-validation**  
- Evaluated with **R²**, **RMSE**, **MAE** for regression; **Accuracy** for classification  
- Compared performance across models and feature sets

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

| Model         | Baseline (Train/Test RMSE) | Detrended Wage (Train/Test) | Hyperparameter Tuning (Train/Test) |
|--------------------------|-----------------|-------------|------------|
| Decision Tree            | 30,860.31 / 30,806.11      | 28,571.5 / 28604.1   | 27,422.98 / 27,806.1708  |
| Random Forest            | 30,739.24 / 30,680.9      | 28,434.47 / 28,468.69   | 27,677.8554 / 27,717.0163  |
| Gradient-Boosted Trees   | 29,803.27 / 29,763.33     | 27,576.2 / 27,611.71   | 26,960.2027 / 27,041.1945  |

#### Tuning Parameters
##### Decision Tree Regressor 
- `maxDepth`: [5, 10]  

##### Random Forest Regressor
- `maxDepth`: [5, 10]  
- `numTrees`: [20, 50]  

##### Gradient-Boosted Tree Regressor 
- `maxDepth`: [5, 10]  
- `maxIter`: [20, 50]  

---

- Accuracy nearly doubled from the original (16%) baseline.
- The tuned model captured better decision boundaries and improved generalization.

---

### Multinomial Logistic Regression Results

#### Baseline Performance (Original Wages)

| Model                           | Accuracy (Test) |
|--------------------------------|------------------|
| Multinomial Logistic Regression | **34.57%**       |
| Decision Tree                   | 16%              |
| Random Forest                   | 15%              |

#### Improved Classification Performance

| Model Description                     | Accuracy (Test) |
|--------------------------------------|------------------|
| Cleaned Data (w/ IQR Outlier Removal) | 33.04%           |
| + ZIP Code                            | **37.59%**       |
| + Hyperparameter Tuning               | 36.31%           |

#### Tuning Parameters

- `regParam`: [0.01, 0.1]  
- `elasticNetParam`: [0.0, 1.0]  
- `maxIter`: [50, 100]

### Tree-Based Classification Results

| Model         | Baseline (Train/Test Accuracy) | Detrended Wage (Train/Test) | Hyperparameter Tuning (Train/Test) |
|--------------------------|-----------------|-------------|------------|
| Decision Tree            | 0.16 / 0.16      | 0.29 / 0.29   | 0.3119 / 0.311  |
| Random Forest            | 0.15 / 0.15      | 0.28 / 0.28   | 0.3023 / 0.302  |

#### Tuning Parameters
##### Decision Tree Regressor 
- `maxDepth`: [5, 10]  

##### Random Forest Regressor
- `maxDepth`: [5, 10]  
- `numTrees`: [20, 50]  

##### Gradient-Boosted Tree Regressor 
- `maxDepth`: [5, 10]  
- `maxIter`: [20, 50]  

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



