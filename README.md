# Bigdata_Final_Project title: 

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
- **Target variable**: `WAGE_RATE_OF_PAY_FROM`
- **Engineered features**:
  - `WAGE_DETRENDED`: Inflation-adjusted wages normalized to 2019  
  - `WAGE_BUCKET`: Binned version for classification  
  - `Job_Group`: Aggregated SOC titles  
  - `yrs_of_experience`: Derived from job duration  
  - `YEAR`, `ZIP`, and other geographic features  

---

## Repository Structure


---

## Approach and Setup

### Timeline / Deliverables
- **Milestone 3**: Feature engineering + hyperparameter tuning  
  - Due: April 14, 2025  
  - Tasks completed:
    - Created detrended wage and wage bucket features  
    - Tuned models using grid search with cross-validation  
    - Evaluated model performance across regression and classification tasks  
- **Next Milestone**: Final capstone deliverable  
  - Goals:
    - Deploy best-performing models  
    - Analyze and communicate model results  
    - Complete comprehensive final report  

### Resources
- PySpark MLlib documentation for scalable machine learning pipelines  
- ANOVA and IQR theory for outlier detection and variance analysis  
- GitHub issues and Jupyter notebooks for version control and experiment tracking  

### How to Contribute
- Fork the repository  
- Create a new feature or bugfix branch  
- Implement and test your changes  
- Submit a pull request with clear documentation  
- Follow code style guidelines and comment where necessary  


---------------- content of M2
### Feature Engineering
- Detrending wages by job group and year  
- IQR-based outlier removal  
- Categorical encoding for job and location  

### Modeling
- Linear regression and multinomial logistic regression  
- Tree-based models: Decision Tree, Random Forest, Gradient-Boosted Trees  

### Hyperparameter Tuning
- Grid search over parameters like `maxDepth`, `numTrees`, `regParam`, `elasticNetParam`  
- Cross-validation to reduce overfitting  

### Setup
Install dependencies and run via PySpark:

```bash
pip install -r requirements.txt


### Feature Engineering
- Detrending wages by job group and year
- IQR-based outlier removal
- Categorical encoding for job and location

### Modeling
- Linear regression and multinomial logistic regression
- Tree-based models: Decision Tree, Random Forest, Gradient-Boosted Trees

### Hyperparameter Tuning
- Grid search over parameters like `maxDepth`, `numTrees`, `regParam`, `elasticNetParam`
- Cross-validation to reduce overfitting

### Setup
Install dependencies and run via PySpark:

```bash
pip install -r requirements.txt

## **1. Feature Engineering**

To improve model readiness and predictive performance, we engineered several new features and applied key data transformations:

New Features Introduced:

- **`YEAR`**: Extracted from `RECEIVED_DATE` to capture the calendar year of each H-1B filing.
- **Outlier Removal**: Applied the **IQR method within each job category** after visualizing wage distribution and identifying extreme values.
- **`WAGE_DETRENDED`**: Normalized wages to 2019 levels using job group– and year–specific multipliers to account for inflation and market trends.
- **`WAGE_BUCKET`**: Created from `WAGE_DETRENDED` to enable classification modeling.

Previous features from last milestone:

- **`Job_Group`**: Categorized `SOC_TITLE` into broader job categories using keyword-based matching (e.g., “Data,” “Healthcare,” “Business & Accounting”).
- **`WAGE_RATE_OF_PAY_FROM`**:
  - Converted to float.
  - Filtered for annual salaries only (`WAGE_UNIT_OF_PAY == "Year"`).
  - Rows with salaries below $60,000 were removed to comply with H-1B minimums.
- **`yrs_of_experience`**: Derived from `END_DATE - BEGIN_DATE` to reflect intended employment duration.

---

## **2. Linear Regression Model Development**

### **Model Comparison Summary**

| Model Description                                | R² (Test) | RMSE (Test) | MAE (Test) |
|--------------------------------------------------|-----------|-------------|------------|
| Baseline (Raw Wage)                              | 0.211     | 30,256.78   | 23,812.02  |
| With `WAGE_DETRENDED`                            | 0.247     | 27,190.82   | 20,616.71  |
| Normalized Detrended Wage + State                | 0.244     | 5,505.04    | 4,347.52   |
| Normalized Detrended Wage + County               | 0.277     | 5,392.56    | 4,218.87   |
| Normalized Detrended Wage + City                 | 0.304     | 5,346.20    | 4,145.19   |
| Normalized Detrended Wage + ZIP Code             | **0.330** | **5,198.04**| **4,015.91**|
| ZIP Code + Hyperparameter Tuning                 | 0.324     | 5,223.33    | 4,111.65   |

### **Insights**
- Detrending removed inflation noise, improving model performance.
- ZIP code consistently outperformed other geographic granularity levels.
- Hyperparameter tuning helped mitigate overfitting, even if accuracy gains were marginal.

---

## **3. Tree-Based Regression and Classification Models**

### **Initial Regression (Original Wages)**

| Model                     | R² (Train/Test) | RMSE (Test) | MAE (Test) |
|--------------------------|-----------------|-------------|------------|
| Decision Tree            | 0.22 / 0.22      | 35,626.89   | 27,422.98  |
| Random Forest            | 0.22 / 0.22      | 35,564.59   | 27,295.29  |
| Gradient-Boosted Trees   | 0.26 / 0.26      | 34,507.07   | 26,235.76  |

### **Regression After Detrending**

| Model                     | RMSE (Test)     |
|--------------------------|-----------------|
| Decision Tree            | 31,667.25       |
| Random Forest            | 31,598.36       |
| Gradient-Boosted Trees   | **30,632.95**   |

- Gradient-Boosted Trees consistently outperformed others, benefiting the most from detrending.

---

### **Fine-Tuned Decision Tree**

#### **Regression (Tuned)**
| Metric      | Value          |
|-------------|----------------|
| RMSE (Train)| 30,794.87      |
| RMSE (Test) | 30,857.26      |
| Best Param  | `maxDepth = 10` |

- Tuning resulted in improved generalization and a ~5,000 drop in RMSE compared to the original untuned model.

#### **Classification (Tuned)**

| Dataset     | Accuracy       |
|-------------|----------------|
| Train       | 30.74%         |
| Test        | 30.59%         |
| Best Param  | `maxDepth = 10` |

- Accuracy nearly doubled from the original (16%) baseline.
- The tuned model captured better decision boundaries and showed **noticeable improvement** from the earlier version.

---

## **4. Multinomial Logistic Regression Development**

### **Baseline Performance (Original Wages)**

| Model                         | Accuracy (Test) |
|------------------------------|------------------|
| Multinomial Logistic Regression | **34.57%**       |
| Decision Tree                | 16%              |
| Random Forest                | 15%              |

### **Improved Classification Performance**

| Model Description                     | Accuracy (Test) |
|--------------------------------------|------------------|
| Cleaned Data (w/ IQR Outlier Removal) | 33.04%           |
| + ZIP Code                            | **37.59%**       |
| + Hyperparameter Tuning              | 36.31%           |

### **Tuning Parameters**
- `regParam`: [0.01, 0.1]
- `elasticNetParam`: [0.0, 1.0]
- `maxIter`: [50, 100]

### **Insights**
- Multinomial Logistic Regression significantly outperformed tree-based models.
- ZIP code emerged as the most predictive feature.
- While tuning didn't increase accuracy, it did improve model **generalization** and reduced **overfitting**.

---

## **5. Final Reflections**

- **Feature engineering**—especially wage detrending, categorical mapping, and geographic enrichment—was crucial to improving both regression and classification models.
- **ZIP code** was the most valuable location-based feature, clearly outperforming state, county, and city granularity.
- **Tree models**, particularly GBT and fine-tuned Decision Trees, demonstrated robust performance after detrending.
- **Hyperparameter tuning** enhanced stability and reduced overfitting, supporting model reliability for deployment.
