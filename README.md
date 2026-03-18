# datacamp_Data_Scientist_certificate_Project

This project was developed as part of the **DataCamp Data Analyst Associate Practical Exam**.  
The objective was to analyze recipe website traffic data and build a predictive model to help the business identify which recipes are more likely to generate **high traffic** when featured on the homepage.

## Business Problem

The product team wants to improve homepage performance by selecting recipes that are more likely to attract users and increase engagement across the website.  
Since the target outcome is whether a recipe generates **high traffic** or not, this project was treated as a **binary classification problem**.

## Project Goals

The project was designed to address the following tasks:

- Validate and clean the dataset
- Explore the data through visual analysis
- Build and compare classification models
- Evaluate model performance using relevant technical and business metrics
- Provide final recommendations for the business

## Dataset Overview

The dataset contains **947 recipes** and includes the following variables:

- `recipe`: unique recipe identifier
- `calories`
- `carbohydrate`
- `sugar`
- `protein`
- `category`
- `servings`
- `high_traffic`: target variable indicating whether the recipe generated high traffic

## Data Validation and Cleaning

Several validation and cleaning steps were performed before modelling:

- Checked the `recipe` column for missing values and duplicates
- Converted nutritional columns (`calories`, `carbohydrate`, `sugar`, `protein`) to numeric format
- Checked numeric columns for invalid negative values
- Standardized the `category` column and compared observed values to the documented categories
- Identified one unexpected category value: `Chicken Breast`
- Cleaned the `servings` column, which was stored as text instead of numeric format
- Converted the `high_traffic` column into a binary target variable
- Treated missing `high_traffic` values as non-high-traffic cases
- Preserved missing numeric values for later imputation in the modelling pipeline

## Exploratory Data Analysis

The exploratory analysis showed several important patterns:

- The distribution of calories was strongly right-skewed
- Recipe categories were relatively balanced, though some appeared more often than others
- Calories and servings alone did not clearly separate high-traffic and low-traffic recipes
- The proportion of high-traffic recipes varied considerably by category, suggesting that category was an important predictor

## Modelling Approach

This project used two classification models:

### 1. Logistic Regression
Used as the baseline model because it is:
- simple
- interpretable
- effective for binary classification tasks

### 2. Random Forest
Used as the comparison model because it can:
- capture more complex relationships
- model non-linear patterns
- serve as a stronger benchmark against a simpler model

## Preprocessing

A preprocessing pipeline was used to prepare the data before modelling:

- Numeric features were imputed with the median and scaled
- Categorical features were imputed with the most frequent value and one-hot encoded

This ensured that both models were trained using the same consistent preprocessing steps.

## Model Performance

### Logistic Regression
- Accuracy: **0.774**
- Precision: **0.846**
- Recall: **0.765**
- F1-score: **0.804**
- ROC AUC: **0.866**

### Random Forest
- Accuracy: **0.737**
- Precision: **0.822**
- Recall: **0.722**
- F1-score: **0.769**
- ROC AUC: **0.830**

## Business Metric

From a business perspective, the most relevant metric was **precision on recipes predicted to be high traffic**.

This answers the practical question:

> Of the recipes the model recommends for the homepage, how many are actually likely to perform well?

The **Logistic Regression** model achieved a precision of **0.846**, which means that approximately **84.6%** of recipes predicted to generate high traffic were actually high-traffic recipes.

Its recall of **0.765** shows that the model was also able to identify a substantial proportion of truly successful recipes, although some were still missed.

## Final Recommendation

The results showed that **Logistic Regression outperformed Random Forest across all reported metrics**.  
Because the business goal is to reduce the risk of promoting poor-performing recipes, Logistic Regression was selected as the recommended model.

### Recommendation for the business:
- Use **Logistic Regression** as a decision-support tool for homepage recipe selection
- Monitor **precision** as the main operational metric
- Do not fully automate homepage selection yet, since the model still misses some successful recipes
- Improve future model performance by collecting richer features such as:
  - recipe title wording
  - preparation time
  - seasonality
  - image quality
  - historical user engagement

## Files in This Repository

- `main.py` — main project code
- `recipe_site_traffic_2212.csv` — dataset used in the analysis
- `recipe_site_traffic_graphs_combined.png` — combined visualization of key graphs
- `recipe_site_traffic_full_outputs.png` — combined file with graphs and tables
- `README.md` — project overview and documentation

## Skills Demonstrated

This project helped demonstrate and strengthen skills in:

- Data Management
- Data Validation and Cleaning
- Exploratory Data Analysis
- Statistical Thinking
- Model Development
- Business-Focused Model Evaluation
- Python for Data Analysis (`pandas`, `matplotlib`, `scikit-learn`)
- Communication and Reporting

## Certification Context

This project was completed as part of a practical exam designed to assess applied skills relevant to a data role, including:

- data management
- exploratory analysis
- model development
- reporting and communication
- solving a real-world business case from start to finish

## Author

**Alan Pessoa**
