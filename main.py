# Recipe Site Traffic Report 
# Business objective

# The product team wants to identify which recipes are likely to generate high traffic when featured on the homepage. This is important because popular homepage recipes can increase traffic to the rest of the website and support subscription growth. The modelling task is therefore a binary classification problem, where the goal is to predict whether a recipe will generate high traffic or not.

# The business would ideally like high-traffic recipes to be predicted correctly 80% of the time. In practice, this means the model should be especially strong at identifying positive cases while avoiding recommending recipes that are unlikely to perform well.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("recipe_site_traffic_2212.csv")  
display(df.head())
df.info()
display(df.describe(include="all"))
display(df.isna().sum())

# Data validation and cleaning

# The dataset documentation states that each row represents a recipe and includes nutritional information, category, servings and whether traffic was high when the recipe was shown on the homepage. Before analysis, each column should be validated against the expected data type and meaning.
# Validation steps:Check for missing values, Check for duplicates, Confirm it is numeric and not used as a predictive feature.

print("Missing recipe values:", df["recipe"].isna().sum())
print("Duplicate recipe values:", df["recipe"].duplicated().sum())

# calories (Numeric)
# Validation steps: Check missing values, Confirm numeric type, Check for implausible values such as negatives or zeros if not reasonable.

df["calories"] = pd.to_numeric(df["calories"], errors="coerce")
display(df["calories"].describe())
print("Negative calories values:", (df["calories"] < 0).sum())

# carbohydrate (Numeric - grams)
# Validation steps: Convert to numeric if necessary, Check missing values and impossible negatives.

df["carbohydrate"] = pd.to_numeric(df["carbohydrate"], errors="coerce")
display(df["carbohydrate"].describe())
print("Negative carbohydrate values:", (df["carbohydrate"] < 0).sum())

# sugar (Numeric - grams)
# Validation steps: Convert to numeric, Check missing values, Check for negative values.

df["sugar"] = pd.to_numeric(df["sugar"], errors="coerce")
display(df["sugar"].describe())
print("Negative sugar values:", (df["sugar"] < 0).sum())

# protein (Numeric - grams)
# Validation steps: Convert to numeric, Check missing values, Check for negative values.

df["protein"] = pd.to_numeric(df["protein"], errors="coerce")
display(df["protein"].describe())
print("Negative protein values:", (df["protein"] < 0).sum())

# category (Character column: Lunch/Snacks, Beverages, Potato, Vegetable, Meat, Chicken, Pork, Dessert, Breakfast and One Dish Meal)
# Validation steps: Standardize text formatting, Check unique values, Identify unexpected labels or spelling inconsistencies.

df["category"] = df["category"].astype(str).str.strip()
display(df["category"].value_counts(dropna=False))

expected_categories = [
    "Lunch/Snacks", "Beverages", "Potato", "Vegetable", "Meat",
    "Chicken", "Pork", "Dessert", "Breakfast", "One Dish Meal"
]

actual_categories = sorted(df["category"].dropna().unique())
unexpected_categories = set(actual_categories) - set(expected_categories)
missing_expected_categories = set(expected_categories) - set(actual_categories)

print("Actual categories:", actual_categories)
print("Unexpected categories:", unexpected_categories)
print("Missing expected categories:", missing_expected_categories)

# servings (Numeric)
# Validation steps: Convert to numeric if possible, Check for missing values, Check for implausible values such as zero or negative servings.

display(df["servings"].value_counts(dropna=False))

df["servings"] = (
    df["servings"]
    .astype(str)
    .str.extract(r"(\d+)", expand=False)
)

df["servings"] = pd.to_numeric(df["servings"], errors="coerce")

display(df["servings"].value_counts(dropna=False))
display(df["servings"].describe())
print("Non-positive servings values:", (df["servings"] <= 0).sum())

# high_traffic (Target column)
# Validation steps: Check unique values, Convert into a binary target.

display(df["high_traffic"].value_counts(dropna=False))

df["high_traffic"] = df["high_traffic"].fillna("Low")
df["high_traffic_flag"] = np.where(df["high_traffic"] == "High", 1, 0)

display(df["high_traffic_flag"].value_counts())

# Cleaning decisions
# Keep recipe only as an identifier, not as a modelling feature.
# Convert all numeric columns to numeric using coercion, so invalid strings become missing values.
# Treat impossible negative nutritional values as invalid and set them to missing if found.
# Standardize category text.
# Convert high_traffic into a binary target variable.
# Impute missing numeric values with the median and categorical values with the most frequent category during modelling.

num_cols = ["calories", "carbohydrate", "sugar", "protein", "servings"]

for col in num_cols:
    df.loc[df[col] < 0, col] = np.nan

df_clean = df.copy()
display(df_clean.head())

# Summary:
# The dataset contained 947 recipes and was checked column by column against the specification. The recipe identifier had no missing values and no duplicates, so it was suitable for use as a unique ID only, not as a predictive feature. The numeric columns calories, carbohydrate, sugar, and protein each contained 52 missing values and were converted to numeric format for validation. The servings column required cleaning because it was stored as text rather than as a numeric field. The high_traffic column contained 373 missing values, and these were treated as non-high-traffic cases before converting the variable into a binary target. The category column was standardized for formatting and checked against the expected categories. Most categories matched the data dictionary, but one unexpected value, Chicken Breast, was found, showing that the dataset did not fully match the documented specification.

# Exploratory analysis

# Graph 1: Distribution of calories

plt.figure(figsize=(8,5))
df_clean["calories"].dropna().hist(bins=30)
plt.title("Distribution of Calories")
plt.xlabel("Calories")
plt.ylabel("Frequency")
plt.show()

# Summary
# The distribution of calories is strongly right-skewed. Most recipes have relatively low to moderate calorie values, while a smaller number of recipes have much higher calorie counts, extending above 3000 calories. This suggests the data contains some extreme observations and is not normally distributed.

# Graphic 2: Recipe category frequency

plt.figure(figsize=(10,5))
df_clean["category"].value_counts().plot(kind="bar")
plt.title("Number of Recipes by Category")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.show()

# Summary
# The number of recipes per category is fairly balanced overall, although some categories appear more often than others. Breakfast has the largest number of recipes, while One Dish Meal and Chicken appear less frequently. The presence of Chicken Breast as a separate category also indicates a mismatch between the observed data and the documented category list.

# Graphic 3: Relationship between servings and calories by traffic outcome

plot_df = df_clean.dropna(subset=["servings", "calories"])
plt.figure(figsize=(8,5))
for label, group in plot_df.groupby("high_traffic_flag"):
    plt.scatter(group["servings"], group["calories"], alpha=0.6, label=f"High Traffic = {label}")
plt.title("Calories vs Servings by Traffic Outcome")
plt.xlabel("Servings")
plt.ylabel("Calories")
plt.legend()
plt.show()

# Summary
# The scatter plot of calories versus servings shows substantial overlap between recipes with high traffic and those without high traffic. This suggests that neither calories nor servings alone is sufficient to clearly separate the two classes. As a result, a model that combines multiple features is more appropriate than relying on a single variable.

# Graphic 4: High traffic by Category

(
    df_clean.groupby("category")["high_traffic_flag"]
    .mean()
    .sort_values(ascending=False)
    .plot(kind="bar", figsize=(10,5))
)
plt.title("Proportion of High Traffic Recipes by Category")
plt.xlabel("Category")
plt.ylabel("Proportion High Traffic")
plt.xticks(rotation=45, ha="right")
plt.show()

# Summary
# The proportion of high-traffic recipes varies considerably by category, suggesting that category is likely to be an important predictor. Vegetable, Potato, and Pork recipes have the highest proportions of high-traffic outcomes, while Beverages, Breakfast, and Chicken have the lowest. This indicates that recipe type may strongly influence homepage performance.

# Model development
# This is a supervised binary classification problem because the target variable has two classes: high traffic and not high traffic. The goal is to predict whether a recipe belongs to the high-traffic class.

# Recommended: a baseline model to establish a minimum benchmark and a comparison model that can capture more structure in the data

# Feature setup
features = ["calories", "carbohydrate", "sugar", "protein", "category", "servings"]
target = "high_traffic_flag"

X = df_clean[features]
y = df_clean[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing
numeric_features = ["calories", "carbohydrate", "sugar", "protein", "servings"]
categorical_features = ["category"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Baseline model
baseline_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

baseline_model.fit(X_train, y_train)
baseline_preds = baseline_model.predict(X_test)
baseline_probs = baseline_model.predict_proba(X_test)[:, 1]

# Comparison model
comparison_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42
    ))
])

comparison_model.fit(X_train, y_train)
comparison_preds = comparison_model.predict(X_test)
comparison_probs = comparison_model.predict_proba(X_test)[:, 1]

# Model evaluation
# The model should be assessed using classification metrics. Since the business wants to correctly identify high-traffic recipes, precision is especially relevant if we want to minimize the chance of promoting unpopular recipes, while recall matters if we want to find as many good recipes as possible. A balanced metric such as F1-score is also useful.

def evaluate_model(name, y_true, preds, probs):
    print(f"--- {name} ---")
    print("Accuracy:", round(accuracy_score(y_true, preds), 3))
    print("Precision:", round(precision_score(y_true, preds), 3))
    print("Recall:", round(recall_score(y_true, preds), 3))
    print("F1:", round(f1_score(y_true, preds), 3))
    print("ROC AUC:", round(roc_auc_score(y_true, probs), 3))
    print("Confusion Matrix:\n", confusion_matrix(y_true, preds))
    print()

evaluate_model("Baseline Logistic Regression", y_test, baseline_preds, baseline_probs)
evaluate_model("Comparison Random Forest", y_test, comparison_preds, comparison_probs)

# Summary:
# Two models were compared: Logistic Regression as the baseline model and Random Forest as the comparison model. Logistic Regression performed better across all evaluation metrics. It achieved an accuracy of 0.774, precision of 0.846, recall of 0.765, F1-score of 0.804, and ROC AUC of 0.866. In contrast, Random Forest achieved an accuracy of 0.737, precision of 0.822, recall of 0.722, F1-score of 0.769, and ROC AUC of 0.830. Since the business objective focuses on identifying high-traffic recipes while minimizing poor homepage recommendations, Logistic Regression is the stronger model.

# Illustrative table
results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [
        accuracy_score(y_test, baseline_preds),
        accuracy_score(y_test, comparison_preds)
    ],
    "Precision": [
        precision_score(y_test, baseline_preds),
        precision_score(y_test, comparison_preds)
    ],
    "Recall": [
        recall_score(y_test, baseline_preds),
        recall_score(y_test, comparison_preds)
    ],
    "F1": [
        f1_score(y_test, baseline_preds),
        f1_score(y_test, comparison_preds)
    ],
    "ROC_AUC": [
        roc_auc_score(y_test, baseline_probs),
        roc_auc_score(y_test, comparison_probs)
    ]
})

display(results)

# Business metric
# The business does not only care about general model accuracy. It wants a practical way to decide whether the model is helping homepage selection. The best business-facing metric here is:

# Precision on recipes predicted as high traffic

# This answers:
# Of the recipes the model recommends for the homepage, how many actually turn out to be high traffic?...Also, this aligns well with the business goal of minimizing the chance of displaying unpopular recipes

# A second useful metric is: Recall on actual high-traffic recipes

# This answers:
# Of all recipes that truly would have been successful, how many did the model catch?

# Estimating business performance
business_summary = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Homepage recommendation precision": [
        precision_score(y_test, baseline_preds),
        precision_score(y_test, comparison_preds)
    ],
    "High-traffic capture rate (recall)": [
        recall_score(y_test, baseline_preds),
        recall_score(y_test, comparison_preds)
    ]
})

display(business_summary)

# Summary:
# For the business, the most useful metric is precision on recipes predicted to be high traffic. This measures how often the model’s homepage recommendations are actually successful. The Logistic Regression model achieved a precision of 0.846, meaning that about 84.6% of recipes predicted to be high traffic were in fact high-traffic recipes. Its recall was 0.765, meaning that it identified 76.5% of all recipes that truly generated high traffic. This indicates that the model performs well as a decision-support tool for homepage selection, although some successful recipes would still be missed.

# Final conclusion 

# This project investigated whether recipe characteristics could be used to predict high traffic on the homepage. After validating and cleaning the data, the analysis showed that the dataset included missing numeric values, a non-numeric servings field, and one unexpected category label, Chicken Breast, which did not fully match the documented specification. Exploratory analysis showed that calorie values were strongly right-skewed and that category appeared to be an important predictor of traffic performance.

# Two classification models were tested: Logistic Regression and Random Forest. Logistic Regression performed better on all reported metrics, achieving a precision of 0.846, recall of 0.765, F1-score of 0.804, and ROC AUC of 0.866. Because the business wants to reduce the risk of highlighting low-performing recipes on the homepage, precision is the most relevant operational metric. Based on this measure, Logistic Regression is the recommended model.

# # The model’s precision of 84.6% suggests that it meets the business goal if success is defined as the proportion of recommended recipes that actually generate high traffic. However, its recall of 76.5% shows that it still misses some successful recipes. For this reason, the model should be used as a decision-support tool rather than as a fully automated system, and further improvements could be made by collecting more predictive features.

# Recommendations

# The business should use the Logistic Regression model as the initial decision-support tool for homepage recipe selection. Performance should be monitored primarily through precision on predicted high-traffic recipes, since this best reflects the business risk of promoting unsuccessful recipes. Additional features such as preparation time, seasonal relevance, recipe title wording, image quality, and previous engagement metrics could improve future performance. The current model is strong enough to support homepage decisions, but it should not yet fully replace human judgment. The business should also continue reviewing whether the 80% target refers to precision or recall, since the model exceeds 80% precision but falls slightly below 80% recall.
