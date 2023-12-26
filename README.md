# Heart Disease Prediction using Logistic Regression and Decision Trees

**Purpose:**

The code aims to predict the occurrence of heart disease based on a dataset using two different classification algorithms: Logistic Regression and Decision Trees. It utilizes data preprocessing, model training, evaluation metrics, and result visualization to analyze and predict heart disease presence.

**Libraries Used:**

Pandas (import pandas as pd): For data manipulation and handling the dataset.

Matplotlib (import matplotlib.pyplot as plt): Used for visualization purposes.

NumPy (import numpy as np): Employed for numerical operations.

**Data Loading and Preparation:**

Loads the heart disease dataset (heart.csv) using Pandas.

Splits the dataset into training and test sets using train_test_split from sklearn.model_selection.

**Modeling:**

Initializes two classification models: Logistic Regression (LogisticRegression()) and Decision Trees (DecisionTreeClassifier()).

**Trains both models using the training data.**

**Evaluation:**

Predicts outcomes using the test data and calculates accuracy scores (accuracy_score from sklearn.metrics).

Generates confusion matrices (confusion_matrix from sklearn.metrics) for both models to evaluate performance.

Computes precision, recall, and F1 scores (precision_score, recall_score, f1_score) for each model.

**Result Visualization:**

Displays the confusion matrices in tabular format using Pandas DataFrames.

Outputs a sample of the actual labels alongside predictions made by both models.

**Conclusion:**

Provides evaluation metrics (accuracy, precision, recall, F1 score) for both Logistic Regression and Decision Trees models to assess their performance in predicting heart disease presence.
