# Databricks notebook source
# import libraries
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# reading data with pandas
# data = pd.read_csv('employee.csv')
data = spark.sql('SELECT * FROM employee')
data = data.toPandas()

# COMMAND ----------

# checking head of data
data.head()

# COMMAND ----------

# check shape of data
data.shape

# COMMAND ----------

# when spark df converted into pandas. categorical null values replaced with None
data = data.fillna(np.nan)
data.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # 👉 Exploratory Data Analysis (EDA)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check Data Types
# MAGIC
# MAGIC Return the dtypes in the DataFrame.
# MAGIC
# MAGIC This returns a Series with the data type of each column. The result’s index is the original DataFrame’s columns. Columns with mixed types are stored with the object dtype. See the User Guide for more.

# COMMAND ----------

data.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check Info
# MAGIC Print a concise summary of a DataFrame.
# MAGIC
# MAGIC This method prints information about a DataFrame including the index dtype and columns, non-null values and memory usage.

# COMMAND ----------

data.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check Missing Values
# MAGIC Detect missing values for an array-like object.
# MAGIC
# MAGIC This function takes a scalar or array-like object and indicates whether values are missing (NaN in numeric arrays, None or NaN in object arrays, NaT in datetimelike).

# COMMAND ----------

# check missing values
data.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Summary Statistics
# MAGIC Generate descriptive statistics.
# MAGIC
# MAGIC Descriptive statistics include those that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.
# MAGIC
# MAGIC Analyzes both numeric and object series, as well as DataFrame column sets of mixed data types. The output will vary depending on what is provided. Refer to the notes below for more detail.

# COMMAND ----------

# summary statistics
data.describe().transpose()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pivot Table
# MAGIC Create a spreadsheet-style pivot table as a DataFrame.
# MAGIC
# MAGIC The levels in the pivot table will be stored in MultiIndex objects (hierarchical indexes) on the index and columns of the result DataFrame.

# COMMAND ----------

data.pivot_table(index='left')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Target Value Counts
# MAGIC Return a Series containing counts of unique values.
# MAGIC
# MAGIC The resulting object will be in descending order so that the first element is the most frequently-occurring element. Excludes NA values by default.

# COMMAND ----------

# check target balance
data['left'].value_counts(normalize=True)

# COMMAND ----------

# bar plot
data['left'].value_counts().plot.bar();

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analyze Target Variable on Features

# COMMAND ----------

features = ['number_project','time_spend_company','Work_accident','promotion_last_5years','department','salary']

fig=plt.subplots(figsize=(10,15))
for i, j in enumerate(features):
    plt.subplot(4, 2, i+1)
    plt.subplots_adjust(hspace = 1.0)
    sns.countplot(x=j,data = data, hue='left')
    plt.xticks(rotation=90)
    plt.title(j)

# COMMAND ----------

# MAGIC %md
# MAGIC # 👉 Data Preparation

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train Test Split

# COMMAND ----------

X = data.drop(['left'], axis=1)
y = data['left']

# COMMAND ----------

# check X variable
X.head()

# COMMAND ----------

# check y variable
y.head()

# COMMAND ----------

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# COMMAND ----------

X_train.shape, y_train.shape, X_test.shape, y_test.shape

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Processing Pipeline

# COMMAND ----------

# numeric features
numeric_features = X_train.select_dtypes(include='number').columns.tolist()
print(numeric_features)

# COMMAND ----------

# categorical features
categorical_features = X_train.select_dtypes(exclude='number').columns.tolist()
print(categorical_features)

# COMMAND ----------

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline

# COMMAND ----------

# build pipeline for numeric features
numeric_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scale', MinMaxScaler())])

# COMMAND ----------

# build pipeline for categorical features
categorical_pipeline = Pipeline(steps=[
    ('cat-impute', SimpleImputer(strategy='most_frequent')),
    ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False))])

# COMMAND ----------

# fit numeric pipeline
numeric_pipeline.fit_transform(X_train.select_dtypes(include='number'))

# COMMAND ----------

# fit categorical pipeline 
categorical_pipeline.fit_transform(X_train.select_dtypes(include='object'))

# COMMAND ----------

from sklearn.compose import ColumnTransformer

data_pipeline = ColumnTransformer(transformers=[
    ('numeric', numeric_pipeline, numeric_features),
    ('categorical', categorical_pipeline, categorical_features)
])

# COMMAND ----------

# fit entire data pipeline
data_pipeline.fit_transform(X_train)

# COMMAND ----------

# we can now use data_pipeline to transform X_train and X_test
X_train_transformed = data_pipeline.transform(X_train)
X_test_transformed = data_pipeline.transform(X_test)

# COMMAND ----------

X_train_transformed.shape, X_test_transformed.shape

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Training: Logistic Regression

# COMMAND ----------

from sklearn.linear_model import LogisticRegression

# COMMAND ----------

lr = LogisticRegression()

# COMMAND ----------

# fit the model
lr.fit(X_train_transformed, y_train)

# COMMAND ----------

# predict labels on X_test
y_pred = lr.predict(X_test_transformed)
y_pred

# COMMAND ----------

# predict probability on X_test
y_pred_proba = lr.predict_proba(X_test_transformed)
y_pred_proba

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Evaluation: Logistic Regression

# COMMAND ----------

# accuracy on test set
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

# COMMAND ----------

# AUC on test set
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred_proba[:,1])

# COMMAND ----------

# recall on test set
from sklearn.metrics import recall_score
recall_score(y_test, y_pred)

# COMMAND ----------

# confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

# COMMAND ----------

# classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# COMMAND ----------

# AUC plot
from sklearn.metrics import plot_roc_curve
plot_roc_curve(lr, X_test_transformed, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Training: Support Vector Machine

# COMMAND ----------

from sklearn.svm import SVC

# COMMAND ----------

svm = SVC(probability=True)

# COMMAND ----------

# fit the model
svm.fit(X_train_transformed, y_train)

# COMMAND ----------

# predict labels on X_test
y_pred = svm.predict(X_test_transformed)
y_pred

# COMMAND ----------

# predict probability on X_test
y_pred_proba = svm.predict_proba(X_test_transformed)
y_pred_proba

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Evaluation: Support Vector Machine

# COMMAND ----------

# accuracy on test set
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

# COMMAND ----------

# AUC on test set
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred_proba[:,1])

# COMMAND ----------

# recall on test set
from sklearn.metrics import recall_score
recall_score(y_test, y_pred)

# COMMAND ----------

# confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

# COMMAND ----------

# classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# COMMAND ----------

# AUC plot
from sklearn.metrics import plot_roc_curve
plot_roc_curve(svm, X_test_transformed, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC # We can also add Model in Pipeline (to make it more clean)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Logistic Regression Pipeline

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()

# COMMAND ----------

lr_pipeline = Pipeline(steps=[
                    ('preprocess', data_pipeline),
                    ('model', lr_model)])

# COMMAND ----------

lr_pipeline

# COMMAND ----------

from sklearn import set_config
set_config(display='diagram')

# COMMAND ----------

lr_pipeline

# COMMAND ----------

# fit pipeline with model
lr_pipeline.fit(X_train, y_train)

# COMMAND ----------

# predictions from pipeline
y_pred = lr_pipeline.predict(X_test)
y_pred

# COMMAND ----------

# accuracy on test set using lr_pipeline
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

# COMMAND ----------

# confusion matrix from lr_pipeline
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Support Vector Machine Pipeline

# COMMAND ----------

from sklearn.svm import SVC
svm_model = SVC(probability=True)

# COMMAND ----------

svm_pipeline = Pipeline(steps=[
                    ('preprocess', data_pipeline),
                    ('model', svm_model)])

# COMMAND ----------

# fit pipeline with model
svm_pipeline.fit(X_train, y_train)

# COMMAND ----------

# predictions from pipeline
y_pred = svm_pipeline.predict(X_test)
y_pred

# COMMAND ----------

# accuracy on test set using svm_pipeline
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

# COMMAND ----------

# confusion matrix from svm_pipeline
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Random Forest Pipeline

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()

# COMMAND ----------

rf_pipeline = Pipeline(steps=[
                    ('preprocess', data_pipeline),
                    ('model', rf_model)])

# COMMAND ----------

# fit pipeline with model
rf_pipeline.fit(X_train, y_train)

# COMMAND ----------

# predictions from pipeline
y_pred = rf_pipeline.predict(X_test)
y_pred

# COMMAND ----------

# accuracy on test set using rf_pipeline
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

# COMMAND ----------

# confusion matrix from rf_pipeline
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC ## K Nearest Neighbour Pipeline

# COMMAND ----------

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier()

# COMMAND ----------

knn_pipeline = Pipeline(steps=[
                    ('preprocess', data_pipeline),
                    ('model', knn_model)])

# COMMAND ----------

# fit pipeline with model
knn_pipeline.fit(X_train, y_train)

# COMMAND ----------

# predictions from pipeline
y_pred = knn_pipeline.predict(X_test)
y_pred

# COMMAND ----------

# accuracy on test set using knn_pipeline
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

# COMMAND ----------

# confusion matrix from knn_pipeline
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC # Save Sklearn Model / Pipeline

# COMMAND ----------

from joblib import dump
dump(knn_pipeline, 'knn_pipeline_sklearn.pkl') 

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Sklearn Model / Pipeline

# COMMAND ----------

from joblib import load
loaded_knn_pipeline = load('knn_pipeline_sklearn.pkl') 

# COMMAND ----------

loaded_knn_pipeline

# COMMAND ----------

loaded_knn_pipeline.predict_proba(X_test)

# COMMAND ----------

!ls