# Databricks notebook source
# MAGIC %md # Loading Dataset

# COMMAND ----------

data = spark.sql('SELECT * FROM spam')
data = data.toPandas()
data.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # Exploratory Data Analysis (EDA)

# COMMAND ----------

# check shape of data
data.shape

# COMMAND ----------

# check missing values
data.isnull().sum()

# COMMAND ----------

# check target balance
data['label'].value_counts(normalize = True)

# COMMAND ----------

# check target balance in a plot
data['label'].value_counts().plot.bar()

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

plt.xscale('log')
bins = 1.15**(np.arange(0,50))
plt.hist(data[data['label']=='ham']['length'],bins=bins,alpha=0.8)
plt.hist(data[data['label']=='spam']['length'],bins=bins,alpha=0.8)
plt.legend(('ham','spam'))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Modeling

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Test Split

# COMMAND ----------

# Create Feature and Label sets
X = data['message']
y = data['label']

# COMMAND ----------

X

# COMMAND ----------

y

# COMMAND ----------

# train test split (66% train - 33% test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)

print('Training Data :', X_train.shape)
print('Testing Data : ', X_test.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Count Vectorizer

# COMMAND ----------

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)
X_train_cv.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ## Logistic Regression

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train_cv, y_train)

# COMMAND ----------

from sklearn import metrics

# transform X_test using CV
X_test_cv = cv.transform(X_test)

# Create a prediction set:
predictions = lr.predict(X_test_cv)

# COMMAND ----------

# confusion matrix

import pandas as pd
df = pd.DataFrame(metrics.confusion_matrix(y_test,predictions), index=['ham','spam'], columns=['ham','spam'])
df

# COMMAND ----------

# classification report
print(metrics.classification_report(y_test,predictions))

# COMMAND ----------

# Check AUC
print(metrics.roc_auc_score(y_test,lr.predict_proba(X_test_cv)[:, 1]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Naive Bayes

# COMMAND ----------

# train a naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train_cv, y_train)

# COMMAND ----------

from sklearn import metrics

# Create a prediction set:
predictions_nb = nb.predict(cv.transform(X_test)) # transformed on the fly

# COMMAND ----------

# confusion matrix
df = pd.DataFrame(metrics.confusion_matrix(y_test,predictions_nb), index=['ham','spam'], columns=['ham','spam'])
df

# COMMAND ----------

# Check AUC
print(metrics.roc_auc_score(y_test,nb.predict_proba(X_test_cv)[:, 1]))