# Databricks notebook source
# MAGIC %md
# MAGIC # Loading Dataset

# COMMAND ----------

data = spark.sql('SELECT * from amazon_cells')
data = data.toPandas()
data.head()

# COMMAND ----------

data.shape

# COMMAND ----------

data['Label'].value_counts().plot.bar()

# COMMAND ----------

# MAGIC %md
# MAGIC # Method 1 - NLTK

# COMMAND ----------

import nltk

# COMMAND ----------

nltk.download('all')

# COMMAND ----------

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sent = SentimentIntensityAnalyzer()

# COMMAND ----------

sent.polarity_scores("I am loving it.")

# COMMAND ----------

sent.polarity_scores("The food was horrible.")

# COMMAND ----------

sent.polarity_scores("The food was horrible overall but Pizza was amazing.")

# COMMAND ----------

# define a function
def check_sentiment(x):
    return sent.polarity_scores(x)['compound']

# COMMAND ----------

data['compound_score'] = data['Text'].apply(lambda x: check_sentiment(x))

# COMMAND ----------

data.head()

# COMMAND ----------

data['compound_score'].hist(bins=50, figsize=(12,6))

# COMMAND ----------

# generate label function
def generate_label(x):
    return 1 if x>0 else 0

# COMMAND ----------

data['Predicted_Label'] = data['compound_score'].apply(lambda x: generate_label(x))

# COMMAND ----------

data.head()

# COMMAND ----------

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
accuracy_score(data['Label'],data['Predicted_Label'])

# COMMAND ----------

print(confusion_matrix(data['Label'],data['Predicted_Label']))

# COMMAND ----------

# MAGIC %md
# MAGIC # Method 2 - TextBlob

# COMMAND ----------

from textblob import TextBlob

# COMMAND ----------

tb = TextBlob("I am loving this.")

# COMMAND ----------

# check sentiment
tb.sentiment

# COMMAND ----------

# directly access polarity
tb.polarity

# COMMAND ----------

TextBlob("The food was horrible.").sentiment

# COMMAND ----------

TextBlob("The food is amazing").sentiment

# COMMAND ----------

def check_sentiment(x):
    return TextBlob(x).polarity

# COMMAND ----------

data['polarity_score'] = data['Text'].apply(lambda x: check_sentiment(x))

# COMMAND ----------

data.head()

# COMMAND ----------

data['polarity_score'].hist(bins=50, figsize=(12,6))

# COMMAND ----------

def generate_label(x):
    return 1 if x>0 else 0

# COMMAND ----------

data['Predicted_Label_Textblob'] = data['polarity_score'].apply(lambda x: generate_label(x))

# COMMAND ----------

data.head()

# COMMAND ----------

data.iloc[3]

# COMMAND ----------

data.iloc[3]['Text']

# COMMAND ----------

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
accuracy_score(data['Label'],data['Predicted_Label_Textblob'])

# COMMAND ----------

print(confusion_matrix(data['Label'],data['Predicted_Label_Textblob']))

# COMMAND ----------

# MAGIC %md
# MAGIC # Method 3 - Machine Learning
# MAGIC Team Exercise

# COMMAND ----------

