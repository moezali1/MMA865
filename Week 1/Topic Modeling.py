# Databricks notebook source
# MAGIC %md
# MAGIC # Loading Dataset

# COMMAND ----------

data = spark.sql('SELECT * FROM amazon_apps')
data = data.toPandas()
data.head()

# COMMAND ----------

# drop label column as we don't need it for topic modeling
data.drop('Positive', axis=1, inplace=True)
data.head()

# COMMAND ----------

data.isnull().sum()

# COMMAND ----------

data.sort_values(by='reviewText', ascending=True).head()

# COMMAND ----------

data.sort_values(by='reviewText', ascending=False).head()

# COMMAND ----------

# MAGIC %md
# MAGIC # Text Preprocessing

# COMMAND ----------

import nltk
nltk.download('all')

# COMMAND ----------

# create a list of text data from dataframe
text = list(data['reviewText'])
type(text), len(text)

# COMMAND ----------

import re
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

corpus = []

for i in range(len(text)):
    r = re.sub('[^a-zA-Z]', ' ', text[i])
    r = r.lower()
    r = r.split()
    r = [word for word in r if word not in stopwords.words('english')]
    r = [lemmatizer.lemmatize(word) for word in r]
    r = ' '.join(r)
    corpus.append(r)

# COMMAND ----------

for i in corpus[:10]:
    print(i)
    print('--------')

# COMMAND ----------

# MAGIC %md
# MAGIC # Vectorization

# COMMAND ----------

# Vectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words = 'english', min_df = 10)
doc = cv.fit_transform(corpus)

# COMMAND ----------

# see the type of doc
doc

# COMMAND ----------

# if you want to see it as a dataframe 
import pandas as pd
df = pd.DataFrame(doc.toarray(), columns = cv.get_feature_names())
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # Latent Dirichlet Allocation

# COMMAND ----------

# Topic model with 10 topics
from sklearn.decomposition import LatentDirichletAllocation
LDA = LatentDirichletAllocation(n_components=10, random_state=123)

# COMMAND ----------

# This can long time
LDA.fit(doc)

# COMMAND ----------

LDA.components_.shape

# COMMAND ----------

pd.DataFrame(LDA.components_, columns = cv.get_feature_names())

# COMMAND ----------

# see the most common words by topic
for index,topic in enumerate(LDA.components_):
    print(f'THE TOP 30 WORDS FOR TOPIC #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[:30]])
    print('\n')

# COMMAND ----------

# applying the model to dataset
topic_results = LDA.transform(doc)
data_topics = pd.DataFrame(topic_results)
data_topics.head()

# COMMAND ----------

data_topics.shape

# COMMAND ----------

data['Topic'] = topic_results.argmax(axis=1)
data.head()

# COMMAND ----------

data['Topic'].value_counts().plot.bar()