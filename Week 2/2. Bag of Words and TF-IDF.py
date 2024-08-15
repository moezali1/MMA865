# Databricks notebook source
data = spark.sql('SELECT * from linkedin')
data = data.toPandas()
data.head()

# COMMAND ----------

print(data['Text'][0])

# COMMAND ----------

print(data['Text'][1])

# COMMAND ----------

# MAGIC %md
# MAGIC # Text Preprocessing
# MAGIC
# MAGIC - Convert text into tokens
# MAGIC - Remove stop words
# MAGIC - Stem / Lemmatize
# MAGIC - Build bag of words
# MAGIC
# MAGIC The only thing different this time is, instead of working with one text we have 10 text (in NLP terminology: we have 10 documents).

# COMMAND ----------

text = list(data['Text'])

# COMMAND ----------

type(text)

# COMMAND ----------

len(text)

# COMMAND ----------

import nltk
nltk.download('all')

# COMMAND ----------



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

type(corpus)

# COMMAND ----------

len(corpus)

# COMMAND ----------

print(text[0])

# COMMAND ----------

print(corpus[0])

# COMMAND ----------

# MAGIC %md
# MAGIC # Bag of Words (CountVectorizer)
# MAGIC The bag-of-words model is a simplifying representation used in natural language processing and information retrieval (IR). In this model, a text (such as a sentence or a document) is represented as the bag (multiset) of its words, disregarding grammar and even word order but keeping multiplicity. The bag-of-words model has also been used for computer vision.
# MAGIC
# MAGIC The bag-of-words model is commonly used in methods of document classification where the (frequency of) occurrence of each word is used as a feature for training a classifier.
# MAGIC
# MAGIC An early reference to "bag of words" in a linguistic context can be found in Zellig Harris's 1954 article on Distributional Structure. (Wikipedia)

# COMMAND ----------

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(min_df=2)

# COMMAND ----------

# fit transform the CountVectorizer
X = cv.fit_transform(corpus)

# COMMAND ----------

# check the type of X
type(X)

# COMMAND ----------

# what is stored in X
print(X)

# COMMAND ----------

# convert X to array
X_array = X.toarray()

# COMMAND ----------

# check type of X_array
type(X_array)

# COMMAND ----------

# shape of X_array
X_array.shape

# COMMAND ----------

print(X_array)

# COMMAND ----------

# convert X_array into pandas dataframe
import pandas as pd
df = pd.DataFrame(X_array)
df

# COMMAND ----------

# assign column names to df
df.columns = cv.get_feature_names()
df.head(10)

# COMMAND ----------

# You can now concat these features in the original data
new_data = pd.concat([data,df], axis=1)
new_data

# COMMAND ----------

# MAGIC %md
# MAGIC # TF-IDF
# MAGIC In information retrieval, tf–idf, TF*IDF, or TFIDF, short for term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. It is often used as a weighting factor in searches of information retrieval, text mining, and user modeling. The tf–idf value increases proportionally to the number of times a word appears in the document and is offset by the number of documents in the corpus that contain the word, which helps to adjust for the fact that some words appear more frequently in general. tf–idf is one of the most popular term-weighting schemes today. A survey conducted in 2015 showed that 83% of text-based recommender systems in digital libraries use tf–idf.
# MAGIC
# MAGIC Variations of the tf–idf weighting scheme are often used by search engines as a central tool in scoring and ranking a document's relevance given a user query. tf–idf can be successfully used for stop-words filtering in various subject fields, including text summarization and classification. (Wikipedia)
# MAGIC

# COMMAND ----------

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()

# fit transform the TfidfVectorizer
X = tfidf.fit_transform(corpus)

# convert X to array
X_array = X.toarray()

df = pd.DataFrame(X_array)
df.columns = tfidf.get_feature_names()

new_data = pd.concat([data,df], axis=1)
new_data

# COMMAND ----------

