# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Before We Start...
# MAGIC
# MAGIC Basic concepts of Spark: 
# MAGIC - RDD (Resilient Distributed Datasets): fundamental data structure for distributing data among cluster nodes. Immutable.
# MAGIC - Transformation: operations on RDD that returns an RDD, such as map, filter, reduce, and reduceByKey.
# MAGIC - Action: operations on RDD that returns a non-RDD value, such as collect.
# MAGIC
# MAGIC We will be mainly using Spark Dataframe APIs instead of RDD APIs, to simplify development.
# MAGIC - Spark Dataframes are very similar to tables in relational databases. They have schema. Most of the operations on them are similar to querying a relational database as well. You can consider Spark Dataframe as a wrap on top of RDD.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading Data

# COMMAND ----------

# Reading data from Delta Lake

amazon_review_raw = spark.sql("SELECT * FROM default.reviews_train").sample(0.25)

# COMMAND ----------

display(amazon_review_raw)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Cleaning Data

# COMMAND ----------

# Drop duplicates

print("Before duplication removal: ", amazon_review_raw.count())
amazon_review_distinct = amazon_review_raw.dropDuplicates(['reviewerID', 'asin'])
print("After duplication removal: ", amazon_review_distinct.count())

# COMMAND ----------

# Convert Unix timestamp to readable date

from pyspark.sql.functions import from_unixtime, to_date
from pyspark.sql.types import *

amazon_review_with_date = amazon_review_distinct.withColumn("reviewTime", to_date(from_unixtime(amazon_review_distinct.unixReviewTime))) \
                                                .drop("unixReviewTime")

# COMMAND ----------

display(amazon_review_with_date)

# COMMAND ----------

# MAGIC %md
# MAGIC As comparison, for pandas dataframe you will use .apply() to apply a function to a column. See: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html
# MAGIC
# MAGIC For example: amz_review['Date'] = amz_review['Time'].apply(to_date)

# COMMAND ----------

# Tokenization

from pyspark.ml.feature import RegexTokenizer

regexTokenizer = RegexTokenizer(inputCol="reviewText", outputCol="reviewWord", pattern="\\W")

amazon_review_tokenized = regexTokenizer.transform(amazon_review_with_date.fillna("", subset=["reviewText"]))

# COMMAND ----------

# Remove stop words

from pyspark.ml.feature import StopWordsRemover

remover = StopWordsRemover(inputCol="reviewWord", outputCol="reviewWordFiltered")
amazon_review_stop_word_removed = remover.transform(amazon_review_tokenized)

# COMMAND ----------

# Stemming

from nltk.stem.porter import PorterStemmer
from pyspark.sql.functions import udf

def stemming(col):
    p_stemmer = PorterStemmer()
    return [p_stemmer.stem(w) for w in col]

stemming_udf = udf(stemming, ArrayType(StringType()))
amazon_review_stemmed = amazon_review_stop_word_removed.withColumn("reviewWordCleaned", stemming_udf(amazon_review_stop_word_removed.reviewWordFiltered))

# COMMAND ----------

# Dropping temporary columns, and cache results (note that cache is also a lazy operation)

amazon_review_cleaned = amazon_review_stemmed.drop("reviewWord").drop("reviewWordFiltered").cache()

display(amazon_review_cleaned)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratory Analysis

# COMMAND ----------

# Let's use Spark SQL for some simple exploratory analysis. Firstly, we need to create a temporary view based on the dataframe.

amazon_review_cleaned.createOrReplaceTempView("amazon_book_reviews")

# COMMAND ----------

# Distribution of the star ratings of book reviews

star_rating = spark.sql('''
  SELECT 
    overall AS star_rating, 
    COUNT(*) AS count 
  FROM
    amazon_book_reviews
  GROUP BY
    overall
  ORDER BY
    overall
''')

display(star_rating)

# COMMAND ----------

# Number of reviews over time

review_over_time = spark.sql('''
  SELECT 
    reviewTime AS date, 
    COUNT(*) AS count 
  FROM
    amazon_book_reviews
  WHERE
    reviewTime >= '2015-01-01'
  GROUP BY
    reviewTime
  ORDER BY
    reviewTime
''')

display(review_over_time)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Review Score Prediction

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC As comparison, without Spark we commonly use sklearn in Python for machine learning (read more: https://scikit-learn.org/stable/user_guide.html); or NLTK for natural language processing (read more: https://www.nltk.org/)

# COMMAND ----------

# Extract verified 5-star and 1-star reviews for prediction

prediction_df = amazon_review_cleaned.where( ((amazon_review_cleaned.overall == 1) | (amazon_review_cleaned.overall == 5)) \
                                             & amazon_review_cleaned.verified == True )

# This is equivalent to the following Spark SQL command:

prediction_df = spark.sql("SELECT * FROM amazon_book_reviews WHERE (overall = 1 OR overall = 5) AND verified = TRUE")

display(prediction_df)

# COMMAND ----------

# Take a stratified sample

print("Number of rows before sampling: ", prediction_df.count())
prediction_df_sampled = prediction_df.sampleBy("overall", fractions = {1:0.001, 5:0.001}, seed = 16).cache()
print("Number of rows after sampling: ", prediction_df_sampled.count())

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### TF-IDF with Hashing Trick + Random Forest

# COMMAND ----------

# Copy prediction data

prediction_tfidf_hash = prediction_df_sampled.select('*')

# COMMAND ----------

# Extract bigram

from pyspark.ml.feature import NGram
from pyspark.sql.functions import array_union

ngram = NGram(n = 2, inputCol="reviewWordCleaned", outputCol="reviewBigrams")
prediction_tfidf_hash = ngram.transform(prediction_tfidf_hash)

prediction_tfidf_hash = prediction_tfidf_hash.withColumn("reviewNgrams", \
                                                         array_union(prediction_tfidf_hash.reviewWordCleaned, \
                                                                     prediction_tfidf_hash.reviewBigrams))

# COMMAND ----------

# Getting tf-idf values for 1-2grams

from pyspark.ml.feature import HashingTF, IDF

hashtf = HashingTF(numFeatures=2**12, inputCol="reviewNgrams", outputCol='TF')
tf = hashtf.transform(prediction_tfidf_hash)
idf = IDF(minDocFreq=3, inputCol="TF", outputCol="TF-IDF")
idfModel = idf.fit(tf)
prediction_tfidf_hash = idfModel.transform(tf)

# COMMAND ----------

display(prediction_tfidf_hash)

# COMMAND ----------

# Random Forest

from pyspark.ml import Pipeline
from pyspark.ml.feature import IndexToString, StringIndexer
from pyspark.ml.classification import RandomForestClassifier

labelIndexer = StringIndexer(inputCol="overall", outputCol="indexedScore").fit(prediction_tfidf_hash)
rf = RandomForestClassifier(labelCol="indexedScore", featuresCol="TF-IDF", numTrees=40)
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

pipeline = Pipeline(stages=[labelIndexer, rf, labelConverter])

(trainingData, testData) = prediction_tfidf_hash.randomSplit([0.7, 0.3])

rf_model = pipeline.fit(trainingData)
predictions = rf_model.transform(testData)


# COMMAND ----------

display(predictions.select("overall", "indexedScore", "rawPrediction", "probability", "prediction", "predictedLabel"))

# COMMAND ----------

# Calculate AUC for train/test split

from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(labelCol="indexedScore", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print("AUC = %g" % auc)

# COMMAND ----------

# Performance evaluation with 5-fold cross validation

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

evaluator = BinaryClassificationEvaluator(labelCol="indexedScore", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
paramGrid = ParamGridBuilder().build()
cv = CrossValidator(estimator=pipeline, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds=5)
cvModel = cv.fit(prediction_tfidf_hash)

print("Average AUC = %g" % cvModel.avgMetrics[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Doc2Vec + Random Forest

# COMMAND ----------

# Copy prediction data

prediction_doc2vec = prediction_df_sampled.select('*')

# COMMAND ----------

# Calculate Doc2Vec

from pyspark.ml.feature import Word2Vec

word2Vec = Word2Vec(inputCol="reviewWordCleaned", outputCol="doc2vec")
w2v_model = word2Vec.fit(prediction_doc2vec)

prediction_doc2vec = w2v_model.transform(prediction_doc2vec)

# COMMAND ----------

display(prediction_doc2vec)

# COMMAND ----------

# Random Forest

from pyspark.ml import Pipeline
from pyspark.ml.feature import IndexToString, StringIndexer
from pyspark.ml.classification import RandomForestClassifier

labelIndexer = StringIndexer(inputCol="overall", outputCol="indexedScore").fit(prediction_doc2vec)
rf = RandomForestClassifier(labelCol="indexedScore", featuresCol="doc2vec", numTrees=40)
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

pipeline = Pipeline(stages=[labelIndexer, rf, labelConverter])

(trainingData, testData) = prediction_doc2vec.randomSplit([0.7, 0.3])

rf_model = pipeline.fit(trainingData)
predictions = rf_model.transform(testData)

from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(labelCol="indexedScore", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print("AUC = %g" % auc)

# COMMAND ----------

# Performance evaluation with 10-fold cross validation

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

paramGrid = ParamGridBuilder().build()
cv = CrossValidator(estimator=pipeline, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds=5)
cvModel = cv.fit(prediction_doc2vec)

print("Average AUC = %g" % cvModel.avgMetrics[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Interpretation

# COMMAND ----------

# Extract bigram

interpret_tfidf = prediction_df_sampled.select('*')

from pyspark.ml.feature import NGram
from pyspark.sql.functions import array_union

ngram = NGram(n = 2, inputCol="reviewWordCleaned", outputCol="reviewBigrams")
interpret_tfidf = ngram.transform(interpret_tfidf)

interpret_tfidf = interpret_tfidf.withColumn("reviewNgrams", \
                                             array_union(interpret_tfidf.reviewWordCleaned, \
                                                         interpret_tfidf.reviewBigrams))

# COMMAND ----------

# Calculating TF-IDF without hashing; limit vocabulary to top 2^12 (4096) ngrams

from pyspark.ml.feature import CountVectorizer, IDF

tf = CountVectorizer(inputCol="reviewNgrams", outputCol='TF', minDF=2.0, vocabSize=2**12)
tf_model = tf.fit(interpret_tfidf)
tf_transformed = tf_model.transform(interpret_tfidf)
idf = IDF(minDocFreq=3, inputCol="TF", outputCol="TF-IDF")
idfModel = idf.fit(tf_transformed)
interpret_tfidf = idfModel.transform(tf_transformed)

# COMMAND ----------

# Building a full Random Forest model with all the data, using TF-IDF embedding without hashing

from pyspark.ml import Pipeline
from pyspark.ml.feature import IndexToString, StringIndexer
from pyspark.ml.classification import RandomForestClassifier

labelIndexer = StringIndexer(inputCol="overall", outputCol="indexedScore").fit(interpret_tfidf)
rf = RandomForestClassifier(labelCol="indexedScore", featuresCol="TF-IDF", numTrees=40)
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

pipeline = Pipeline(stages=[labelIndexer, rf, labelConverter])

rf_model = pipeline.fit(interpret_tfidf)

# COMMAND ----------

# Getting feature importance from the Random Forest model

feature_importance = rf_model.stages[-2].featureImportances
print(feature_importance)

# COMMAND ----------

# Get the top 20 most important feature's indices, and its importance metric

import numpy as np
import pandas as pd

top20_indice = np.flip(np.argsort(feature_importance.toArray()))[:20].tolist()
top20_importance = []
for index in top20_indice:
    top20_importance.append(feature_importance[index])

top20_df = spark.createDataFrame(pd.DataFrame(list(zip(top20_indice, top20_importance)), columns =['index', 'importance']))

display(top20_df)

# COMMAND ----------

# Create a map between each ngram and its index

from pyspark.sql.functions import explode, udf, col
from pyspark.sql.types import *

make_list_udf = udf(lambda col: [col], ArrayType(StringType()))
remove_list_udf = udf(lambda col: col[0], StringType())

def get_index(col):
    if len(col.indices) == 0:
        return -1   # Mark the ngram's index as -1 if it is not the top 2^12 ngrams
    else:
        return int(col.indices[0])
get_index_udf = udf(get_index, IntegerType())

ngram_index = interpret_tfidf.select(explode(interpret_tfidf.reviewNgrams).alias("reviewNgrams")).distinct() \
                             .withColumn("reviewNgrams", make_list_udf("reviewNgrams"))
ngram_index = tf_model.transform(ngram_index)
ngram_index = ngram_index.withColumn("reviewNgrams", remove_list_udf("reviewNgrams")) \
                         .withColumn("index", get_index_udf("TF")) \
                         .select("reviewNgrams", "index")

# COMMAND ----------

display(ngram_index.where(ngram_index.index > -1))

# COMMAND ----------

# Find the ngrams that map to the top 20 most important features

# Note that if you used hashingTF for word embedding, there would be multiple ngrams under the same index, because of the collision introduced by hashing, all of which would share and contribute to one importance score, and we don't have a way to separate their contribution to the importance score.
# Here in order to avoid such collision (so just one index per ngram), I used CountVectorizer instead of HashingTF during encoding.

import pyspark.sql.functions as f

top20_ngram = top20_df.join(ngram_index, on="index", how="left_outer")
display(top20_ngram.groupby("importance").agg(f.collect_list(top20_ngram.reviewNgrams).alias("reviewNgrams")).orderBy("importance", ascending=False))