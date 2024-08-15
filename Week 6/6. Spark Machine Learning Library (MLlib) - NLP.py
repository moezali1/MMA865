# Databricks notebook source
# MAGIC %md
# MAGIC # What is Spark MLlib?
# MAGIC
# MAGIC Apache Spark’s Machine Learning Library (MLlib) is designed for simplicity, scalability, and easy integration with other tools. With the scalability, language compatibility, and speed of Spark, data scientists can focus on their data problems and models instead of solving the complexities surrounding distributed data (such as infrastructure, configurations, and so on). Built on top of Spark, MLlib is a scalable machine learning library consisting of common learning algorithms and utilities, including classification, regression, clustering, collaborative filtering, dimensionality reduction, and underlying optimization primitives. Spark MLLib seamlessly integrates with other Spark components such as Spark SQL, Spark Streaming, and DataFrames and is installed in the Databricks runtime. The library is usable in Java, Scala, and Python as part of Spark applications, so that you can include it in complete workflows. MLlib allows for preprocessing, munging, training of models, and making predictions at scale on data. You can even use models trained in MLlib to make predictions in Structured Streaming. Spark provides a sophisticated machine learning API for performing a variety of machine learning tasks, from classification to regression, clustering to deep learning. 
# MAGIC
# MAGIC (https://databricks.com/glossary/what-is-machine-learning-library)

# COMMAND ----------

# MAGIC %md
# MAGIC # Loading Dataset

# COMMAND ----------

data = spark.sql('SELECT * FROM spam')
data.display()

# COMMAND ----------

# check schema
data.printSchema()

# COMMAND ----------

# check the number of rows in the dataset
data.count()

# COMMAND ----------

# check missing values
from pyspark.sql.functions import isnan, when, count, col
data.select([count(when(isnan(c), c)).alias(c) for c in data.columns]).show()

# COMMAND ----------

# check if there is a clear boundary bw spam and ham
# notice that data has some quality issues 'ham"""'
data.groupBy('label').mean().show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Processing the text data

# COMMAND ----------

from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer

# COMMAND ----------

# algorithms cannot work directly with string labels (ham/spam)
# so we have to convert it into 1 or 0
label_encoder = StringIndexer(inputCol='label', outputCol='labelEncoded')

# COMMAND ----------

# tokenizer parses the string text into tokens
tokenizer = Tokenizer(inputCol='message', outputCol='textToken')

# COMMAND ----------

# stop word remover removes common english words
stopper = StopWordsRemover(inputCol='textToken', outputCol= 'stopperToken')

# COMMAND ----------

# count vectorizer creates a vector of token count
count_vec = CountVectorizer(inputCol='stopperToken', outputCol='countVector')

# COMMAND ----------

# idf converts the countvector into tf-idf
tf_idf = IDF(inputCol='countVector', outputCol = 'tfidfVector')

# COMMAND ----------

# vector assembler creates a sparse vector for ML 
# same as other examples done in past
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=['tfidfVector'], outputCol='features')

# COMMAND ----------

# add everything in pipeline
from pyspark.ml import Pipeline
data_pipeline = Pipeline(stages = [label_encoder, tokenizer, stopper, count_vec, tf_idf, assembler])

# COMMAND ----------

# fit the data pipeline
data_pipeline_fit = data_pipeline.fit(data)

# COMMAND ----------

# see the transformed data
final_data = data_pipeline_fit.transform(data)
final_data = final_data.select('features','labelEncoded')
final_data.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Test Split

# COMMAND ----------

train, test = final_data.randomSplit([0.7,0.3])

# COMMAND ----------

train.count(), test.count()

# COMMAND ----------

# MAGIC %md
# MAGIC # Naive Bayes

# COMMAND ----------

from pyspark.ml.classification import NaiveBayes

# COMMAND ----------

nb = NaiveBayes(featuresCol = 'features', labelCol='labelEncoded')

# COMMAND ----------

nb_fit = nb.fit(train)

# COMMAND ----------

results = nb_fit.transform(test)

# COMMAND ----------

results.display()

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------

# accuracy
accuracy = MulticlassClassificationEvaluator(labelCol = 'labelEncoded', metricName='accuracy')
accuracy.evaluate(results)

# COMMAND ----------

# f1
f1 = MulticlassClassificationEvaluator(labelCol = 'labelEncoded', metricName='f1')
f1.evaluate(results)

# COMMAND ----------

# log loss
recall = MulticlassClassificationEvaluator(labelCol = 'labelEncoded', metricName='logLoss')
recall.evaluate(results)