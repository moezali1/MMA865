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

data = spark.sql('SELECT * FROM JEWELLERY')
data.display()

# COMMAND ----------

data.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # Vector Assembler

# COMMAND ----------

# import vector assembler
from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

data.columns

# COMMAND ----------

assembler = VectorAssembler(inputCols=data.columns, outputCol='features')

# COMMAND ----------

final_data = assembler.transform(data)

# COMMAND ----------

# notice a new column features created using VectorAssembler
# it is same as what we have seen in supervised experiments
final_data.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Scaling

# COMMAND ----------

from pyspark.ml.feature import StandardScaler

# COMMAND ----------

scaler = StandardScaler(inputCol='features', outputCol='scaledFeatures')

# COMMAND ----------

# fit and transform in one step
final_data = scaler.fit(final_data).transform(final_data)

# COMMAND ----------

final_data.display()

# COMMAND ----------

final_data.head(1)

# COMMAND ----------

# MAGIC %md
# MAGIC # KMeans Clustering

# COMMAND ----------

from pyspark.ml.clustering import KMeans

# COMMAND ----------

kmeans = KMeans(featuresCol='scaledFeatures', k=4)

# COMMAND ----------

kmeans_fit = kmeans.fit(final_data)

# COMMAND ----------

# check cluster centers
kmeans_fit.clusterCenters()

# COMMAND ----------

# which distance measure it is using
kmeans_fit.getDistanceMeasure()

# COMMAND ----------

# do you know how to check all the methods and attributes available?
# using python's builtin goodness "dir"

dir(kmeans_fit)

# COMMAND ----------

# MAGIC %md
# MAGIC # Generate Cluster Labels?

# COMMAND ----------

kmeans_fit.transform(final_data).display()

# COMMAND ----------

