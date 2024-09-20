# Databricks notebook source
# MAGIC %md
# MAGIC # What is PySpark
# MAGIC
# MAGIC PySpark is the collaboration of Apache Spark and Python.
# MAGIC
# MAGIC Apache Spark is an open-source cluster-computing framework, built around speed, ease of use, and streaming analytics whereas Python is a general-purpose, high-level programming language. It provides a wide range of libraries and is majorly used for Machine Learning and Real-Time Streaming Analytics.
# MAGIC
# MAGIC In other words, it is a Python API for Spark that lets you harness the simplicity of Python and the power of Apache Spark in order to tame Big Data.
# MAGIC
# MAGIC (https://www.edureka.co/blog/pyspark-programming/#Learnpysparkprogramming)

# COMMAND ----------

# MAGIC %md
# MAGIC # Loading Dataset

# COMMAND ----------

# loading dataset
data = spark.sql('SELECT * FROM boston')

# COMMAND ----------

type(data)

# COMMAND ----------

data.show()

# COMMAND ----------

data.display()

# COMMAND ----------

data.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC # Show Schema

# COMMAND ----------

data.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # Column Names

# COMMAND ----------

data.columns

# COMMAND ----------

# MAGIC %md
# MAGIC # Statistical Summary

# COMMAND ----------

# this is similar descr() function in R and .describe() in Pandas Python
data.describe().display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Selecting Column

# COMMAND ----------

# this will work in Python but not in Spark
data['crim']

# COMMAND ----------

# notice the type of this
type(data['crim'])

# COMMAND ----------

# right way to select is the select function
data.select('crim').display()

# COMMAND ----------

# select multiple columns
data.select(['crim', 'nox']).display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Adding New Columns

# COMMAND ----------

# adding new column
data.withColumn('crim_times_two', data['crim']*2).display()

# COMMAND ----------

# check data again
data.display()

# COMMAND ----------

# to add column permanently you have to assign it to variable
data = data.withColumn('crim_times_two', data['crim']*2)
data.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Using SQL in Spark

# COMMAND ----------

# register dataframe into SQL view
data.createOrReplaceTempView('sql_table')

# COMMAND ----------

output = spark.sql('SELECT * from sql_table')
output.display()

# COMMAND ----------

# little bit more complicated query
output2 = spark.sql('select * from sql_table where age < 7')
output2.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Filtering using PySpark

# COMMAND ----------

# filter data using native filter function
data.filter('age < 7').display()

# COMMAND ----------

# multiple conditions in filter function
data.filter((data['age'] < 7) & (data['crim'] < 0.1)).display()

# COMMAND ----------

# using collect instead of show
# show function only shows the dataset, it doesn't store it in variable
# most times you will need access to data and so you can use collect
# it will return a list and then you can access the data normally as you would do
# with python lists

results = data.filter((data['age'] < 7) & (data['crim'] < 0.1)).collect()

# COMMAND ----------

results

# COMMAND ----------

results[1]

# COMMAND ----------

# MAGIC %md
# MAGIC # Groupby Operations

# COMMAND ----------

# average diamond price by cut
data.groupBy('rad').mean().display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Learn More? 
# MAGIC
# MAGIC Check this documentation: https://spark.apache.org/docs/latest/api/python/getting_started/index.html