# Databricks notebook source
# DBTITLE 1,Load Data
from pyspark.sql.functions import lit

# Load in one of the tables
df = spark.sql("select * from default.reviews_train")

# Take a sample (useful for code development purposes)
df = df.sample(False, 0.15, seed=0)

df = df.cache()

print((df.count(), len(df.columns)))

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# DBTITLE 1,Describe Data
# Let's look at some quick summary statistics
df.withColumnRenamed('summary', 'summaryCol').describe().show()

# COMMAND ----------

from pyspark.sql.functions import col
display(df.groupBy("overall").count().orderBy("overall"))

# COMMAND ----------

# The most common product IDs
display(df.groupBy("asin").count().orderBy(col("count").desc()).head(50))

# COMMAND ----------

display(df.groupBy("label").count().orderBy("label"))

# COMMAND ----------

# DBTITLE 1,Create a Data Transformation Pipeline
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
from pyspark.sql import functions as f

# We'll tokenize the text using a simple RegexTokenizer
tokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words", pattern="\\W")

# Remove standard Stopwords
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered")

pipeline = Pipeline(stages=[tokenizer, stopwordsRemover])

pipelineFit = pipeline.fit(df)
df = pipelineFit.transform(df)

# COMMAND ----------

# DBTITLE 1,Get Term Frequencies
counts = df.select(f.explode('filtered').alias('col')).groupBy('col').count().sort(f.desc('count')).collect()
display(counts)