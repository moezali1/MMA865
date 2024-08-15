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

data = spark.sql('SELECT * FROM employee')
data.display()

# COMMAND ----------

from pyspark.sql.functions import isnan, when, count, col
data.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in data.columns]).show()

# COMMAND ----------

data = data.na.drop("any")

# COMMAND ----------

data.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in data.columns]).show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Dealing with Categorical Features in PySpark

# COMMAND ----------

from pyspark.ml.feature import Imputer, VectorAssembler, VectorIndexer, OneHotEncoder, StringIndexer
from pyspark.sql.types import StringType, DoubleType

# COMMAND ----------

# numeric features
numeric_features =[f.name for f in data.schema.fields if not isinstance(f.dataType, StringType)]
print(numeric_features)

# COMMAND ----------

# categorical features
categorical_features = [f.name for f in data.schema.fields if isinstance(f.dataType, StringType)]
print(categorical_features)

# COMMAND ----------

# string indexer for department
department_indexer = StringIndexer(inputCol='department', outputCol='departmentIndex')

# COMMAND ----------

# one hot encoder for department
department_encoder = OneHotEncoder(inputCol='departmentIndex', outputCol='departmentVec')

# COMMAND ----------

# string indexer for salary
salary_indexer = StringIndexer(inputCol='salary', outputCol='salaryIndex')

# COMMAND ----------

# one hot encoder for salary
salary_encoder = OneHotEncoder(inputCol='salaryIndex', outputCol='salaryVec')

# COMMAND ----------

# MAGIC %md
# MAGIC # Vector Assembler

# COMMAND ----------

# vector assembler
assembler = VectorAssembler(inputCols=['satisfaction_level', 'last_evaluation', 'number_project', 
                                      'average_montly_hours', 'time_spend_company', 'Work_accident',
                                      'promotion_last_5years', 'departmentVec', 'salaryVec'],
                            outputCol='features')

# COMMAND ----------

# MAGIC %md
# MAGIC # Logistic Regression

# COMMAND ----------

# lets now import the model now
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol='features', labelCol='left')

# COMMAND ----------

# MAGIC %md
# MAGIC # What is Pipeline?
# MAGIC
# MAGIC A machine learning pipeline is a way to codify and automate the workflow it takes to produce a machine learning model. Machine learning pipelines consist of multiple sequential steps that do everything from data extraction and preprocessing to model training and deployment.
# MAGIC
# MAGIC For data science teams, the production pipeline should be the central product. It encapsulates all the learned best practices of producing a machine learning model for the organization’s use-case and allows the team to execute at scale. Whether you are maintaining multiple models in production or supporting a single model that needs to be updated frequently, an end-to-end machine learning pipeline is a must.
# MAGIC
# MAGIC (https://valohai.com/machine-learning-pipeline/)

# COMMAND ----------

from pyspark.ml import Pipeline

# COMMAND ----------

pipeline = Pipeline( stages= [salary_indexer, salary_encoder,
                              department_indexer, department_encoder,
                              assembler, lr])

# COMMAND ----------

# split data
train, test = data.randomSplit([0.7,0.3])

# COMMAND ----------

# fit the pipeline
pipeline_fit = pipeline.fit(train)

# COMMAND ----------

results = pipeline_fit.transform(test)

# COMMAND ----------

results.head(1)

# COMMAND ----------

results.select('prediction', 'left').show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Evaluator

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

evaluator = BinaryClassificationEvaluator(labelCol='left')

# COMMAND ----------

# by default it displays  AUC
evaluator.evaluate(results)

# COMMAND ----------

