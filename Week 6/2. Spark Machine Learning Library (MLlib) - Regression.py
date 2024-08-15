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

data = spark.sql('SELECT * FROM boston')
data.display()

# COMMAND ----------

data.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # Transform the Data to work with MLlib

# COMMAND ----------

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

data.columns

# COMMAND ----------

assembler = VectorAssembler(inputCols=['crim', 'zn', 'indus', 'chas','nox','rm','age','dis','rad','tax','ptratio','black','lstat'],
                            outputCol='features')

# COMMAND ----------

output = assembler.transform(data)

# COMMAND ----------

output.display()

# COMMAND ----------

final_data = output.select('features', 'medv')

# COMMAND ----------

final_data.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Train Test Split

# COMMAND ----------

train, test = final_data.randomSplit([0.7,0.3])

# COMMAND ----------

train.describe().display()

# COMMAND ----------

test.describe().display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Linear Regression

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

# COMMAND ----------

lr = LinearRegression(labelCol='medv')

# COMMAND ----------

lr_fit = lr.fit(train)

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Evaluation

# COMMAND ----------

# calling evaluate function on fitted model
test_results = lr_fit.evaluate(test)

# COMMAND ----------

# Root Mean Squared Error (RMSE) on test set
test_results.rootMeanSquaredError

# COMMAND ----------

# R2 on test set
test_results.r2

# COMMAND ----------

# Mean Absolute Error (MAE) on test set
test_results.meanAbsoluteError

# COMMAND ----------

# Mean Squared Error (MSE) on test set
test_results.meanSquaredError

# COMMAND ----------

# residuals on test set
test_results.residuals.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Scoring on new data

# COMMAND ----------

# lets create a new data by removing target col from test set
new_data = test.select('features')
new_data.display()

# COMMAND ----------

pred = lr_fit.transform(new_data)
pred.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Random Forest

# COMMAND ----------

from pyspark.ml.regression import RandomForestRegressor

# COMMAND ----------

# create random forest instance
rf = RandomForestRegressor(labelCol='medv')

# COMMAND ----------

# fit the model
rf_fit = rf.fit(train)

# COMMAND ----------

# generate predictions on new data
rf_fit.transform(new_data).display()

# COMMAND ----------

# feature importance
rf_fit.featureImportances

# COMMAND ----------

# MAGIC %md
# MAGIC # Cross-Validation / Hyperparameter Tuning, etc.
# MAGIC
# MAGIC Read Documentation: https://spark.apache.org/docs/latest/ml-tuning.html

# COMMAND ----------

