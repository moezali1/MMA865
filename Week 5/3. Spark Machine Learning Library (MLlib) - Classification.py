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

data = spark.sql('SELECT * FROM cancer')
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

assembler = VectorAssembler(inputCols=['age','menopause','tumor-size','inv-nodes','node-caps','deg-malig',\
                                       'breast', 'breast-quad','irradiat'], outputCol='features')

# COMMAND ----------

output = assembler.transform(data)

# COMMAND ----------

output.display()

# COMMAND ----------

final_data = output.select('features', 'Class')
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
# MAGIC # Model Training

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

# COMMAND ----------

lr = LogisticRegression(labelCol='Class')

# COMMAND ----------

lr_fit = lr.fit(train)

# COMMAND ----------

# check summary
lr_fit.summary.predictions.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Evaluation

# COMMAND ----------

test.display()

# COMMAND ----------

# calling evaluate function on fitted model
test_results = lr_fit.evaluate(test)

# COMMAND ----------

# check Accuracy on test set
test_results.accuracy

# COMMAND ----------

# AUC on test set
test_results.areaUnderROC

# COMMAND ----------

# precision by label
test_results.precisionByLabel

# COMMAND ----------

# Recall by label
test_results.recallByLabel

# COMMAND ----------

# You can access raw predictions, probability and prediction
test_results.predictions.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # PySpark Evaluator

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

evaluator = BinaryClassificationEvaluator(labelCol='Class')

# COMMAND ----------

# this will match with the above AUC output on test set
evaluator.evaluate(test_results.predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ## AUC Plot

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC import matplotlib.pyplot as plt
# MAGIC
# MAGIC ModelSummary = lr_fit.summary
# MAGIC roc = ModelSummary.roc.toPandas()
# MAGIC plt.plot(roc['FPR'], roc['TPR'])
# MAGIC plt.xlabel('True Positive Rate')
# MAGIC plt.ylabel('False Positive Rate')
# MAGIC plt.title('ROC-AUC Curve')
# MAGIC plt.show()
# MAGIC print('Training Set AUC: ' + str(ModelSummary.areaUnderROC))

# COMMAND ----------

# MAGIC %md
# MAGIC # Scoring new data

# COMMAND ----------

# lets create a new data by removing target col from test set
new_data = test.select('features')
new_data.display()

# COMMAND ----------

pred = lr_fit.transform(new_data)
pred.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Gradient Boosted Tree

# COMMAND ----------

from pyspark.ml.classification import GBTClassifier

# COMMAND ----------

# create random forest instance
gbt = GBTClassifier(labelCol='Class')

# COMMAND ----------

# fit the model
gbt_fit = gbt.fit(train)

# COMMAND ----------

# generate predictions on new data
gbt_fit.transform(new_data).display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Cross-Validation / Hyperparameter Tuning, etc.
# MAGIC
# MAGIC Read Documentation: https://spark.apache.org/docs/latest/ml-tuning.html