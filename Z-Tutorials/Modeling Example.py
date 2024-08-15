# Databricks notebook source
# DBTITLE 1,Load Data
# Load in the table
df = spark.sql("select * from default.reviews_train").sample(0.1)

df = df.cache()

print((df.count(), len(df.columns)))

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# DBTITLE 1,Data Wrangling/Prep
# For our intitial modeling efforts, we are not going to use the following features
drop_list = ['summary', 'asin', 'reviewID', 'reviewerID', 'summary', 'unixReviewTime','reviewTime', 'image', 'style', 'reviewerName']
df = df.select([column for column in df.columns if column not in drop_list])
df = df.na.drop(subset=["reviewText", "label"])
df.show(5)
print((df.count(), len(df.columns)))

# COMMAND ----------

# DBTITLE 1,Create a Data Transformation/ML Pipeline
# In Spark's MLLib, it's considered good practice to combine all the preprocessing steps into a pipeline.
# That way, you can run the same steps on both the training data, and testing data and beyond (new data)
# without copying and pasting any code.

# It is possible to run all of these steps one-by-one, outside of a Pipeline, if desired. But that's
# not how I am going to do it here.

from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, VectorAssembler, IDF
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayes
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# We'll tokenize the text using a simple RegexTokenizer
tokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words", pattern="\\W")

# Remove standard Stopwords
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered")

# TODO: insert other clearning steps here (and put into the pipeline, of course!)
# E.g., n-grams? document length?


# Vectorize the sentences using simple BOW method. Other methods are possible:
# https://spark.apache.org/docs/2.2.0/ml-features.html#feature-extractors
tf = CountVectorizer(inputCol="filtered", outputCol="rawFeatures", vocabSize=2000, minTF=1, maxDF=0.40)

# Generate Inverse Document Frequency weighting
idf = IDF(inputCol="rawFeatures", outputCol="idfFeatures", minDocFreq=100)

# Combine all features into one final "features" column
assembler = VectorAssembler(inputCols=["verified", "overall", "idfFeatures"], outputCol="features")

# Machine Learning Algorithm
ml_alg  = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.0)

pipeline = Pipeline(stages=[tokenizer, stopwordsRemover, tf, idf, assembler, ml_alg])

paramGrid = ParamGridBuilder() \
    .addGrid(ml_alg.regParam, [0.3, 0.5, 0.7]) \
    .addGrid(ml_alg.elasticNetParam, [0.0]) \
    .addGrid(tf.minTF, [1, 100, 1000]) \
    .addGrid(tf.vocabSize, [500, 1000, 2500, 5000]) \
    .build()


# COMMAND ----------

# DBTITLE 1,Split into testing/training
# set seed for reproducibility
(trainingData, testData) = df.randomSplit([0.9, 0.1], seed = 47)
print("Training Dataset Count: " + str(trainingData.count()))
print("Test Dataset Count:     " + str(testData.count()))

# COMMAND ----------

# DBTITLE 1,Transform Training Data
pipelineFit = pipeline.fit(trainingData)

# COMMAND ----------

# DBTITLE 1,Predict Testing Data
predictions = pipelineFit.transform(testData)
predictions.groupBy("prediction").count().show()

# COMMAND ----------

# DBTITLE 1,Performance Metrics on Testing Data
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
pre_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
rec_evaluator = MulticlassClassificationEvaluator(metricName="weightedRecall")
pr_evaluator  = BinaryClassificationEvaluator(metricName="areaUnderPR")
auc_evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")

#print("Test Accuracy       = %g" % (acc_evaluator.evaluate(predictions)))
#print("Test Precision      = %g" % (pre_evaluator.evaluate(predictions)))
#print("Test Recall         = %g" % (rec_evaluator.evaluate(predictions)))
#print("Test areaUnderPR    = %g" % (pr_evaluator.evaluate(predictions)))
print("Test areaUnderROC   = %g" % (auc_evaluator.evaluate(predictions)))

# COMMAND ----------

# DBTITLE 1,Make Predictions on Kaggle Test Data
test_df = spark.sql("select * from default.reviews_test")
kaggle_pred = pipelineFit.transform(test_df)
kaggle_pred.show(5)
kaggle_pred.groupBy("prediction").count().show()

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

probelement=udf(lambda v:float(v[1]),FloatType())
submission_data = kaggle_pred.select('reviewID', probelement('probability')).withColumnRenamed('<lambda>(probability)', 'label')

# COMMAND ----------

# Download this and submit to Kaggle!
display(submission_data.select(["reviewID", "label"]))