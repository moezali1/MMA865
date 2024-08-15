# Databricks notebook source
# Load the table
data = spark.sql("select * from default.reviews_train")

data = data.sample(False, 0.01, seed=0)

data = data.cache()

print((data.count(), len(data.columns)))

# COMMAND ----------

from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import Tokenizer, Normalizer, StopWordsCleaner, Stemmer

from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer, HashingTF, IDF, StringIndexer, SQLTransformer, IndexToString, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier


# convert text column to spark nlp document
document_assembler = DocumentAssembler() \
    .setInputCol("reviewText") \
    .setOutputCol("document")


# convert document to array of tokens
tokenizer = Tokenizer() \
  .setInputCols(["document"]) \
  .setOutputCol("token")
 
# clean tokens 
normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normalized")

# remove stopwords
stopwords_cleaner = StopWordsCleaner()\
      .setInputCols("normalized")\
      .setOutputCol("cleanTokens")\
      .setCaseSensitive(False)

# stems tokens to bring it to root form
stemmer = Stemmer() \
    .setInputCols(["cleanTokens"]) \
    .setOutputCol("stem")

# Convert custom document structure to array of tokens.
finisher = Finisher() \
    .setInputCols(["stem"]) \
    .setOutputCols(["token_features"]) \
    .setOutputAsArray(True) \
    .setCleanAnnotations(False)

# Generate Term Frequency
tf = CountVectorizer(inputCol="token_features", outputCol="rawFeatures", vocabSize=10000, minTF=1, minDF=50, maxDF=0.40)

# Generate Inverse Document Frequency weighting
idf = IDF(inputCol="rawFeatures", outputCol="idfFeatures", minDocFreq=5)

# Combine all features into one final "features" column
assembler = VectorAssembler(inputCols=["verified", "overall", "idfFeatures"], outputCol="features")

# Machine Learning Algorithm
#ml_alg  = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.0)
ml_alg  = RandomForestClassifier(numTrees=100, featureSubsetStrategy="auto", impurity='gini', maxDepth=4, maxBins=32)

nlp_pipeline = Pipeline(
    stages=[document_assembler, 
            tokenizer,
            normalizer,
            stopwords_cleaner, 
            stemmer, 
            finisher,
            tf,
            idf,
            assembler,
            ml_alg])
            

# COMMAND ----------

(trainingData, testData) = data.randomSplit([0.8, 0.2], seed = 47)

# COMMAND ----------

pipeline_model = nlp_pipeline.fit(trainingData)

# COMMAND ----------

predictions =  pipeline_model.transform(testData)
display(predictions)

# COMMAND ----------

predictions.groupBy("label").count().show()
predictions.groupBy("prediction").count().show()

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
pre_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
rec_evaluator = MulticlassClassificationEvaluator(metricName="weightedRecall")
pr_evaluator  = BinaryClassificationEvaluator(metricName="areaUnderPR")
auc_evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")

print("Test Accuracy       = %g" % (acc_evaluator.evaluate(predictions)))
print("Test Precision      = %g" % (pre_evaluator.evaluate(predictions)))
print("Test Recall         = %g" % (rec_evaluator.evaluate(predictions)))
print("Test areaUnderPR    = %g" % (pr_evaluator.evaluate(predictions)))
print("Test areaUnderROC   = %g" % (auc_evaluator.evaluate(predictions)))