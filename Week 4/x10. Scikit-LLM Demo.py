# Databricks notebook source
from skllm.config import SKLLMConfig
SKLLMConfig.set_openai_key(" ")

# COMMAND ----------

from skllm.models.gpt.classification.zero_shot import ZeroShotGPTClassifier
from skllm.datasets import get_classification_dataset

# demo sentiment analysis dataset
# labels: positive, negative, neutral
X, y = get_classification_dataset()

# COMMAND ----------

len(X), len(y)

# COMMAND ----------

print(X[0])
print('-------')
print(y[0])

# COMMAND ----------

print(X[1])
print('-------')
print(y[1])

# COMMAND ----------

# select only 5 from X and y
# X = X[:5]
# y = y[:5]

# COMMAND ----------

clf = ZeroShotGPTClassifier(openai_model="gpt-4")
clf.fit(X, y)
labels = clf.predict(X)

# COMMAND ----------

labels

# COMMAND ----------

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y, labels)
print(f"Accuracy: {accuracy:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC Learn more about scikit-llm: https://github.com/iryna-kondr/scikit-llm