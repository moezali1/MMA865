# Databricks notebook source
# MAGIC %md
# MAGIC # Hugging Face Transformers 🤗
# MAGIC
# MAGIC The Hugging Face transformers package is an immensely popular Python library providing pretrained models that are extraordinarily useful for a variety of natural language processing (NLP) tasks. It previously supported only PyTorch, but, as of late 2019, TensorFlow 2 is supported as well. While the library can be used for many tasks from Natural Language Inference (NLI) to Question-Answering, text classification remains one of the most popular and practical use cases.
# MAGIC
# MAGIC Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between Jax, PyTorch and TensorFlow.
# MAGIC
# MAGIC https://huggingface.co/transformers/

# COMMAND ----------

from transformers import pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC # Sentiment Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ## distilbert-base-uncased-finetuned-sst-2-english

# COMMAND ----------

classifier = pipeline('sentiment-analysis')

# COMMAND ----------

classifier("The food was good overall but Pizza was horrible")

# COMMAND ----------

classifier("I am loving it.")

# COMMAND ----------

classifier("I hate waking up early on the weekends.")

# COMMAND ----------

data = spark.sql('SELECT * from amazon_cells')
data = data.toPandas()
data.head()

# COMMAND ----------

text = list(data['Text'][:5])
text

# COMMAND ----------

for i in text:
    print(classifier(i))

# COMMAND ----------

# MAGIC %md
# MAGIC ## cardiffnlp/twitter-roberta-base-emotion
# MAGIC https://huggingface.co/cardiffnlp/twitter-roberta-base-emotion?text=I+am+very+excited+about+Big+Data+course.+I+was+looking+forward+to+it.

# COMMAND ----------

classifier2 = pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-emotion')

# COMMAND ----------

for i in text:
    print(classifier2(i))

# COMMAND ----------

for i in text:
    print(classifier2(i))

# COMMAND ----------

