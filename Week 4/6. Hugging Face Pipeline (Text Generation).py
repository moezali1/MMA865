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
# MAGIC # Text Generation

# COMMAND ----------

generator = pipeline('text-generation')

# COMMAND ----------

generator("Adam and Natalie are good friends.")

# COMMAND ----------

generator("Since I have joined Queens University.")  

# COMMAND ----------

generator("Canada is a country.....")