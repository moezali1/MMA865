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
# MAGIC # Question Answering

# COMMAND ----------

nlp = pipeline("question-answering")

# COMMAND ----------

context = "All the Students are pretty excited in the MMA programme. After the machine learning course, big data is introduced."

# COMMAND ----------

nlp(question="What is the next course after machine learning?", context=context)

# COMMAND ----------

context2 = "PyCaret is an open-source low code machine learning library in Python."

# COMMAND ----------

nlp(question="Which language is PyCaret written?", context=context2)

# COMMAND ----------

context3 = "I can't wait to finish MMA programme so that I can go to Greece for vacations."

# COMMAND ----------

nlp(question="where do you want to go after MMA?", context=context3)

# COMMAND ----------

context4 = 'I am feeling too sleepy'

# COMMAND ----------

nlp(question="Are you hungry?", context=context4)

# COMMAND ----------

context5 = 'There was a farmer, he had a dog and Bingo was his name.'

# COMMAND ----------

nlp(question="Whose name is Bingo?", context=context5)