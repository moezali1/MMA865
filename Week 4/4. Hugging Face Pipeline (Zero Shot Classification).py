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
# MAGIC # Zero Shot Classification

# COMMAND ----------

classifier = pipeline('zero-shot-classification')

# COMMAND ----------

text = """
PyCaret is an open-source, low-code machine learning library in Python that automates machine learning workflows. It is an end-to-end machine learning and model management tool that speeds up the experiment cycle exponentially and makes you more productive.

In comparison with the other open-source machine learning libraries, PyCaret is an alternate low-code library that can be used to replace hundreds of lines of code with few words only. This makes experiments exponentially fast and efficient. PyCaret is essentially a Python wrapper around several machine learning libraries and frameworks such as scikit-learn, XGBoost, LightGBM, CatBoost, spaCy, Optuna, Hyperopt, Ray, and many more.

The design and simplicity of PyCaret is inspired by the emerging role of citizen data scientists, a term first used by Gartner. Citizen Data Scientists are power users who can perform both simple and moderately sophisticated analytical tasks that would previously have required more expertise. Seasoned data scientists are often difficult to find and expensive to hire but citizen data scientists can be an effective way to mitigate this gap and address data-related challenges in the business setting.

PyCaret is a great library which not only simplifies the machine learning tasks for citizen data scientists but also helps new startups to reduce the cost of investing in a team of data scientists. Therefore, this library has not only helped the citizen data scientists but has also helped individuals who want to start exploring the field of data science, having no prior knowledge in this field. Iniitial idea of PyCaret was inspired by Caret library in R.
"""

# COMMAND ----------

classifier(text, candidate_labels = ['pycaret', 'data science', 'machine learning', 'politics', 'music'])

# COMMAND ----------

text2 = """Today, the Honourable Chrystia Freeland, Deputy Prime Minister and Minister of Finance, 
           the Honourable Ahmed Hussen, Minister of Families, Children and Social Development, the Honourable Sandy Silver, 
           Yukon Premier, and the Honourable Jeanie McLean, Yukon Minister of Education, announced an agreement that 
           significantly improves early learning and child care for children in Yukon."""

# COMMAND ----------

classifier(text2, candidate_labels = ['education', 'data science', 'environment', 'politics', 'music'])

# COMMAND ----------

text3 = """Pre-existing T-cell immunity to SARS-CoV-2 in unexposed healthy controls in Ecuador, as 
            detected with a COVID-19 Interferon-Gamma Release Assay."""

# COMMAND ----------

classifier(text3, candidate_labels = ['COVID-19', 'health', 'virus', 'politics', 'music'])