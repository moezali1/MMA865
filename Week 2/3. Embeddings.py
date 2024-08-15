# Databricks notebook source
data = spark.sql('SELECT * from linkedin')
data = data.toPandas()
data.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # OpenAI Embeddings API

# COMMAND ----------

import openai
openai.api_key = " "

# COMMAND ----------

def get_embedding(text_to_embed):
	# Embed a line of text
	response = openai.Embedding.create(
    	model= "text-embedding-ada-002",
    	input=[text_to_embed]
	)
	# Extract the AI output embedding as a list of floats
	embedding = response['data'][0]['embedding']
    
	return embedding

# COMMAND ----------

data['embedding'] = data['Text'].astype(str).apply(get_embedding)
data.head()

# COMMAND ----------

data['embedding'][0]

# COMMAND ----------

len(data['embedding'][0])

# COMMAND ----------

import numpy as np
np.array(data['embedding'][0])

# COMMAND ----------

np.array(data['embedding'][0]).shape

# COMMAND ----------

# MAGIC %md
# MAGIC #HuggingFace open-source Embeddings

# COMMAND ----------

import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Define a function to get embeddings for a piece of text using BERT
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs['last_hidden_state'][:, 0, :].squeeze().numpy()

# COMMAND ----------

# Get embeddings for each text
data['embedding_hf'] = data['Text'].apply(get_embedding)

data.head(1)

# COMMAND ----------

