# Databricks notebook source
import os
os.environ["OPENAI_API_KEY"] = ' '
os.environ["HF_API_KEY"] = ' '

# COMMAND ----------

# MAGIC %md
# MAGIC # Using OpenAI Models with LangChain

# COMMAND ----------

from langchain_openai import ChatOpenAI
llm = ChatOpenAI()
llm.invoke("Tell me a joke about data scientist")

# COMMAND ----------

llm.invoke("Tell me a joke about data scientist").pretty_print()

# COMMAND ----------

# MAGIC %md
# MAGIC # Chain

# COMMAND ----------

from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a expert travel consultant. I will give you country name and you will return a json dictionary of number 1 city to visit and top 10 things to do in that city as a bulleted list."),
    ("user", "{input}")
])

# COMMAND ----------

chain = prompt | llm 

# COMMAND ----------

chain.invoke({"input": "Canada"})

# COMMAND ----------

type(chain.invoke({"input": "Canada"}))

# COMMAND ----------

chain.invoke({"input": "Canada"}).pretty_print()

# COMMAND ----------

# MAGIC %md
# MAGIC # String Output Parser

# COMMAND ----------

from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# COMMAND ----------

chain.invoke({"input": "France"})

# COMMAND ----------

# MAGIC %md
# MAGIC # Prompt Templates

# COMMAND ----------

from langchain.prompts import PromptTemplate

# COMMAND ----------

prompt = (
    PromptTemplate.from_template("Tell me a joke about {topic}")
    + ", make it funny"
    + "\n\nand in {language}"
)

# COMMAND ----------

prompt

# COMMAND ----------

prompt.format(topic="sports", language="english")

# COMMAND ----------

llm.invoke(prompt.format(topic="sports", language="english")).pretty_print()

# COMMAND ----------

# MAGIC %md
# MAGIC # Chat Models

# COMMAND ----------

from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(content="What is the purpose of model regularization? Write a 1000 word essay"),
]

# COMMAND ----------

llm.invoke(messages)

# COMMAND ----------

for chunk in llm.stream(messages):
    print(chunk.content, end="", flush=True)

# COMMAND ----------

# MAGIC %md
# MAGIC # Embedding Models

# COMMAND ----------

from langchain_openai import OpenAIEmbeddings
embeddings_model = OpenAIEmbeddings()

# COMMAND ----------

embeddings = embeddings_model.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ]
)
len(embeddings), len(embeddings[0])

# COMMAND ----------

# MAGIC %md
# MAGIC # Using Models from HuggingFace

# COMMAND ----------

from langchain import HuggingFaceHub
llm_hf = HuggingFaceHub(repo_id = "google/flan-t5-large", huggingfacehub_api_token = os.environ.get('HF_API_KEY'))
print(llm_hf("What is the purpose of model regularization?"))

# COMMAND ----------

from langchain import HuggingFaceHub
llm_hf = HuggingFaceHub(repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1", huggingfacehub_api_token = os.environ.get('HF_API_KEY'))
print(llm_hf("What is the purpose of model regularization?"))

# COMMAND ----------

# from langchain import HuggingFaceHub
# llm_hf = HuggingFaceHub(repo_id = "TencentARC/LLaMA-Pro-8B", huggingfacehub_api_token = os.environ.get('HF_API_KEY'))
# print(llm_hf("What is the purpose of model regularization?"))

# COMMAND ----------

