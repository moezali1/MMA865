# Databricks notebook source
import os
os.environ['OPENAI_API_KEY'] = ' '

# COMMAND ----------

# download training dataset in jsonl format from moezali1 repo
!wget https://raw.githubusercontent.com/moezali1/MMA865/main/train.jsonl

# COMMAND ----------

# read jsonl file
import json

# Path to the JSONL file
file_path = 'train.jsonl'

# List to store the JSON objects
data = []

# Read the file
with open(file_path, 'r') as file:
    for line in file:
        # Parse the JSON object and add it to the list
        data.append(json.loads(line))

# COMMAND ----------

len(data)

# COMMAND ----------

for i in data:
  print(i)

# COMMAND ----------

# upload jsonl file. max limit 1GB
from openai import OpenAI
client = OpenAI()

client.files.create(
  file=open("train.jsonl", "rb"),
  purpose="fine-tune"
)

# COMMAND ----------

# start fine-tuning job, this may take 20-25 minutes for 3 epochs
client.fine_tuning.jobs.create(
  training_file="file-qrZwZBuQDe2O1Ha6pC43AxDu", 
  model="gpt-3.5-turbo"
)

# COMMAND ----------

# List fine-tuning jobs
client.fine_tuning.jobs.list(limit=10)

# COMMAND ----------

# Retrieve the state of a fine-tune job using job_id
client.fine_tuning.jobs.retrieve("ftjob-Na7BnF5y91wwGJ4EgxtzVyDD")

# COMMAND ----------

# Cancel a job
# client.fine_tuning.jobs.cancel("ftjob-abc123")

# List up to 10 events from a fine-tuning job
# client.fine_tuning.jobs.list_events(fine_tuning_job_id="ftjob-abc123", limit=10)

# Delete a fine-tuned model (must be an owner of the org the model was created in)
# client.models.delete("ft:gpt-3.5-turbo:acemeco:suffix:abc123")

# COMMAND ----------

from openai import OpenAI

response = client.chat.completions.create(
  model="ft:gpt-3.5-turbo-0613:personal::8jzj6H5D",
  messages=[
    {"role": "system", "content": "You are teaching assistant for Machine Learning. You should help to user to answer on his question."},
    {"role": "user", "content": "What is max_depth in random forest?"}
  ]
)

response.choices[0].message.content

# COMMAND ----------

from openai import OpenAI

response = client.chat.completions.create(
  model="ft:gpt-3.5-turbo-0613:personal::8jzj6H5D",
  messages=[
    {"role": "system", "content": "You are teaching assistant for Machine Learning. You should help to user to answer on his question."},
    {"role": "user", "content": "Why is Neural Network powerful than Logistic Regression?"}
  ]
)

response.choices[0].message.content

# COMMAND ----------

