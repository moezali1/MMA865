# Databricks notebook source
import os
os.environ["OPENAI_API_KEY"] = " "

# COMMAND ----------

# MAGIC %md
# MAGIC # Text Generation API

# COMMAND ----------

from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  ]
)

print(completion.choices[0].message)

# COMMAND ----------

from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-4-1106-preview",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  ]
)

print(completion.choices[0].message)

# COMMAND ----------

# for clean output access `content`
print(completion.choices[0].message.content)

# COMMAND ----------

# MAGIC %md
# MAGIC All OpenAI models: https://platform.openai.com/docs/models/overview

# COMMAND ----------

# MAGIC %md
# MAGIC # Image Generation API

# COMMAND ----------

from openai import OpenAI
client = OpenAI()

response = client.images.generate(
  model="dall-e-3",
  prompt="man sitting on bench sad looking into darkness, line illustration by Herge, dark aesthetic",
  size="1024x1024",
  quality="standard",
  n=1,
)

image_url = response.data[0].url
image_url

# COMMAND ----------

# MAGIC %md
# MAGIC # Vision API

# COMMAND ----------

from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
  model="gpt-4-vision-preview",
  messages=[
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What’s in this image?"},
        {
          "type": "image_url",
          "image_url": {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
          },
        },
      ],
    }
  ],
  max_tokens=300,
)

print(response.choices[0])

# COMMAND ----------

# MAGIC %md
# MAGIC # Text to Speech API

# COMMAND ----------

from pathlib import Path
from openai import OpenAI
client = OpenAI()

input = """

There are two major learning goals in this course:
1.	Natural Language Processing
Natural Language Processing (NLP) is one of the six AI disciplines. We will discuss the major practice areas of NLP and several of its use-cases across many different industries. These key areas include Information Extraction, Document Classification, Sentiment Analysis, Language Generation, Chatbots, and Machine Translation. 

"""
speech_file_path = "/speech.mp3"
response = client.audio.speech.create(
  model="tts-1",
  voice="alloy",
  input=input
)

response.stream_to_file(speech_file_path)

# COMMAND ----------

from IPython.display import Audio
Audio("/speech.mp3")


# COMMAND ----------

# MAGIC %md
# MAGIC # Speech to Text API

# COMMAND ----------

from openai import OpenAI
client = OpenAI()

audio_file= open("/speech.mp3", "rb")
transcript = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file
)

transcript.text