# Databricks notebook source
import os
os.environ["OPENAI_API_KEY"] = " "

# COMMAND ----------

from embedchain import Pipeline as App
elon_bot = App()

# COMMAND ----------

elon_bot.add("https://en.wikipedia.org/wiki/Elon_Musk")
elon_bot.add("https://www.forbes.com/profile/elon-musk")

# COMMAND ----------

elon_bot.query("How many companies does Elon Musk run and name those?")

# COMMAND ----------

mma_bot = App()
mma_bot.add('https://smith.queensu.ca/grad_studies/mma/index.php')
mma_bot.add('https://smith.queensu.ca/grad_studies/mma/program-details/curriculum.php')

# COMMAND ----------

mma_bot.query('What is the ideal audience for MMA program?')

# COMMAND ----------

mma_bot.query('What kind of technical skills I will learn in this program?')

# COMMAND ----------

mma_bot.query('Do I need to have any pre-requisite technical knowledge such as Python to apply for this program?')

# COMMAND ----------

