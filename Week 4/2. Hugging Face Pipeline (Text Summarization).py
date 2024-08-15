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
# MAGIC # Text Summarization

# COMMAND ----------

summarizer = pipeline('summarization')

# COMMAND ----------

# MAGIC %md
# MAGIC ## MMA865 Syllabus

# COMMAND ----------

summarizer("""
COURSE DESCRIPTION
There are two main learning objectives of this course:
1.	Big Data 
In this course, we discuss the relatively new discipline known as big data. The term big data is overloaded and simultaneously refers to the description of the data itself, the IT techniques necessary for handling massive data sets, the analytics and AI techniques that are possible with large data sets, and the business decisions/problems that we can address by leveraging the data. We will investigate all of these areas. We will also survey key IT technologies, such as enterprise data warehouses, enterprise data lakes, Hadoop, Spark, NoSQL databases, the cloud, and more. 
2.	Natural Language Processing
Natural Language Processing is one of the six AI disciplines. In this part of the course, we will discuss the major practice areas of NLP such as Document Classification, Sentiment Analysis, Language Generation, Chatbots, and Machine Translation. We will also discuss the relatively new concept of Transfer Learning in NLP and how it has derived the massive progress in the area of language modeling (BERT, GPT3, BART, etc.). We will also discuss EDA and preprocessing on text data in much detail as it forms the foundation of NLP. You will learn tools and techniques for handling text data, including text preprocessing and representation, extracting structure, topic modeling, text classification, and sentiment analysis. 
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Elon Musk Biopic
# MAGIC https://www.biography.com/business-figure/elon-musk

# COMMAND ----------

summarizer("""
Elon Musk is a South African-born American entrepreneur and businessman who founded X.com in 1999 (which later became PayPal), SpaceX in 2002 and Tesla Motors in 2003. Musk became a multimillionaire in his late 20s when he sold his start-up company, Zip2, to a division of Compaq Computers. 

Musk made headlines in May 2012, when SpaceX launched a rocket that would send the first commercial vehicle to the International Space Station. He bolstered his portfolio with the purchase of SolarCity in 2016 and cemented his standing as a leader of industry by taking on an advisory role in the early days of President Donald Trump's administration.

In January 2021, Musk reportedly surpassed Jeff Bezos as the wealthiest man in the world.

Early Life
Musk was born on June 28, 1971, in Pretoria, South Africa. As a child, Musk was so lost in his daydreams about inventions that his parents and doctors ordered a test to check his hearing.

At about the time of his parents’ divorce, when he was 10, Musk developed an interest in computers. He taught himself how to program, and when he was 12 he sold his first software: a game he created called Blastar.

In grade school, Musk was short, introverted and bookish. He was bullied until he was 15 and went through a growth spurt and learned how to defend himself with karate and wrestling.

Family
Musk’s mother, Maye Musk, is a Canadian model and the oldest woman to star in a Covergirl campaign. When Musk was growing up, she worked five jobs at one point to support her family.

Musk’s father, Errol Musk, is a wealthy South African engineer.

Musk spent his early childhood with his brother Kimbal and sister Tosca in South Africa. His parents divorced when he was 10.

Education
At age 17, in 1989, Musk moved to Canada to attend Queen’s University and avoid mandatory service in the South African military. Musk obtained his Canadian citizenship that year, in part because he felt it would be easier to obtain American citizenship via that path.

In 1992, Musk left Canada to study business and physics at the University of Pennsylvania. He graduated with an undergraduate degree in economics and stayed for a second bachelor’s degree in physics.

After leaving Penn, Musk headed to Stanford University in California to pursue a PhD in energy physics. However, his move was timed perfectly with the Internet boom, and he dropped out of Stanford after just two days to become a part of it, launching his first company, Zip2 Corporation in 1995. Musk became a U.S. citizen in 2002.
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ### facebook/bart-large-xsum
# MAGIC https://huggingface.co/facebook/bart-large-xsum

# COMMAND ----------

summarizer2 = pipeline('summarization', model='facebook/bart-large-xsum')

# COMMAND ----------

summarizer2("""
Elon Musk is a South African-born American entrepreneur and businessman who founded X.com in 1999 (which later became PayPal), SpaceX in 2002 and Tesla Motors in 2003. Musk became a multimillionaire in his late 20s when he sold his start-up company, Zip2, to a division of Compaq Computers. 

Musk made headlines in May 2012, when SpaceX launched a rocket that would send the first commercial vehicle to the International Space Station. He bolstered his portfolio with the purchase of SolarCity in 2016 and cemented his standing as a leader of industry by taking on an advisory role in the early days of President Donald Trump's administration.

In January 2021, Musk reportedly surpassed Jeff Bezos as the wealthiest man in the world.

Early Life
Musk was born on June 28, 1971, in Pretoria, South Africa. As a child, Musk was so lost in his daydreams about inventions that his parents and doctors ordered a test to check his hearing.

At about the time of his parents’ divorce, when he was 10, Musk developed an interest in computers. He taught himself how to program, and when he was 12 he sold his first software: a game he created called Blastar.

In grade school, Musk was short, introverted and bookish. He was bullied until he was 15 and went through a growth spurt and learned how to defend himself with karate and wrestling.

Family
Musk’s mother, Maye Musk, is a Canadian model and the oldest woman to star in a Covergirl campaign. When Musk was growing up, she worked five jobs at one point to support her family.

Musk’s father, Errol Musk, is a wealthy South African engineer.

Musk spent his early childhood with his brother Kimbal and sister Tosca in South Africa. His parents divorced when he was 10.

Education
At age 17, in 1989, Musk moved to Canada to attend Queen’s University and avoid mandatory service in the South African military. Musk obtained his Canadian citizenship that year, in part because he felt it would be easier to obtain American citizenship via that path.

In 1992, Musk left Canada to study business and physics at the University of Pennsylvania. He graduated with an undergraduate degree in economics and stayed for a second bachelor’s degree in physics.

After leaving Penn, Musk headed to Stanford University in California to pursue a PhD in energy physics. However, his move was timed perfectly with the Internet boom, and he dropped out of Stanford after just two days to become a part of it, launching his first company, Zip2 Corporation in 1995. Musk became a U.S. citizen in 2002.
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Declaration of Independence of Lower Canada
# MAGIC
# MAGIC https://en.wikisource.org/wiki/Declaration_of_Independence_of_Lower_Canada

# COMMAND ----------

summarizer("""
WHEREAS, the solemn covenant made with the people of Lower Canada, and recorded in the Statute Book of the United Kingdom of Great Britain and Ireland, as the thirty-first chapter of the Act passed in the thirty-first year of the Reign of King George III hath been continually violated by the British Government, and our rights usurped. And, whereas our humble petitions, addresses, protests, and remonstrances against this injurious and unconstitutional interference have been made in vain. That the British Government hath disposed of our revenue without the constitutional consent of the local Legislature — pillaged our treasury — arrested great numbers of our citizens, and committed them to prison — distributed through the country a mercenary army, whose presence is accompanied by consternation and alarm — whose track is red with the blood of our people — who have laid our villages in ashes — profaned our temples — and spread terror and waste through the land. And, whereas we can no longer suffer the repeated violations of our dearest rights, and patiently support the multiplied outrages and cruelties of the Government of Lower Canada, we, in the name of the people of Lower Canada, acknowledging the decrees of a Divine Providence, which permits us to put down a Government, which hath abused the object and intention for which it was created, and to make choice of that form of Government which shall re-establish the empire of justice — assure domestic tranquillity — provide for common defence — promote general good, and secure to us and our posterity the advantages of civil and religious liberty,

""")

# COMMAND ----------

summarizer2("""
WHEREAS, the solemn covenant made with the people of Lower Canada, and recorded in the Statute Book of the United Kingdom of Great Britain and Ireland, as the thirty-first chapter of the Act passed in the thirty-first year of the Reign of King George III hath been continually violated by the British Government, and our rights usurped. And, whereas our humble petitions, addresses, protests, and remonstrances against this injurious and unconstitutional interference have been made in vain. That the British Government hath disposed of our revenue without the constitutional consent of the local Legislature — pillaged our treasury — arrested great numbers of our citizens, and committed them to prison — distributed through the country a mercenary army, whose presence is accompanied by consternation and alarm — whose track is red with the blood of our people — who have laid our villages in ashes — profaned our temples — and spread terror and waste through the land. And, whereas we can no longer suffer the repeated violations of our dearest rights, and patiently support the multiplied outrages and cruelties of the Government of Lower Canada, we, in the name of the people of Lower Canada, acknowledging the decrees of a Divine Providence, which permits us to put down a Government, which hath abused the object and intention for which it was created, and to make choice of that form of Government which shall re-establish the empire of justice — assure domestic tranquillity — provide for common defence — promote general good, and secure to us and our posterity the advantages of civil and religious liberty,

""")