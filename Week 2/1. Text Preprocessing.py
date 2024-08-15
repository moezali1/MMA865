# Databricks notebook source
# MAGIC %md
# MAGIC # Natural Language Toolkit
# MAGIC
# MAGIC NLTK is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries, and an active discussion forum.

# COMMAND ----------

import nltk

# COMMAND ----------

# this command will download all the bells & whistles of nltk library (it will take some time)
nltk.download('all')

# COMMAND ----------

text = """
MMA 865: Big Data Analytics
MMA 2022S Syllabus
Updated Dec 6, 2021
COURSE DESCRIPTION
There are two main learning objectives of this course:
1.	Big Data 
In this course, we discuss the relatively new discipline known as big data. The term big data is overloaded and simultaneously refers to the description of the data itself, the IT techniques necessary for handling massive data sets, the analytics and AI techniques that are possible with large data sets, and the business decisions/problems that we can address by leveraging the data. We will investigate all of these areas. We will also survey key IT technologies, such as enterprise data warehouses, enterprise data lakes, Hadoop, Spark, NoSQL databases, the cloud, and more. 
2.	Natural Language Processing
Natural Language Processing is one of the six AI disciplines. In this part of the course, we will discuss the major practice areas of NLP such as Document Classification, Sentiment Analysis, Language Generation, Chatbots, and Machine Translation. We will also discuss the relatively new concept of Transfer Learning in NLP and how it has derived the massive progress in the area of language modeling (BERT, GPT3, BART, etc.). We will also discuss EDA and preprocessing on text data in much detail as it forms the foundation of NLP. You will learn tools and techniques for handling text data, including text preprocessing and representation, extracting structure, topic modeling, text classification, and sentiment analysis. 
INSTRUCTOR
Moez Ali
•	Email: moez.ali@queensu.ca
•	Website: https://www.moez.ai
•	Blog: https://www.moez.ai/blog
•	Google Scholar: https://scholar.google.ca/scholar?hl=en&as_sdt=0%2C5&q=pycaret
•	LinkedIn: https://www.linkedin.com/in/profile-moez/
•	Twitter: https://twitter.com/moezpycaretorg1
•	Medium: https://moez-62905.medium.com/
•	GitHub: https://github.com/moezali1
TEACHING ASSISTANT
Rishi Jotsinghani, MMA, rishi.jotsinghani@queensu.ca
OFFICE HOURS
Office hours will be every Tuesday 5:30 – 6:30 on Zoom with TA Rishi Jotsinghani. 
COURSE RESOURCES
Textbooks
We will draw some of the recommended reading material from the following books: 
•	Steven Bird; Ewan Klein; Edward Loper. “Natural Language Processing with Python.” O’Reillly Media, Inc. ISBN-13: 978-0596550967
•	Bill Chambers, Matie Zaharia. "Spark: The Definitive Guide: Big Data Processing Made Simple.", O'Reilly Media, Inc. ISBN-13: 978-1491912294
Course Portal
The course portal contains supplementary reading for each session. It is recommended that you visit the course portal and review the material before each session.
EVALUATION
Item	Value	Due*
Team Project	50%	February 16/17, 2022
Individual Assignment	50%	March 3, 2022
*by 11:59pm Eastern Time.
Team Project
Please see the separate file: Project Brief – Amazon.docx.
Individual Assignment
Please see Individual Assignment under “Assignment” section on the course portal.

COURSE SCHEDULE
Session 1: Introduction to Big Data
•	Course introduction
•	Big data overview and landscape
•	Introduction to Databricks
•	Team project overview and discussion
•	Team Breakout: Machine Learning on Databricks
Session 2: Intro to Natural Language Processing
•	Introduction to NLP
•	Text Preprocessing
o	Tokenization
o	Stop Words
o	Stemming and Lemmatization
o	BOW and Embeddings
•	Text EDA
•	Team Breakout: Text Preprocessing in Python
Session 3: Practice Areas of NLP
•	Information Extraction
•	Document Classification
•	Chatbots
•	Sentiment Analysis
•	Machine Translation
•	Language Generation
•	Team Breakout: Predict Document Classification
Session 4: NLP: Topic Modeling and Transfer Learning
•	Topic Modeling
•	Transfer Learning
•	Team Breakout: Topic Model on Financial Loans

 
Session 5: Hadoop, Spark, and Cloud
•	Challenges of Scale
•	Hadoop (HDFS, MapReduce, YARN)
•	Spark
•	Cloud
•	Team Breakout: Machine Learning using MLLib
Session 6: Data Stores
•	Introduction to Data Stores
o	Relation 
o	Non-Relational
•	Use-cases and practical applications
•	Non-Relational Data Stores
o	Key-Value
o	Graph Database
o	Document
o	Columnar
•	Team Breakout: Prepare for final presentations 
Session 7: Team Presentations
•	Team Project presentations
COURSE POLICIES
Late Work
There will be a 10% penalty per day for late work. 
Extensions
Deadline extensions will only be given for extenuating circumstances. Examples of circumstances that are NOT extenuating include:
•	You have a deadline at work
•	You are really busy at work
•	You have family/friends are in town (unexpectedly)
•	You had a work trip
•	You had a personal trip
•	You are generally very busy in life
Rounding
I round marks to the nearest whole number. I.e., 89.49999 gets rounded to 89; 89.50000 gets rounded to 90.
Appealing Marks
You may appeal a mark if you believe an error has occurred. To appeal, please write a brief memo outlining what the error was and how you recommend the assignment/project should be re-marked. 
Please wait until at least ONE WEEK AFTER you receive your mark to submit your appeal.
You must submit your appeal within TWO WEEKS of receiving your initial mark.
Please note that if I re-mark your assignment, I reserve the right to decrease your mark if I feel that the initial mark was too generous.
ACADEMIC INTEGRITY
Definition of Academic Integrity
Any behaviour that compromises the fundamental scholarly values of honesty, trust, fairness, respect and responsibility in the academic setting is considered a departure from academic integrity and is subject to remedies or sanctions as established by Smith School of Business and Queen's University. 
These behaviours may include plagiarism, use of unauthorized materials, unauthorized collaboration, facilitation, forgery and falsification among other actions. It is every student's responsibility to become familiar with the Smith School of Business policy regarding academic integrity and ensure that his or her actions do not depart, intentionally or unintentionally, from the standards described at: http://business.queensu.ca/about/academic_integrity/index.php. 
Helpful FAQ's about academic integrity are at: http://business.queensu.ca/about/academic_integrity/faq.php  
To assist you in identifying the boundary between acceptable collaboration and a departure from academic integrity in this specific course, I provide the following guidelines for individual and group work. If my expectations still are not clear to you, ask me! The onus is on you to ensure that your actions do not violate standards of academic integrity.
Individual Work
Assignments and examinations identified as individual in nature must be the result of your individual effort. Outside sources must be properly cited and referenced in assignments; be careful to cite all sources, not only of direct quotations but also of ideas. Ideas, information and quotations taken from the internet must also be properly cited and referenced. Help for citing sources is available through the Queen's University library: http://library.queensu.ca/help-services/citing-sources.
Group Work
I will clearly indicate when groups may consult with one another or with other experts or resources. Otherwise, in a group assignment, the group members will work together to develop an original, consultative response to the assigned topic. Group members must not look at, access or discuss any aspect of any other group's solution (including a group from a previous year), nor allow anyone outside of the group to look at any aspect of the group's solution. Likewise, you are prohibited from utilizing the internet or any other means to access others' solutions to, or discussions of, the assigned material. The names of each group member must appear on the submitted assignment, and no one other than the people whose names appear on the assignment may have contributed in any way to the submitted solution. In short, the group assignments must be the work of your group, and your group only. All group members are responsible for ensuring the academic integrity of the work that the group submits. 
Consequences of a Breach of Academic Integrity
Any student who is found to have departed from academic integrity may face a range of sanctions, from a warning, to a grade of zero on the assignment, to a recommendation to Queen's Senate that the student be required to withdraw from the University for a period of time, or even that a degree be rescinded. As an instructor, I have a responsibility to investigate any suspected breach of academic integrity. If I determine that a departure from Academic Integrity has occurred, I am required to report the departure to the Dean's office, where a record of the departure will be filed and sent to the program office to be recorded in the student file.
Turnitin.com
Turnitin.com (http://turnitin.com) is a plagiarism detection tool used by many educational institutions, including Smith School of Business. Turnitin is a leader in the area of originality checking and plagiarism prevention. Its purpose is to verify the originality of a deliverable (i.e. assignment) and, in doing so, it validates the effort each student puts into a course deliverable. I may ask you to submit assignments through Turnitin, which is easily done through the course portal. 
"""

# COMMAND ----------

print(text)

# COMMAND ----------

# MAGIC %md
# MAGIC # Sentence Tokenization
# MAGIC Sentence tokenization is the process of splitting text into individual sentences

# COMMAND ----------

type(text)

# COMMAND ----------

# sentence tokenize function
sentence = nltk.sent_tokenize(text)

# COMMAND ----------

type(sentence)

# COMMAND ----------

len(sentence)

# COMMAND ----------

print(sentence[0])

# COMMAND ----------

print(sentence[1])

# COMMAND ----------

print(sentence[2])

# COMMAND ----------

# MAGIC %md
# MAGIC # Word Tokenization
# MAGIC Word tokenization is the process of splitting a large sample of text into words. This is a requirement in natural language processing tasks where each word needs to be captured and subjected to further analysis like classifying and counting them for a particular sentiment etc.

# COMMAND ----------

# word_tokenize function
words = nltk.word_tokenize(text)

# COMMAND ----------

type(words)

# COMMAND ----------

len(words)

# COMMAND ----------

print(words[0:10])

# COMMAND ----------

# MAGIC %md
# MAGIC # Visualize Token Frequency

# COMMAND ----------

import pandas as pd
df = pd.DataFrame(words, columns=['token'])
df.head()

# COMMAND ----------

df['token'].value_counts()

# COMMAND ----------

df['token'].value_counts()[:10].plot.bar()

# COMMAND ----------

# alternate (only work in databricks)
pd.DataFrame(df.value_counts()).reset_index().display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Stop Word Removal
# MAGIC
# MAGIC In computing, stop words are words which are filtered out before or after processing of natural language data (text). Though "stop words" usually refers to the most common words in a language, there is no single universal list of stop words used by all natural language processing tools, and indeed not all tools even use such a list. Some tools specifically avoid removing these stop words to support phrase search.
# MAGIC
# MAGIC Any group of words can be chosen as the stop words for a given purpose. For some search engines, these are some of the most common, short function words, such as the, is, at, which, and on. In this case, stop words can cause problems when searching for phrases that include them, particularly in names such as "The Who", "The The", or "Take That". Other search engines remove some of the most common words—including lexical words, such as "want"—from a query in order to improve performance.

# COMMAND ----------

# import stopwords
from nltk.corpus import stopwords

# COMMAND ----------

# check the list of stopwords in english
stopwords.words('english')

# COMMAND ----------

# check the list of stopwords in french
stopwords.words('french')

# COMMAND ----------

# check the list of stopwords in spanish
stopwords.words('spanish')[:10]

# COMMAND ----------

stopwords.words('arabic')

# COMMAND ----------

# check the length of words
len(words)

# COMMAND ----------

words[:20]

# COMMAND ----------

# simple list comprehension to exclude stop words
words_processed = [word.lower() for word in words if word.lower() not in stopwords.words('english')]

# COMMAND ----------

# check the length of words
len(words_processed)

# COMMAND ----------

(len(words_processed) - len(words)) / len(words) 

# COMMAND ----------

import pandas as pd
df = pd.DataFrame(words_processed, columns=['token'])
df

# COMMAND ----------

df['token'].value_counts()[:10].plot.bar()

# COMMAND ----------

# alternate (only work in databricks)
pd.DataFrame(df.value_counts()).reset_index().display()

# COMMAND ----------

processed_sentence = ' '.join(words_processed)

# COMMAND ----------

# MAGIC %md
# MAGIC # Remove Punctuations

# COMMAND ----------

import re
processed_sentence_regex = re.sub(r'[^\w\s]',' ',processed_sentence)

# COMMAND ----------

processed_sentence

# COMMAND ----------

processed_sentence_regex

# COMMAND ----------

word_tokenized_after_regex = nltk.word_tokenize(processed_sentence_regex)

# COMMAND ----------

len(word_tokenized_after_regex)

# COMMAND ----------

import pandas as pd
df = pd.DataFrame(word_tokenized_after_regex, columns=['token'])
df

# COMMAND ----------

df['token'].value_counts()[:10].plot.bar()

# COMMAND ----------

# alternate (only work in databricks)
pd.DataFrame(df.value_counts()).reset_index().display()

# COMMAND ----------

# MAGIC %md
# MAGIC #Stemming
# MAGIC
# MAGIC In linguistic morphology and information retrieval, stemming is the process of reducing inflected (or sometimes derived) words to their word stem, base or root form—generally a written word form. The stem need not be identical to the morphological root of the word; it is usually sufficient that related words map to the same stem, even if this stem is not in itself a valid root. Algorithms for stemming have been studied in computer science since the 1960s. Many search engines treat words with the same stem as synonyms as a kind of query expansion, a process called conflation.

# COMMAND ----------

word_tokenized_after_regex

# COMMAND ----------

len(word_tokenized_after_regex)

# COMMAND ----------

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
words_stemmed = [stemmer.stem(word) for word in word_tokenized_after_regex]

# COMMAND ----------

words_stemmed

# COMMAND ----------

# MAGIC %md
# MAGIC # Lemmatization
# MAGIC
# MAGIC Lemmatisation (or lemmatization) in linguistics is the process of grouping together the inflected forms of a word so they can be analysed as a single item, identified by the word's lemma, or dictionary form.
# MAGIC
# MAGIC In computational linguistics, lemmatisation is the algorithmic process of determining the lemma of a word based on its intended meaning. Unlike stemming, lemmatisation depends on correctly identifying the intended part of speech and meaning of a word in a sentence, as well as within the larger context surrounding that sentence, such as neighboring sentences or even an entire document. As a result, developing efficient lemmatisation algorithms is an open area of research.
# MAGIC
# MAGIC ### Difference between Stemming and Lemmatization
# MAGIC
# MAGIC The goal of both stemming and lemmatization is to reduce inflectional forms and sometimes derivationally related forms of a word to a common base form.
# MAGIC
# MAGIC However, the two words differ in their flavor. Stemming usually refers to a crude heuristic process that chops off the ends of words in the hope of achieving this goal correctly most of the time, and often includes the removal of derivational affixes. Lemmatization usually refers to doing things properly with the use of a vocabulary and morphological analysis of words, normally aiming to remove inflectional endings only and to return the base or dictionary form of a word, which is known as the lemma. (https://stackoverflow.com/questions/1787110/what-is-the-difference-between-lemmatization-vs-stemming)

# COMMAND ----------

word_tokenized_after_regex

# COMMAND ----------

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
words_lemmatized = [lemmatizer.lemmatize(word) for word in word_tokenized_after_regex]

# COMMAND ----------

words_lemmatized

# COMMAND ----------

import pandas as pd
df = pd.DataFrame(words_lemmatized, columns=['token'])
df

# COMMAND ----------

df['token'].value_counts()[:10].plot.bar()

# COMMAND ----------

# alternate (only work in databricks)
pd.DataFrame(df.value_counts()).reset_index().display()