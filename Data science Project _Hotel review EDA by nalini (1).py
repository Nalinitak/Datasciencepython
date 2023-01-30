#!/usr/bin/env python
# coding: utf-8

# #Importing libraries

# In[22]:



#Importing the necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
import nltk
from nltk.corpus import stopwords
from nltk import ngrams
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import wordcloud
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

import warnings
warnings.filterwarnings('ignore')

get_ipython().system('pip install textblob')
get_ipython().system('pip install wordcloud')


# #Importing dataset

# In[23]:


df=pd.read_excel("D:\\Nalini tak\\Data SciencE\\Poject\\Hotelreview\\hotel_reviews.xlsx")


# In[7]:


df.head()


# In[8]:


df.describe()


# In[9]:


df.shape


# In[10]:


df.columns


# In[11]:


df.nunique()          # Unique Values


# In[12]:


##Count of null values#

count=df.isnull().sum().sort_values(ascending=True)
percentage=((df.isnull().sum()/len(df)*100))
missing_data=pd.concat([count,percentage],axis=1,keys=["Count","Percentage"])


# In[13]:


#Rating Count
sns.set_style("darkgrid")
sns.countplot(x="Rating",hue="Rating",data=df)


# In[27]:


#Importing the libraries
from collections import Counter
import nltk
import string
nltk.download('stopwords')


# #Importing libraries for text preprocessing

# In[43]:


import spacy
from spacy.lang.en.stop_words import STOP_WORDS
STOP_WORDS |= {"nt","hotel","room","good","best","worst"}
print(STOP_WORDS)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




