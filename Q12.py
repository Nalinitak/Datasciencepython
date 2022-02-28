#!/usr/bin/env python
# coding: utf-8

# In[2]:


#)  Below are the scores obtained by a student in tests 
#34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56
#1)	Find mean, median, variance, standard deviation.
#2)	What can we say about the student marks


# In[6]:


import pandas as pd


# In[7]:


import numpy as np


# In[8]:


import matplotlib.pyplot as plt


# In[9]:


data=pd.Series([34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56])


# In[24]:


data.mean()


# In[25]:


data.describe()


# In[26]:


data.hist()


# In[38]:


fig = plt.figure
plt.plot(data)
#plt.xlim(0, 18)
#plt.ylim(0,60)
plt.xlabel('student')
plt.ylabel('mark')


# In[39]:


data.median()


# In[41]:


data.var()# variance 


# In[1]:


#other way


# In[2]:


df=[34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56]


# In[3]:


df


# In[10]:


Df=pd.DataFrame(df)


# In[12]:


Df.mean()


# In[ ]:




