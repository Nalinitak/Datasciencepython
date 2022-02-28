#!/usr/bin/env python
# coding: utf-8

# In[1]:


#) 9 a )Calculate Skewness, Kurtosis & draw inferences on the following data
     # Cars speed and distance 


# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import scipy as sc


# In[4]:


import matplotlib.pyplot as plt


# In[5]:



import seaborn as sns


# In[6]:


df=pd.read_csv('Q9_a.csv')


# In[7]:


df.head()


# In[8]:


df.hist()


# In[9]:


df.skew ()


# In[10]:


df.kurt()


# In[11]:


np.round(df.speed.skew())#round up the value


# In[12]:


np.round(df.dist.skew())


# In[13]:


df.speed.skew()


# In[14]:


np.round(df.speed.kurt())


# In[18]:


np.round(df.dist.kurt())


# In[ ]:


#9b)Calculate Skewness, Kurtosis & draw inferences on the following data


# In[15]:


df1=pd.read_csv('Q9_b.csv')


# In[16]:


df1.head()


# In[17]:


df1.SP.skew()


# In[18]:


df1.WT.skew()


# In[19]:


df1.kurt()


# In[20]:


df1.skew()


# In[1]:


from scipy.stats import skew


# In[30]:


plt.figure()
sns.displot(df1.SP)


# In[31]:


plt.figure()
sns.displot(df1.WT)


# In[ ]:




