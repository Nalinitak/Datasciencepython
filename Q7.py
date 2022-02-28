#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


df=pd.read_csv('Q7.csv')


# In[4]:


df.head()


# In[19]:


df.describe()


# In[20]:


df.mean()


# In[34]:


df.median()


# In[21]:


df.mode


# In[8]:


df.median()


# In[36]:


df.std()


# In[5]:


df.var()


# In[14]:


df.hist()


# In[7]:


df.Weigh.skew()


# In[8]:


df.Points.skew()


# In[9]:


df.Score.skew()


# In[10]:


df.skew()


# In[15]:


df.max()


# In[12]:


(108+110+123+134+135+145+167+187+199)/9


# In[16]:


df.min()


# In[17]:


df_points_range=df.Points.max()-df.Points.min()


# In[18]:


print ('points range',df_points_range)


# In[20]:


df_score_range=df.Score.max()-df.Score.min()


# In[21]:


print('score range is',df_score_range)


# In[23]:


df_weigh_range=df.Weigh.max()-df.Weigh.min()


# In[25]:


print('weigh range is',df_weigh_range)


# In[24]:


import matplotlib.pyplot as plt


# In[32]:


plt.hist(df['Score']) 
plt.title('Score')


# In[31]:


plt.boxplot(df['Score'])


# In[38]:


plt.hist(df['Points'])
plt.title('Points')


# In[39]:


plt.boxplot(df['Points'])


# In[40]:


plt.hist(df['Weigh'])
plt.title('Weigh')


# In[41]:


plt.boxplot(df['Weigh'])


# In[ ]:




