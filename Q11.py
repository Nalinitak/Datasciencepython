#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Suppose we want to estimate the average weight of an adult male in    
#Mexico. We draw a random sample of 2,000 men from a population of 3,000,000 men and weigh them. 
#We find that the average person in our sample weighs 200 pounds, and the standard deviation of the sample is 30 pounds. 
#Calculate 94%,98%,96% confidence interval?


# In[1]:


import pandas as pd 


# In[2]:


import numpy as np
from scipy import stats


# In[6]:


stats.norm.interval(0.97,200,30)


# In[15]:


print('confidences interval at 94% is',ci1)


# In[17]:


ci2=stats.norm.interval(0.98,200,30)


# In[18]:


print('confidences interval at 98% is',ci2)


# In[19]:


ci3=stats.norm.interval(0.96,200,30)


# In[20]:


print('confidences interval at 96% is',ci3)


# In[ ]:




