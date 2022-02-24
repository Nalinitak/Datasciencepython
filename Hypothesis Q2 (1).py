#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# anova anlyis of variances
#Hn=threiis no significances diffrences in lab1,2,3,4 ie u1=u2=u3=u4
#Ha= significant diffrences u1!=u2!=u3!=u4


# In[3]:


import pandas as pd
import numpy as np
import scipy 
from scipy import stats


# In[4]:


data=pd.read_csv("LabTAT (1).csv")


# In[5]:


data


# In[6]:


stats.f_oneway(data.iloc[:,0],data.iloc[:,1],data.iloc[:,2],data.iloc[:,3])


# In[10]:


print("As p value is less than signifcant leavel alfa null hypotis is False "
    "  some  difference in average TAT among the different laboratories at 5 significance lev")


# In[ ]:




