#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Q1 A F&B manager wants to determine whether there is any significant difference in the diameter of the cutlet between two units. A randomly selected sample of cutlets was collected from both units and measured? Analyze the data and draw inferences at 5% significance level. Please state the assumptions and tests that you carried out to check validity of the assumptions.

# proprtion test will use hear wher threis significnt diffrence in diameter of cutlets of unit A and Unit b
# hypothiess defination -----null hypothesis HN=no significant diffrence in A and B
#                            altrnative hypothesis Ha = signifcant diffrence in Unit A and B


# In[2]:


# one way 


# In[3]:


import pandas as pd
import numpy as np
import scipy
from scipy import stats


# In[4]:


data=pd.read_csv("Cutlets (1).csv")


# In[5]:


data.head()


# In[6]:


stats.f_oneway(data.iloc[:,0],data.iloc[:,1])


# In[26]:


u1=pd.Series(data.iloc[:,0])
u2=pd.Series(data.iloc[:,1])


# In[27]:


u1


# In[28]:


u2


# In[31]:


p_value=stats.ttest_ind( u1,u2)
p_value


# In[32]:


p_value[1]


# In[34]:


if p_value[1]>0.05:
    print(" P value is larger than threshold value so we are accepting Null hypothesis .\n Thre is no significant dfifrence between diameter ofncutleats between Unit A And Unit B")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




