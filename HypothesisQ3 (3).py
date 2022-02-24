#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Q3Sales of products in four different regions is tabulated for males and females. Find if male-female buyer rations are similar across regions.
#Hn=all proption equal
#Ha= not all prportion equl


# In[2]:


import pandas as pd 


# In[3]:


get_ipython().system('pip install pandas-profiling==2.7.1')
get_ipython().system('pip install sweetviz')


# In[4]:



import pandas as pd 
import numpy as np
import scipy
from scipy import stats
from scipy.stats import chisquare
from scipy.stats import chi2_contingency


# In[5]:


data=pd.read_csv("BuyerRatio.csv")


# In[6]:


m=np.array([50,142,131,70])
f=np.array([435,1523,1356,750])


# In[7]:


observed=np.array([[50,142,131,70],[435,1523,1356,750]])


# In[8]:


chi2_contingency(observed)


# In[9]:


stats,p,dof,expected=chi2_contingency(observed)


# In[10]:


p


# In[11]:


if p>0.05:
    print("p high null flay we will accept null hypthis male-female buyer rations are similar across regions")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




