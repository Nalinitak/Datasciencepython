#!/usr/bin/env python
# coding: utf-8

# In[1]:


#TeleCall uses 4 centers around the globe to process customer order forms. They audit a certain %  of the customer order forms. Any error in order form renders it defective and has to be reworked before processing.  The manager wants to check whether the defective %  varies by centre. Please analyze the data at 5% significance level and help the manager draw appropriate inferences


# In[2]:


#Ho= % of defective of all countries are equ
#al i.e % Phillipness=%Indonesia=%Malta=%India
#Ha =at lest  one defective% is  not equal 


# In[3]:


import  pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2_contingency


# In[4]:


data=pd.read_csv("Costomer+OrderForm (1).csv")


# In[5]:


data


# In[6]:


data.Phillippines.value_counts()


# In[7]:


data.Malta.value_counts()


# In[8]:


data.Indonesia.value_counts()


# In[9]:


data.India.value_counts()


# In[10]:


observed=np.array([[271,269,267,280],[29,31,33,20]])


# In[11]:


observed


# In[12]:


chi2_contingency(observed)


# In[13]:


stats,p,dof,expected=chi2_contingency(observed)


# In[14]:


p


# In[15]:


if p>0.05:
    print("p high Ho fly hence accept null hypothesis i.e percentage defective  in all countries are equalinferences")


# In[ ]:




