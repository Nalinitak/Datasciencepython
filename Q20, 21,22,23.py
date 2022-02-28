#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Q20Calculate probability from the given dataset for the below cases

#Data _set: Cars.csv
#Calculate the probability of MPG  of Cars for the below cases.
# MPG <- Cars$MPG
#a.	P(MPG>38)
#b.	P(MPG<40)
#c. P (20<MPG<50)
#ans useing 


# In[1]:


import pandas as pd


# In[2]:


cars=pd.read_csv('Cars.csv')


# In[3]:


cars.head()


# In[4]:


cars.MPG


# In[5]:


import numpy as np


# In[6]:


from scipy import stats
from scipy.stats import norm
from scipy import stats
from scipy.stats import norm


# In[7]:


np.round(1-stats.norm.cdf(38,loc = cars.MPG.mean(), scale = cars.MPG.std()),3)


# In[38]:


np.round(stats.norm.cdf(40,loc = cars.MPG.mean(), scale = cars.MPG.std()),3)


# In[51]:


a=np.round(stats.norm.cdf(50,loc = cars.MPG.mean(), scale = cars.MPG.std()),3)


# In[50]:


b=np.round(stats.norm.cdf(20,loc =cars.MPG.mean(),scale =cars.MPG.std()),3)


# In[52]:


print ('c.	P (20<MPG<50) ',a-b)


# In[ ]:


#Q21Check whether the data follows normal distribution
#a)	Check whether the MPG of Cars follows Normal Distribution 
#ANS----- NO not follws normal distribution its right-skewed  


# In[11]:


import matplotlib.pyplot as plt


# In[12]:


plt.hist(cars['MPG']) 


# In[13]:


#Q21 b)	Check Whether the Adipose Tissue (AT) and Waist Circumference(Waist)  from wc-at data set  follows Normal Distribution 
       #Dataset: wc-at.csv
#Ans_---At is right skewes not follow normal distributiom
#Waist mean and meian is near about equal but mode is not near to that so its not follows normal distribution


# In[14]:


df=pd.read_csv('wc-at.csv')


# In[15]:


df


# In[17]:


plt.hist(df['AT'])


# In[18]:


plt.hist(df['Waist'])


# In[63]:


df.describe()


# In[64]:


df.median()


# In[65]:


df.mean()


# In[66]:


df.mode()


# In[23]:


from scipy import stats
from scipy.stats import norm


# In[22]:


print('Z scores at 94% confidence interval is', np.round(stats.norm.ppf(.97), 2))
print('Z scores at 60% confidence interval is', np.round(stats.norm.ppf(.80), 2))
print('z score at 90% ci is',np.round(stats.norm.ppf(0.95),2))


# In[24]:


stats.norm.ppf(0.95)


# In[25]:


norm.ppf(0.95, loc=0, scale=1)


# In[27]:


print(' t scores at 95% confidence interval is', np.round(stats.t.ppf(0.975, df = 24), 2))
print(' t scores at 96% confidence interval is', np.round(stats.t.ppf(0.98, df = 24), 2))
print(' t scores at 99% confidence interval is', np.round(stats.t.ppf(0.995, df = 24), 2))


# In[30]:


print('t score at 95ci',np.round(stats.t.ppf(0.97,df=24),2))


# In[30]:


stats.t.cdf(-0.471,df=17)


# In[14]:


stats.norm.ppf(0.99)


# In[15]:


stats.norm.ppf(0.96)


# In[17]:


#Q24#A Government  company claims that an average light bulb lasts 270 days. A researcher randomly selects 18 bulbs for testing. The sampled bulbs last an average of 260 days, with a standard deviation of 90 days. If the CEO's claim were true, what is the probability that 18 randomly selected bulbs would have an average life of no more than 260 days


# In[ ]:


#t value=( x-x(bar)) / (s/(sqirroot n)
#(260-270)/(90/sqirtoot 18)=-0.47


# In[18]:


stats.t.cdf(-0.471,df=17)


# In[ ]:


#the probability that 18 randomly selected bulbs would have an average life of no more than 260 days is 0.3218140331685075


# In[ ]:




