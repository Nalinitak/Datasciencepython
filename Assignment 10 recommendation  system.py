#!/usr/bin/env python
# coding: utf-8

# In[82]:



import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
import warnings


# In[41]:


get_ipython().system('pip install mlxtend')


# In[45]:


books_df = pd.read_csv("Bookrecom.csv",encoding='latin-1')


# In[46]:


books_df


# In[53]:


# Renaming the columns

books_data = books_df.rename({'User.ID':'userid','Book.Title':'booktitle','Book.Rating':'bookrating'},axis = 1)


# In[55]:



books_data


# In[59]:


# Dropping the column

books1 = books_data.drop(['Unnamed: 0'], axis = 1)


# In[61]:



books1.head()


# In[63]:


books1.info()


# In[64]:


books1.isna()


# In[65]:


# Checking the number of userid

len(books1['userid'].unique())


# In[66]:


# saving the unique userids

array_user = books1['userid'].unique()


# In[67]:



array_user


# In[69]:


books_data1 = books1.pivot_table(index = 'userid',
                        columns = 'booktitle',
                        values = 'bookrating').reset_index(drop = True)


# In[70]:


books_data1


# In[71]:


books_data1.index = books1.userid.unique()


# In[72]:


books_data1.index


# In[79]:


# Filling the NaN values with 0

books_data1.fillna(0, inplace = True)


# In[80]:


books_data1.head()


# In[75]:


#Calculating Cosine Similarity between Users
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine,correlation


# In[83]:


warnings.filterwarnings("ignore")

user = 1 - pairwise_distances(books_data1.values, metric = 'cosine')


# In[84]:


user


# In[85]:


#Store the results in a dataframe
user_sim_df = pd.DataFrame(user)


# In[86]:


user_sim_df 


# In[87]:


user_sim_df.iloc[0:5, 0:5]


# In[20]:


np.fill_diagonal(user_sim, 0)
user_sim_df.iloc[0:5, 0:5]


# In[90]:


user_sim_df.index = books1.userid.unique()
user_sim_df.columns = books1.userid.unique()


# In[91]:


user_sim_df.iloc[0:5,0:5]


# In[92]:



np.fill_diagonal(user,0)


# In[94]:


user_sim_df.idxmax(axis = 1)


# In[96]:


# printing the only values wheree userid is 162107 or 276726

books1[(books1['userid'] == 162107) | (books1['userid'] == 276726)]


# In[98]:


# Merging the data
user_1 = books1[books1['userid'] == 276729]
user_2 = books1[books1['userid'] == 276726]
pd.merge(user_1,user_2, on = 'booktitle', how = 'outer')


# In[ ]:




