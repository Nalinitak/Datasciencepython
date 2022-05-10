#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder


# In[2]:


# Loading dataset
data = pd.read_csv('my_movies.csv')


# In[3]:


data


# In[4]:


data.info()


# In[5]:


data.sample(10)


# In[6]:


data.isna().sum()


# In[8]:


data.columns


# In[9]:



# data processing
data=data.drop(['V1', 'V2', 'V3', 'V4', 'V5'], axis = 1)
data.head()


# In[10]:


movie_count = []
col_names = data.columns
for col_name in col_names:
    movie_count.append(data[col_name].value_counts()[1])
    


# In[11]:


movie_count 


# In[12]:


plt.figure(figsize=(10, 10), dpi=80)    
plt.bar(col_names, movie_count)


# In[13]:


# Apriori Algorithm for min_support = 0.1
frequent_itemsets1 = apriori(data, min_support=0.1, use_colnames=True)
frequent_itemsets1 


# In[14]:


frequent_itemsets1 = apriori(data, min_support = 0.1, use_colnames=True)
frequent_itemsets1['length'] = frequent_itemsets1['itemsets'].apply(lambda x: len(x))
frequent_itemsets1


# In[15]:


# Rules when min_support = 0.1 and min_threshold for lift is 0.5
rules1 = association_rules(frequent_itemsets1, metric="lift", min_threshold=1)
rules1


# In[16]:


rules1 = association_rules(frequent_itemsets1, metric ="lift", min_threshold = 1)
rules1 = rules1.sort_values(['confidence', 'lift'], ascending =[False, False])
rules1


# In[17]:


plt.scatter(rules1['support'], rules1['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


# In[18]:


# Rules when min_support = 0.1 and min_threshold for confidence is 0.5
rules2 = association_rules(frequent_itemsets1, metric="confidence", min_threshold=0.5)
rules2


# In[19]:



rules2 = rules2.sort_values(['confidence', 'lift'], ascending =[False, False])
rules2


# In[21]:


plt.scatter(rules2['support'], rules2['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


# In[22]:


# Apriori Algorithm for min_support = 0.2
frequent_itemsets2 = apriori(data, min_support=0.2, use_colnames=True)
frequent_itemsets2


# In[23]:


frequent_itemsets2 = apriori(data, min_support = 0.2, use_colnames=True)
frequent_itemsets2['length'] = frequent_itemsets2['itemsets'].apply(lambda x: len(x))
frequent_itemsets2


# In[24]:


# Rules when min_support = 0.2 and min_threshold for lift is 0.5
rules3 = association_rules(frequent_itemsets2, metric="lift", min_threshold=0.1)
rules3


# In[25]:


plt.scatter(rules3['support'], rules3['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


# In[26]:


# Rules when min_support = 0.1 and min_threshold for confidence is 0.5
rules4 = association_rules(frequent_itemsets2, metric="confidence", min_threshold=0.5)
rules4


# In[27]:


rules4 = rules4.sort_values(['confidence', 'lift'], ascending =[False, False])
rules4


# In[28]:


plt.scatter(rules4['support'], rules4['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


# In[ ]:




