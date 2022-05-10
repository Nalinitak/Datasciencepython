#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install mlxtend')


# In[ ]:





# In[2]:



# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder


# In[5]:



# Loading dataset
data = pd.read_csv('book.csv')
data


# In[7]:


data.shape


# In[8]:


data.info()


# In[9]:


get_ipython().system(' pip install wordcloud')


# In[10]:



book_count = []
col_names = data.columns
for col_name in col_names:
    book_count.append(data[col_name].value_counts()[1])


# In[11]:


book_count 


# In[12]:


plt.bar(col_names, book_count)


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
rules1 = association_rules(frequent_itemsets1, metric="lift", min_threshold=0.5)
rules1


# In[16]:


rules1.sort_values('lift',ascending = False)


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


# In[20]:


plt.scatter(rules2['support'], rules2['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


# In[21]:


# Apriori Algorithm for min_support = 0.2
frequent_itemsets2 = apriori(data, min_support=0.2, use_colnames=True)
frequent_itemsets2


# In[26]:


frequent_itemsets2 = apriori(data, min_support = 0.2, use_colnames=True)
frequent_itemsets2['length'] = frequent_itemsets2['itemsets'].apply(lambda x: len(x))
frequent_itemsets2


# In[ ]:


# Rules when min_support = 0.2 and min_threshold for lift is 0.5
rules3 = association_rules(frequent_itemsets2, metric="lift", min_threshold=0.1)
rules3

