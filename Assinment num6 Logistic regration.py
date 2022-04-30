#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[2]:


# Importing the dataset
bank=pd.read_csv('bank-full.csv',sep =";")
bank


# In[3]:


bank.info()


# In[4]:


data1=pd.get_dummies(bank,columns=['job','marital','education','contact','poutcome','month'])
data1


# In[5]:


# To see all columns
pd.set_option("display.max.columns", None)
data1


# In[6]:


data1.info()


# In[7]:


# Custom Binary Encoding of Binary o/p variables
data1['default'] = np.where(data1['default'].str.contains("yes"), 1, 0)
data1['housing'] = np.where(data1['housing'].str.contains("yes"), 1, 0)
data1['loan'] = np.where(data1['loan'].str.contains("yes"), 1, 0)
data1['y'] = np.where(data1['y'].str.contains("yes"), 1, 0)
data1


# In[8]:


data1.info()


# In[9]:


# Dividing our data into input and output variables
x=pd.concat([data1.iloc[:,0:10],data1.iloc[:,11:]],axis=1)
y=data1.iloc[:,10]


# In[10]:


# Logistic regression model
classifier=LogisticRegression()
classifier.fit(x,y)


# In[11]:


y_pred=classifier.predict(x)
y_pred


# In[12]:



y_pred_df=pd.DataFrame({'actual_y':y,'y_pred_prob':y_pred})
y_pred_df


# In[13]:


# Confusion Matrix for the model accuracy
confusion_matrix = confusion_matrix(y,y_pred)
confusion_matrix


# In[15]:


#The model accuracy is calculated by (a+d)/(a+b+c+d)
(39156+1162)/(39156+766+4127+1162)


# In[18]:



# As accuracy = 0.8933, which is greater than 0.5; Thus [:,1] Threshold value>0.5=1 else [:,0] Threshold value<0.5=0 

classifier.predict_proba(x)[:,1]


# In[ ]:



fpr,tpr,thresholds=roc_curve(y,classifier.predict_proba(x)[:,1])
plt.plot(fpr,tpr,color='red')
auc=roc_auc_score(y,y_pred)

plt.plot(fpr,tpr,color='red',label='logit model(area  = %0.2f)'%auc)
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')
plt.show()

print('auc accuracy:',auc)


# In[ ]:


# ROC Curve plotting and finding AUC value
fpr,tpr,threshold=roc_curve(y,classifier.predict_proba(x)[:,1])
plt.plot(fpr,tpr,colour='red')
auc=roc_auc_score(y,y_pred)
plt.plot(fpr)

