#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd 
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from nltk.stem import WordNetLemmatizer 
data =pd.read_excel("D:\\Nalini tak\\Data SciencE\\Poject\\Hotelreview\\hotel_reviews.xlsx")


# In[2]:


data.head()


# **Attaching a sentiment based on the rating
# 
# If rating=5, sentiment is 2 (means very positive)
# 
# If rating is between 3 and 4, sentiment is 1 (means neutral)
# 
# If rating is between 1 and 2, sentiment is 0 (means very negative)**

# In[34]:


pos = [5] 
neg = [1, 2]
neu = [3, 4] 

def sentiment(rating):
    if rating in pos: 
        return 2 
    elif rating in neg:
        return 0 
    else:
        return 1

data['Setiment'] = data['Rating'].apply(sentiment)


# In[35]:


data.head()


# In[36]:


data = data.iloc[0:2000]


# In[37]:


data.shape


# In[38]:


data.head()


# **Converting the entire corpus to lowercase**

# In[39]:


def convert_lower(text):
    return text.lower()

data['Review'] = data['Review'].apply(convert_lower)


# **Removing all special characters, punctuation marks, numbers from the corpus**

# In[40]:


def remove_special(text):
    s = ''
    for i in text:
        if i.isalpha():
            s = s+i 
        else: 
            s = s+ ' '
    return s 

data['Review'] = data['Review'].apply(remove_special)


# **Removing stopwords like is, are etc. from the corpus and appending the text as a list**

# In[41]:


import nltk 
from nltk.corpus import stopwords 
nltk.download('stopwords')

def remove_stopwords(text):
    x = []
    for i in text.split():
        if i not in stopwords.words('english'):
            x.append(i) 
    y = x[:]
    x.clear()
    return y 

data['Review'] = data['Review'].apply(remove_stopwords)


# In[42]:


data.head()


# **Lemmatization**

# In[43]:


from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')
nltk.download('omw-1.4')

x  = []
def lemmatize_words(text):
    for i in text: 
        x.append(lemmatizer.lemmatize(i)) 
    y = x[:]
    x.clear()
    return y 

data['Review'] = data['Review'].apply(lemmatize_words)


# In[44]:


data.head()


# **Joining the words in the list to form sentences**

# In[45]:


def join_text(text):
    return " ".join(text)

data['Review'] = data['Review'].apply(join_text)


# In[46]:


data.head()


# In[47]:


pip install text2emotion


# In[48]:


import text2emotion as te 


# In[49]:


def feeling(text):
    all_emotions_value = te.get_emotion(text)
    keymax_value = max(zip(all_emotions_value.values(), all_emotions_value.keys()))[1]
    return keymax_value

data['Emotion'] = data['Review'].apply(feeling)


# In[50]:


data.head()


# In[52]:


dummies = pd.get_dummies(data.Emotion)
dummies 


# In[53]:


data = pd.concat([data, dummies], axis='columns')
data.head()


# In[54]:


pd.crosstab(data.Emotion, data.Rating).plot(kind='bar')


# In[73]:


docs = list(data['Review'])[:2000]


# In[74]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(use_idf=True, max_features=60)


# In[75]:


tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(docs)


# In[76]:


print(len(tfidf_vectorizer.vocabulary_))


# In[77]:


X = tfidf_vectorizer_vectors.toarray()
Y = data['Emotion'][:2000]


# In[61]:


import matplotlib.pyplot as plt
x = data['Setiment'].unique()
y = data['Setiment'].value_counts()
plt.bar(x, y)
plt.xlabel('Sentiments')
plt.ylabel('Values')
plt.show()


# In[62]:


def sentiment(rating):
    if rating==2: 
        return 'Positive'
    elif rating==1:
        return 'Neutral'
    else:
        return 'Negative'

data['Overall'] = data['Setiment'].apply(sentiment)
        


# In[63]:


data.head()


# In[92]:


data.drop(['Overall'], axis='columns', inplace=True)


# In[93]:


data.head()


# **Splitting into train and test data**

# In[94]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=123, stratify=Y, test_size=0.2)


# In[95]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, Y_train)
y_pred_train = gnb.predict(X_train)
y_pred_test = gnb.predict(X_test)
print("Training Accuracy score: "+str(round(accuracy_score(Y_train,lr.predict(X_train)),4)))
print("Testing Accuracy score: "+str(round(accuracy_score(Y_test,lr.predict(X_test)),4)))


# **Accuracy Analysis**

# In[96]:


from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, classification_report, roc_auc_score,roc_curve,auc


# In[97]:


print(classification_report(Y_test, y_pred_test))


# In[98]:


cm = confusion_matrix(Y_test, y_pred_test)
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[99]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=123).fit(X_train, Y_train)
y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)
print("Training Accuracy score: "+str(round(accuracy_score(Y_train,lr.predict(X_train)),4)))
print("Testing Accuracy score: "+str(round(accuracy_score(Y_test,lr.predict(X_test)),4)))


# In[100]:


print(classification_report(Y_test, y_pred_test))


# In[101]:


cm = confusion_matrix(Y_test, y_pred_test)
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[102]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, Y_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
print("Training Accuracy score: "+str(round(accuracy_score(Y_train,clf.predict(X_train)),4)))
print("Testing Accuracy score: "+str(round(accuracy_score(Y_test,clf.predict(X_test)),4)))


# In[103]:


print(classification_report(Y_test, y_pred_test))


# In[104]:


cm = confusion_matrix(Y_test, y_pred_test)
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


# In[6]:


X=data.iloc[:,1:501]
Y=data.iloc[:,0]


# In[7]:


X.head(3)


# In[8]:


x_train, x_test,y_train,y_test = train_test_split(X,Y, test_size=0.2,random_state=40)


# In[ ]:


#Logistic regression and fit the model
model_1 = LogisticRegression()
model_1.fit(x_train,y_train)


# In[ ]:


#Predict for train dataset
pred_train_LR=model_1.predict(x_train)
np.mean(pred_train_LR==y_train)


# In[ ]:


pd.Series(pred_train_LR).value_counts()pd.Series(pred_train_LR).value_counts()


# In[ ]:


np.mean(pred_test_LR==y_test)


# In[ ]:





# In[ ]:


pred_test_LR=model_1.predict(x_test)


# In[ ]:


# Confusion Matrix for the model accuracy
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_train,pred_train_LR)
print (confusion_matrix)

