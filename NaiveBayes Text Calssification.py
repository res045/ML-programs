#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


# In[2]:


data = pd.read_csv('text_classification.csv',names=['Message','Label'])
data


# In[4]:


data['LabelNo'] = data['Label'].map({'pos':1,'neg':0})
input_data = data['Message']
output = data['LabelNo']

x_train,x_test,y_train,y_test = train_test_split(input_data,output,test_size=0.2,random_state=5)

cv = CountVectorizer()
x_train_fit = cv.fit_transform(x_train)
x_test_fit = cv.transform(x_test)

data12 = pd.DataFrame(x_train_fit.toarray(),columns=cv.get_feature_names_out())

mnb = MultinomialNB()
mnb.fit(x_train_fit,y_train)
pred = mnb.predict(x_test_fit)


# In[5]:


print('Accuracy is :',metrics.accuracy_score(y_test,pred))
print('Precision is :',metrics.precision_score(y_test,pred))
print('Recall is :',metrics.recall_score(y_test,pred))


# In[ ]:




