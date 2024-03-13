#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import pandas as pd


# In[2]:


dp = pd.read_excel('play_tennis.xlsx')
dp.dtypes


# In[3]:


le = LabelEncoder()
le.fit(['outlook','temp','humidity','wind','play'])
for i in dp.columns:
    dp[i] = le.fit_transform(dp[i])
imp_data = dp.iloc[:,1:5:1]
opt = dp.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(imp_data,opt,test_size=0.2,random_state=5)
clf = GaussianNB()
clf = clf.fit(x_train,y_train)
pred = clf.predict(x_test)
print('Accuracy is :',metrics.accuracy_score(y_test,pred))


# In[ ]:




