
# coding: utf-8

# In[32]:


import pandas as pd


# Import the dataset
# and check if there is any missing value

# In[10]:


df = pd.read_csv('UCI_Credit_Card.csv')
df.info()


# change the ugly feature name

# In[11]:


df = df.rename(columns = {'PAY_0':'PAY_1'})
df[['PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']].describe()


# Notice that there are some strange value like -2 in features such as PAY_1 to PAY_6
# and I will deal with it later

# In[34]:


df[['SEX','MARRIAGE','EDUCATION']].describe()


# Same situation happend in 'SEX','MARRIAGE','EDUCATION'

# In[13]:


df[['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']].describe()


# Try to draw some figure to find out if there is some correlation between 'PAY_' features

# In[14]:


import matplotlib.pyplot as plt
df_test = df.copy()
df_test.sort_values(by = ['PAY_1'])
df_test.plot.scatter(x = 'PAY_1', y = 'PAY_6')
plt.show()


# In[15]:


df_test.sort_values(by=['AGE'])
df_test.plot.scatter(x = 'AGE', y = 'PAY_1')
df_test.sort_values(by=['MARRIAGE'])
df_test.plot.scatter(x = 'MARRIAGE', y = 'PAY_1')


# In[16]:


df_test


# In[17]:


from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.metrics import fbeta_score


# In[18]:


x = df[['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1', 'PAY_2',
       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']]
y = df['default.payment.next.month']


# In[19]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
x_train


# In[20]:


classifier = DecisionTreeClassifier(max_depth = 10, random_state = 14)
classifier.fit(x_train, y_train)
prediction = classifier.predict(x_test)
print('accuracy:')
print(accuracy_score(y_true = y_test, y_pred = prediction))
print('f_beta:')
print(fbeta_score(y_true = y_test, y_pred = prediction, beta = 2))


# In[21]:


classifier = DecisionTreeClassifier(max_depth = 100, random_state = 14)
classifier.fit(x_train, y_train)
prediction = classifier.predict(x_test)
print('accuracy:')
print(accuracy_score(y_true = y_test, y_pred = prediction))
print('f_beta:')
print(fbeta_score(y_true = y_test, y_pred = prediction, beta = 2))


# In[22]:


df = df[~df['PAY_1'].isin([-2])]


# In[23]:


df = df[~df['PAY_2'].isin([-2])]


# In[24]:


df = df[~df['PAY_3'].isin([-2])]


# In[25]:


df = df[~df['PAY_4'].isin([-2])]


# In[26]:


df = df[~df['PAY_5'].isin([-2])]


# In[27]:


df = df[~df['PAY_6'].isin([-2])]
df


# In[28]:


x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, test_size = 0.2, random_state = 14)
classifier = DecisionTreeClassifier(max_depth = 10, random_state = 14)
classifier.fit(x_train2, y_train2)
prediction2 = classifier.predict(x_test2)
print('accuracy:')
print(accuracy_score(y_true = y_test2, y_pred = prediction2))
print('f_beta:')
print(fbeta_score(y_true = y_test2, y_pred = prediction2, beta = 2))


# In[29]:


from sklearn.feature_selection import VarianceThreshold


# In[30]:


sel = VarianceThreshold(0.7*(1 - 0.7))
sel.fit_transform(df)
df


# In[31]:


x_train3, x_test3, y_train3, y_test3 = train_test_split(x, y, test_size = 0.2, random_state = 14)
classifier = DecisionTreeClassifier(max_depth = 10, random_state = 14)
classifier.fit(x_train3, y_train3)
prediction3 = classifier.predict(x_test3)
print('accuracy:')
print(accuracy_score(y_true = y_test3, y_pred = prediction3))
print('f_beta:')
print(fbeta_score(y_true = y_test3, y_pred = prediction3, beta = 2))

