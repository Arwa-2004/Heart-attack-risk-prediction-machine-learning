#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt 
import seaborn as sns 

get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


heart= pd.read_csv('heart_attack_prediction.csv')


# In[17]:


heart.head()


# In[18]:


heart = heart.drop(columns=['Country', 'Continent','Hemisphere'])


# In[19]:


heart.head()


# In[20]:


heart.info()


# In[21]:


ordinal_map = {'Healthy':2,'Average':1,'Unhealthy':0}
heart['Diet'] = heart['Diet'].map(ordinal_map)


# In[22]:


heart.describe()


# In[23]:


ordinal_map_sex= {'Male':0, 'Female':1}
heart['Sex']=heart['Sex'].map(ordinal_map_sex)
heart.head(5)



# In[24]:


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
heart['Patient ID'] = encoder.fit_transform(heart['Patient ID'])


# In[25]:


heart[['BP_Systolic', 'BP_Diastolic']] = heart['Blood Pressure'].str.split('/', expand=True)

# Convert the columns to numeric
heart['BP_Systolic'] = pd.to_numeric(heart['BP_Systolic'])
heart['BP_Diastolic'] = pd.to_numeric(heart['BP_Diastolic'])
heart = heart.drop('Blood Pressure', axis=1)


# In[26]:


heart.isna().sum()


# In[27]:


sns.countplot(data=heart, x='Sex', palette= 'pastel')


# In[ ]:


sns.pairplot(data=heart, hue='Sex')
plt.show()


# In[ ]:


# specifying target and features
heart.columns


# In[ ]:


X= heart[['Patient ID', 'Age', 'Sex', 'Cholesterol',
       'Heart Rate', 'Diabetes', 'Family History', 'Smoking', 'Obesity',
       'Alcohol Consumption', 'Exercise Hours Per Week', 'Diet',
       'Previous Heart Problems', 'Medication Use', 'Stress Level',
       'Sedentary Hours Per Day', 'Income', 'BMI', 'Triglycerides',
       'Physical Activity Days Per Week', 'Sleep Hours Per Day',
       'Heart Attack Risk', 'BP_Systolic', 'BP_Diastolic']]


# In[15]:


y= heart['Heart Attack Risk']


# In[16]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
                                                    
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
                       


# In[17]:


sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[18]:


classifier = KNeighborsClassifier(n_neighbors=5)


# In[19]:


classifier.fit(X_train, y_train)


# In[20]:


predictions = classifier.predict(X_test)


# In[21]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# In[22]:


cm=confusion_matrix(y_test, predictions)
print(confusion_matrix(y_test, predictions))

cr=classification_report(y_test, predictions)
print(classification_report(y_test, predictions))


# In[23]:


ar=accuracy_score(y_test, predictions)
print(accuracy_score(y_test, predictions))


# In[24]:


sns.heatmap(cm, annot=True, cmap='Greens', fmt='d')


# In[25]:


# 568= TP, CORRECTLY PREDICTED TO BE AT RISK OF HAVING HEART ATTACK :(
# 1140 = TN, CORRECTLY PREDICATED TO NOT BE AT RISK OF HEART ATTACK :)
# 2= FP, FALSLY PREDICATED TO HAVE RISK OF HEART ATTACK
# 43= FN, FALSLY PREDICTED TO HAVE NO RISK OF HEART ATTACK

