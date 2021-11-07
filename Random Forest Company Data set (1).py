#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Import Lib

import pandas as pd
import numpy as np
import matplotlib.pyplot


# In[4]:


Company= pd.read_csv("C:\\Users\\Admin\\Downloads\\Company_Data (2).csv")
Company.head()


# In[7]:


##Converting the Sales variable to bucketing. 
Company["Sales"]="<=30000"
Company.loc[Company["Income"]>=30000,"Income"]="Good"
Company.loc[Company["Income"]<=30000,"Income"]="Risky"


# In[9]:


##Droping theincome variable
Company.drop(["Income"],axis=1,inplace=True)

Company.rename(columns={"Sales":"Sales","CompPrice":"compPrice","Income":"income","Advertising":"advertising","Urban":"urban"},inplace=True)
## As we are getting error as "ValueError: could not convert string to float: 'YES'".
## Model.fit doesnt not consider String. So, we encode


# In[11]:


from sklearn import preprocessing
le=preprocessing.LabelEncoder()
for column_name in Company.columns:
    if Company[column_name].dtype == object:
        Company[column_name] = le.fit_transform(Company[column_name])
    else:
        pass


# In[12]:


##Splitting the data into featuers and labels
features = Company.iloc[:,0:5]
labels = Company.iloc[:,5]


# In[13]:


## Collecting the column names
colnames = list(Company.columns)
predictors = colnames[0:5]
target = colnames[5]


# In[14]:


##Splitting the data into train and test

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size = 0.2,stratify = labels)


# In[15]:


##Model building
from sklearn.ensemble import RandomForestClassifier as RF
model = RF(n_jobs = 3,n_estimators = 15, oob_score = True, criterion = "entropy")
model.fit(x_train,y_train)

model.estimators_
model.classes_
model.n_features_
model.n_classes_

model.n_outputs_

model.oob_score_


# In[16]:


##Predictions on train data
prediction = model.predict(x_train)


# In[17]:


##Accuracy
# For accuracy 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_train,prediction)


# In[18]:


np.mean(prediction == y_train)


# In[19]:


##Confusion matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_train,prediction)


# In[20]:


##Prediction on test data
pred_test = model.predict(x_test)


# In[21]:


##Accuracy
acc_test =accuracy_score(y_test,pred_test)


# In[22]:


## In random forest we can plot a Decision tree present in Random forest
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.externals.six import StringIO

tree = model.estimators_[5]

dot_data = StringIO()
export_graphviz(tree,out_file = dot_data, filled = True,rounded = True, feature_names = predictors ,class_names = target,impurity =False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())


# In[23]:


## Creating pdf and png file the selected decision tree
graph.write_pdf('Companyrf.pdf')
graph.write_png('Companyrf.png')


# In[ ]:




