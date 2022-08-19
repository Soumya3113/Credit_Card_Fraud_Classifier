#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
data = pd.read_csv("C:\\Users\\Soumya\\Downloads\\creditcard.csv")
data_final=pd.DataFrame(data)


# In[2]:


import gc
from datetime import datetime 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


# # Data analysis

# In[9]:


data_final.shape


# In[5]:


data_final.dtypes


# In[6]:


data_final.head()


# In[7]:


data_final.describe()


# Missing data analysis

# In[11]:


data_final.isnull()


# In[12]:


total = data_final.isnull().sum().sort_values(ascending=False)
percent = (data_final.isnull().sum()/data_final.isnull().count()*100).sort_values(ascending = False)
pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose()


# No missing data found

# Checking unbalanced data with respect to target variable Class

# In[13]:


a = data_final["Class"].value_counts()
df = pd.DataFrame({'Class': a.index,'values': a.values})

temp = go.Bar(
    x = df['Class'],y = df['values'],
    name="Credit Card Fraud Class - data unbalance (Not fraud = 0, Fraud = 1)",
    marker=dict(color="Red"),
    text=df['values']
)
data = [temp]
layout = dict(title = 'Credit Card Fraud Class - data unbalance (Not fraud = 0, Fraud = 1)',
          xaxis = dict(title = 'Class', showticklabels=True), 
          yaxis = dict(title = 'Number of transactions'),
          hovermode = 'closest',width=600
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='class')


# As we can see from the above graph, only 492 transactions are fraud. This means the given dataset is highly unbalanced

# # Data Exploration

# Timely transactions

# In[14]:


class_0 = data_final.loc[data_final['Class'] == 0]["Time"]
class_1 = data_final.loc[data_final['Class'] == 1]["Time"]

hist_data = [class_0, class_1]
group_labels = ['Not Fraud', 'Fraud']

fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
fig['layout'].update(title='Credit Card Transactions Time Density Plot', xaxis=dict(title='Time [s]'))
iplot(fig, filename='dist_only')


# In[15]:


data_final['Hour'] = data_final['Time'].apply(lambda x: np.floor(x / 3600))
tmp = data_final.groupby(['Hour', 'Class'])['Amount'].aggregate(['min', 'max', 'count', 'sum', 'mean', 'median', 'var']).reset_index()
df = pd.DataFrame(tmp)
df.columns = ['Hour', 'Class', 'Min', 'Max', 'Transactions', 'Sum', 'Mean', 'Median', 'Var']
df.head()


# In[16]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Sum", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Sum", data=df.loc[df.Class==1], color="red")
plt.suptitle("Total Amount")
plt.show();


# In[17]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Transactions", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Transactions", data=df.loc[df.Class==1], color="red")
plt.suptitle("Total Number of Transactions")
plt.show();


# In[18]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Mean", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Mean", data=df.loc[df.Class==1], color="red")
plt.suptitle("Average Amount of Transactions")
plt.show();


# In[19]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Max", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Max", data=df.loc[df.Class==1], color="red")
plt.suptitle("Maximum Amount of Transactions")
plt.show();


# In[20]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Median", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Median", data=df.loc[df.Class==1], color="red")
plt.suptitle("Median Amount of Transactions")
plt.show();


# In[21]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Min", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Min", data=df.loc[df.Class==1], color="red")
plt.suptitle("Minimum Amount of Transactions")
plt.show();


# In[25]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6)) ## Transactions Amount
s = sns.boxplot(ax = ax1, x="Class", y="Amount", hue="Class",data=data_final, palette="PRGn",showfliers=True)
s = sns.boxplot(ax = ax2, x="Class", y="Amount", hue="Class",data=data_final, palette="PRGn",showfliers=False)
plt.show();


# In[26]:


t = data_final[['Amount','Class']].copy()
class_0 = t.loc[t['Class'] == 0]['Amount']
class_1 = t.loc[t['Class'] == 1]['Amount']
class_0.describe()


# In[27]:


class_1.describe()


# In[32]:


fraud = data_final.loc[data_final['Class'] == 1]

trace = go.Scatter(
    x = fraud['Time'],y = fraud['Amount'],
    name="Amount",
     marker=dict(
                color='rgb(238,23,11)',
                line=dict(
                    color='black',
                    width=1),
                opacity=0.5,
            ),
    text= fraud['Amount'],
    mode = "markers"
)
data = [trace]
layout = dict(title = 'Fraud Transactions',
          xaxis = dict(title = 'Time [s]', showticklabels=True), 
          yaxis = dict(title = 'Amount'),
          hovermode='closest'
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='fraud-amount')


# Features Correlation Analysis

# In[34]:


plt.figure(figsize = (14,14))
plt.title('Transactions Feature Correlation Plot (Pearson)')
corr = data_final.corr()
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,cmap="Reds")
plt.show()


# We can observe that Time is related to V3 and Amount to V7, V20, V1. V5

# In[36]:


s = sns.lmplot(x='V20', y='Amount',data=data_final, hue='Class', fit_reg=True,scatter_kws={'s':2}) ## V7 and V20 vs Amount
s = sns.lmplot(x='V7', y='Amount',data=data_final, hue='Class', fit_reg=True,scatter_kws={'s':2})
plt.show()


# In[37]:


s = sns.lmplot(x='V2', y='Amount',data=data_final, hue='Class', fit_reg=True,scatter_kws={'s':2}) ## V2 and V5 vs Amount
s = sns.lmplot(x='V5', y='Amount',data=data_final, hue='Class', fit_reg=True,scatter_kws={'s':2})
plt.show()


# Density Plots

# In[39]:


v = data_final.columns.values

i = 0
t0 = data_final.loc[data_final['Class'] == 0]
t1 = data_final.loc[data_final['Class'] == 1]

sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(8,4,figsize=(16,28))

for feature in v:
    i += 1
    plt.subplot(8,4,i)
    sns.kdeplot(t0[feature], bw=0.5,label="Class = 0")
    sns.kdeplot(t1[feature], bw=0.5,label="Class = 1")
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();


# # Building Predictive Models

# Target and predictors values

# In[40]:


target = 'Class'
predictors = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',       'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',       'Amount']


# Defining metrics

# In[41]:


random_metric = 'gini'
num_random_estimators = 100
num_random_jobs = 4
valid_set = 0.20 #test-train-validation dataset split
test_set = 0.20
num_kfolds_cross_valid = 5 #number of k-folds used in cross validation
randomstate = 2018


# Test, train and validation dataset split

# In[44]:


train_data, test_data = train_test_split(data_final, test_size=test_set, random_state=randomstate, shuffle=True )
train_data, valid_data = train_test_split(train_data, test_size=valid_set, random_state=randomstate, shuffle=True )


# # RandomForest Classifier

# In[45]:


clf = RandomForestClassifier(n_jobs=num_random_jobs, 
                             random_state=randomstate,
                             criterion=random_metric,
                             n_estimators=num_random_estimators,
                             verbose=False)


# In[47]:


clf.fit(train_data[predictors], train_data[target].values)


# In[48]:


preds = clf.predict(valid_data[predictors])


# In[49]:


tmp = pd.DataFrame({'Feature': predictors, 'Feature importance': clf.feature_importances_}) ## Plotting features importance
tmp = tmp.sort_values(by='Feature importance',ascending=False)
plt.figure(figsize = (7,4))
plt.title('Features importance',fontsize=14)
s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show() 


# As we can see from given graph, V17, V12, V14, V10, V11, V16 are most important

# In[51]:


cm = pd.crosstab(valid_data[target].values, preds, rownames=['Actual'], colnames=['Predicted']) ## Plotting confusion matrix
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
sns.heatmap(cm, 
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'],
            annot=True,ax=ax1,
            linewidths=.2,linecolor="Darkblue", cmap="Blues")
plt.title('Confusion Matrix', fontsize=14)
plt.show()


# Calculation of ROC-AUC score

# In[53]:


roc_auc_score(valid_data[target].values, preds)


# # AdaBoost Classifier

# In[54]:


clf = AdaBoostClassifier(random_state=randomstate,
                         algorithm='SAMME.R',
                         learning_rate=0.8,
                             n_estimators=num_random_estimators)


# In[55]:


clf.fit(train_data[predictors], train_data[target].values) ## Model fitting


# In[56]:


preds = clf.predict(valid_data[predictors]) ## Target values prediction


# In[57]:


t = pd.DataFrame({'Feature': predictors, 'Feature importance': clf.feature_importances_}) ## Feature importance plotting
t = t.sort_values(by='Feature importance',ascending=False)
plt.figure(figsize = (7,4))
s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show() 


# In[58]:


c = pd.crosstab(valid_data[target].values, preds, rownames=['Actual'], colnames=['Predicted']) ## Confusion matrix
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
sns.heatmap(c, 
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'],
            annot=True,ax=ax1,
            linewidths=.2,linecolor="Darkblue", cmap="Blues")
plt.title('Confusion Matrix', fontsize=14)
plt.show()


# Calculation of ROC-AUC score

# In[59]:


roc_auc_score(valid_data[target].values, preds)


# In[ ]:




