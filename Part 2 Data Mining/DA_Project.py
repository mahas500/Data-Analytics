#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


# function for plotting regression function between any two variables
def regressionplt(width, height, dataset, xcol, ycol):
    plt.figure(figsize=(width, height))
    sns.regplot(x = xcol, y = ycol, data = dataset, line_kws={'color':'red'})
    plt.ylim(0,)


# In[4]:


inputFile = r'F:\Download1\Salary_Prediction.csv'
input_df = pd.read_csv(inputFile)


# In[5]:


#number of rows and columns in the dataset
print(input_df.shape)


# In[6]:


# describe data
input_df.describe


# In[7]:


# general statistics of input data
input_df.describe(include=['object', 'int64', 'float64', 'bool'])


# In[8]:


# to check for nulls in each attribute
missing_instance = input_df.isnull().sum()
missing_instance


# In[9]:


#to identify distribution of values in categorical features
print(input_df.groupby('Gender').size().reset_index(name='counts'))
print(input_df.groupby('College Degree').size().reset_index(name='counts'))
print(input_df.groupby('Color of Hairs').size().reset_index(name='counts'))
print(input_df.groupby('Occupation').size().reset_index(name='counts'))
print(input_df.groupby('Country').size().reset_index(name='counts'))


# In[10]:


# checking for outliers by plotting a regression curve of salary with one low cardinal feature
regressionplt(8, 6, input_df, 'Height', 'Salary')


# In[12]:


regressionplt(8, 6, input_df, 'Year', 'Salary')


# In[13]:


regressionplt(8, 6, input_df, 'Age', 'Salary')


# In[11]:


# residue plot for checking data spread
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(input_df['Age'], input_df['Salary'])
#plt.show()


# In[6]:


#removal of outliers[salary cannot be negative]
input_df_Outlier_Removed = input_df[input_df['Salary'] > 0]
#input_df_Outlier_Removed = input_df_Outlier_Removed[input_df_Outlier_Removed['Age'] < 90]
input_df_Outlier_Removed = input_df_Outlier_Removed[input_df_Outlier_Removed['Salary'] < 3000000]


# In[32]:


#regressionplt(8, 6, input_df_Outlier_Removed, 'Height', 'Salary')


# In[28]:


# checking coorelation
corr = input_df_Outlier_Removed.corr()
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
input_df_Outlier_Removed.corr()


# In[7]:


#ignoring population as it is has low corelation.
selected_training_columns = ['Year',
                             'Age',
                             'Gender',
                             'Country',
                             #'Population',
                             'College Degree',
                             'Using Glasses',
                             'Color of Hairs',
                             'Occupation',
                             'Height',
                             'Salary'
                             ]


# In[8]:


dataset = input_df_Outlier_Removed[selected_training_columns]


# In[9]:


# Data Cleaning

# Inputing Gender column
dataset['Gender'].replace('0', 'Unknown', inplace=True)
dataset['Gender'].replace('other', 'Unknown', inplace=True)
dataset['Gender'].replace('unknown', 'Unknown', inplace=True)
dataset['Gender'].replace(np.nan, 'Unknown', inplace=True)


dataset['Color of Hairs'].replace('0', 'Unknown', inplace=True)
dataset['Color of Hairs'].replace('Unknown', 'Unknown', inplace=True)
dataset['Color of Hairs'].replace(np.nan, 'Unknown', inplace=True)


print(dataset.groupby('Gender').size().reset_index(name='counts'))
print(dataset.groupby('Color of Hairs').size().reset_index(name='counts'))


# In[9]:


# command to remove 494 nulls value from age
dataset = dataset[dataset['Age'].notna()]
dataset = dataset[dataset['Year'].notna()]
dataset.isnull().sum()


# In[11]:


dataset['Occupation'].replace(np.nan, 'Unknown', inplace=True)
dataset['College Degree'].replace(np.nan, 'Unknown', inplace=True)

#check for null
dataset.isnull().sum()


# In[12]:


# seperating dependant and independent parts of the dataset
y = dataset['Salary']
dataset.drop('Salary', axis=1, inplace=True)
x_dataset = dataset.copy()


# In[13]:


# Encoding categorical features
# Ref: https://www.datacamp.com/community/tutorials/categorical-data
# Ref: http://thedataist.com/when-categorical-data-goes-wrong/

from feature_engine.categorical_encoders import OneHotCategoricalEncoder

columns_to_encode = ['Gender', 'Country', 'Color of Hairs', 'College Degree', 'Occupation']

encoder = OneHotCategoricalEncoder(top_categories=None,
variables=columns_to_encode,  # we can select which variables to encode
drop_last=True)

encoder.fit(x_dataset)
encoded_dataet = encoder.transform(x_dataset)


# In[14]:


# Splitting data into training and testing subset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(encoded_dataet, y, test_size=0.2, random_state=0)


# In[85]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[86]:


# Linear regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[16]:


# Running pediction
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[17]:


# Evaluating Model performance
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[15]:


#graphical representation of the model performance
from sklearn import metrics
def plot_prediction(y_test, y_pred):
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = df.head(25)
    df1.plot(kind='bar', figsize=(10, 8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    # Calculate mean absolute percentage error (MAPE)
    errors = abs(y_pred - y_test)
    mape = 100 * (errors / y_test)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')


# In[19]:


plot_prediction(y_test, y_pred)


# In[10]:


# Alternate data imputation strategy
age_median = dataset['Age'].median()
dataset['Age'].replace(np.nan, age_median, inplace=True)
dataset['Age'] = (dataset['Age'] * dataset['Age']) ** (0.5)

year_median = dataset['Year'].median()
dataset['Year'].replace(np.nan, year_median, inplace=True)


# In[87]:


model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[88]:


plot_prediction(y_test, y_pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[42]:


# Plotting density curve to see if the values are skewed
df = pd.DataFrame({'Salary': y})
sns.distplot(df['Salary'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})


# In[89]:


# Map y to log scale handle skewed value in training
y_train_log = np.log(y_train)


# In[90]:


# retraining the regressor model
model.fit(X_train, y_train_log)
y_pred = np.exp(model.predict(X_test)) #Taking inverse log of the predicted values


# In[91]:


plot_prediction(y_test, y_pred)


# In[ ]:


####......Much Better......####


# In[ ]:


# Trying other regressor models
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = lab_enc.fit_transform(y_train)

logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, training_scores_encoded)
y_pred = logisticRegr.predict(X_test)
plot_prediction(y_test, y_pred)


# In[16]:


# TODO Try other encoding methods and complex models
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 5, random_state = 42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
plot_prediction(y_test, y_pred)


# In[ ]:


# TODO Try other encoding methods and complex models
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 10, random_state = 42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
plot_prediction(y_test, y_pred)


# In[17]:


pip install catboost


# In[19]:


from catboost import CatBoostRegressor
model = CatBoostRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
plot_prediction(y_test, y_pred)


# In[26]:


model = CatBoostRegressor(iterations=10000,
                          learning_rate=0.03,
                          depth=6)
y_train_log = np.log(y_train)
model.fit(X_train, y_train_log)
y_pred = np.exp(model.predict(X_test))
plot_prediction(y_test, y_pred)


# In[ ]:




