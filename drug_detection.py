import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
df=pd.read_csv('dataset5.csv')
import matplotlib.pyplot as plt

plt.scatter(df['BindingAffinity_in_-ve'],df['Number_of_people_cured(in 10K)'])
plt.scatter(df['Docking_value_in_-ve'],df['Number_of_people_cured_in_USA(in 10K)'])
plt.scatter(df['Docking_value_in_-ve'],df['Number_of_people_cured_in_USA(in 10K)'])
x=df[['Docking_value_in_-ve','BindingAffinity_in_-ve']]
y=df['Number_of_people_cured_in_USA(in 10K)']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
LinearRegression(copy_X=True,fit_intercept=True,n_jobs=None,normalize=False)
y_pre=reg.predict(x_test)
reg.score(x_test,y_test)
import sklearn
mse=sklearn.metrics.mean_squared_error(y_test,y_pre)
rmse=math.sqrt(mse)
fig,ax=plt.subplots(figsize(15,8))
ax.barh(df['Drug_Name'],df['BindingAffinity_in_-ve'])
plt.xlabel('Effectivity')
plt.ylabel('Drugs')
from random import randint
import random
c_code=[]
random.seed(1000)
for l in range(len(df.Drugs_Name.unique())):
    c_code.append('#%06x' % randint(0,0xFFFFFF))
colors=dict(zip(df.Drug_Name.unique(),c_code))
fig,ax=plt.subplots(figsize=(15,8))
plt.xlabel('BindingAffinity_in_-ve')
plt.ylabel('Drugs')
ax.barh(df['Drug_Name'],df['BindingAffinity_in_-ve'],color=[colors[x] for x in df['Drug_Name']])
for i, (value,name) in enumerate(zip(df['BindingAffinity_in_-ve'],df['Drug_Name'])):
    ax.text(value,i,value,ha='right')
fig,ax=plt.subplots(figsize=(15,8))
plt.xlabel('Docking_value_in_-ve')
plt.ylabel('Drugs')
ax.barh(df['Drug_Name'],df['Docking_Value_in_-ve'],color=[colors[x] for x in df['Drug_Name']])
for i, (value,name) in enumerate(zip(df['Docking_Value_in_-ve'],df['Drug_Name'])):
    ax.text(value, i, value, ha='right')
    
        
