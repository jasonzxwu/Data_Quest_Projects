import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans

#read the dataframe
df = pd.read_csv("D:\Dropbox\Other MOOC\Data Quest\Data_Quest_Projects\9.Predicting the Stock Market\sphist.csv")
df.head(3)

print (df.columns)
df['Date']=pd.to_datetime(df['Date'])

print(df.info())

#check date
from datetime import datetime
print ((df['Date']>datetime(year=2015,month=4,day=1)).sum())
print (len(df))

#sort dataframe on Date column in ascending order
df.sort_values(by='Date',inplace=True)

#check
df.head(3)

# stock market data is sequential
# each observation comes a day after the previous observation
# thus, the observation are not all independent
# we should be careful not to inject 'future' knowledge into past rows when we do training and prediction

#create new features
#5-day average
df['day_5'] = df['Close'].rolling(5).mean()

#check
df.head(5)

#notice that a caveat for rolling function is that the rolling mean
#use the current day's price 
# e.g the rolling mean calculated for 1950-01-03 will need be assigned to 1950-01-04
df['day_5'] = df['day_5'].shift()

#check
# df.head(6)

#30 day average
df['day_30'] = df['Close'].rolling(30).mean()
df['day_30'] = df['day_30'].shift()
#365 day average
df['day_365'] = df['Close'].rolling(365).mean()
df['day_365'] = df['day_365'].shift()

#check
# df.head(10)

#since we computed 365 days average and the dataset starts
# on '1950-01-03', thus any rows with data occured before 
#'1951-01-03' do not have enough historical data to 
#compute all the indicators

#remove rows from dataframe that fall before 1951-01-03
df = df[df['Date']>datetime(year=1951,month=1,day=2)]

#remove any rows with NaN values
df.dropna(axis=0,inplace=True)

#check there is no missing values
print (df.isnull().sum().sum())

train = df[df['Date']<datetime(year=2013,month=1,day=1)]
test= df[df['Date']>datetime(year=2013,month=1,day=1)]

print (train.shape,test.shape)

#making predictions
#error metrics: mean abslute error is recommended as it is intuitive
# mean squared error also can be used

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error

#when train the linear regression model 
#leave out all the original columns (Close, High, Low, Open,
# Volume, Adj Close, Date) because they all contain knowledge of
# the future that you do not want to feed the model
train.head(3)

target='Close'

#univariant
features = ['day_5','day_30','day_365']

#bivariant
#generate combinations of two features in a list
bi_features=[['day_5','day_30'],['day_5','day_365'],['day_30','day_365']]

y_train=train[target]
y_test = test[target]

#univariant
univariant_mae={}
for f in features:
    x_train = train[[f]]
    x_test=test[[f]]
    lr = LinearRegression()
    lr.fit(x_train,y_train)
    y_pred = lr.predict(x_test)
    mae = mean_absolute_error(y_test,y_pred)
    univariant_mae[f]=mae

print (univariant_mae)

#bivariant_mae
bivariant_mae={}

for f in bi_features:
    x_train = train[f]
    x_test=test[f]
    lr = LinearRegression()
    lr.fit(x_train,y_train)
    y_pred = lr.predict(x_test)
    mae = mean_absolute_error(y_test,y_pred)
    bivariant_mae[f]=mae

print (bivariant_mae)

#all features
x_train=train[features]
y_train=test[features]
lr = LinearRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
mae = mean_absolute_error(y_test,y_pred)
print ('all features: ',mae)