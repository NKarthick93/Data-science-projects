# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 17:19:19 2020

@author: karthick
"""
##importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#importing dataset
train=pd.read_csv("A:\\dataset\\google_stock\\train.csv")

train.head()
training=train.iloc[:,1:2].values

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range = (0,1), copy=True)
train_set=sc.fit_transform(training)

#create adata structure with 60 timesteps and i output without keras
X_train=[]
y_train=[]
for i in range(60,len(train_set)):
    X_train.append(train_set[i-60:i,0])
    y_train.append(train_set[i,0])
X_train,y_train=np.array(X_train), np.array(y_train)    

#reshaping
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1], 1))

#Building with RNN

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Intialising the RNN
regressor=Sequential()

#Adding first lstm layer and Dropout regularisation

regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

#Adding second lstm layer and Dropout regularisation

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

#Adding third lstm layer and Dropout regularisation

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

#Adding fourth lstm layer and Dropout regularisation

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

#Adding output layer
regressor.add(Dense(units=1))


#compile the RNN
regressor.compile(optimizer='adam',loss='mean_squared_error')

#Fitting the RNN to the Training set
regressor.fit(X_train,y_train,epochs=50,batch_size=32)

#making the prediction and visualization the results

#Test set
dataset=pd.read_csv("A:\\dataset\\google_stock\\test_set.csv")
real_price=dataset.iloc[:,1:2].values

#Getting the predicted stock price

dataset_total=pd.concat((train['Open'],dataset['Open']),axis=0)
inputs=dataset_total[len(dataset_total) - len(dataset)-60:].values
inputs=inputs.reshape(-1,1)
inputs=sc.transform(inputs)
X_test=[]
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])
X_test=np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_stock_price=regressor.predict(X_test)
predicted_stock_price=sc.inverse_transform(predicted_stock_price)

#Visualization

plt.plot(real_price, color='green',label=real_price)
plt.plot(predicted_stock_price, color='red',label='Predicted stock_price')
plt.title('Stock price predicted')
plt.xlabel('Time')
plt.ylabel('Stock price')
plt.legend()
plt.show()