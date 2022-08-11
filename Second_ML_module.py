#importing numpy library
import numpy as np

#this is to import pandas 
import pandas as pd

#this code is to import matplotlib library
import matplotlib.pyplot as plt

#creating a variable to store a Dataset
Dataset3 = pd.read_csv("salaryData.csv")

#creating a variable x and y to store the independent and dependent
x= Dataset3.iloc[:,0:1].values
y= Dataset3.iloc[:,1:2].values

#splitting the Daset into train Data and test Data
from sklearn.model_selection import train_test_split


#creating variable for to store x_train, x_test and  y_train, y_test
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)

#training the Linear Regression Module
from sklearn.linear_model import LinearRegression


#creating a variable and assigning the Linear Regression algorithm
MY_second_module = LinearRegression()

#training the Module MY_second_module with x train and y train
MY_second_module.fit(x_train,y_train)

#making a prediction
prediction_result = MY_second_module.predict(x_test)
prediction_result
