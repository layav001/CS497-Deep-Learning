import numpy as np
import pandas as p
from tkinter import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import json
import seaborn as sb
import cv2
from sklearn.linear_model import LogisticRegression
import joblib 

col_names = ['mean_radius','mean_texture','mean_perimeter','mean_area','mean_smoothness','diagnosis']
#this data is for reading breast cancer
data = p.read_csv("Breast_cancer_data.csv",header = None,names = col_names,skiprows = 6)

print(data.shape)
data.head()

corr = data.corr()

feature_cols = ['mean_radius','mean_texture','mean_perimeter','mean_area','mean_smoothness']
x = data[feature_cols]
y = data.diagnosis

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

model = LogisticRegression()
model.fit(X_train,y_train)

filename = 'breastCancer.pkl'
joblib.dump(model,filename)

#this information was used to give the accuracy of our model and see what was predicted right and wrong
#from sklearn.metrics import classification_report
#print(classification_report(y_test,y_pred))

#from sklearn.metrics import confusion_matrix 
#cm = confusion_matrix(y_test, y_pred) 
  
#print ("Confusion Matrix : \n", cm) 