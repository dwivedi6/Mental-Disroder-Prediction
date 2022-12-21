
## Program for heart desease detection on given parameters
################### MT19AIE323 #################

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading the csv data to a Pandas DataFrame
h_db = pd.read_csv('D:/MTech/Semester VI/MTP/dataset/heart.csv')

# checking for missing values
h_db.isnull().sum()

# statistical measures about the data
h_db.describe()

# checking the distribution of Target Variable
h_db['target'].value_counts()

X = h_db.drop(columns='target', axis=1)
Y = h_db['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

model = LogisticRegression()

# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Test data : ', test_data_accuracy)

input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')
  
import pickle

filename = 'D:/MTech/Semester VI/MTP/saved models/mt19aie323_heart_disease_model.sav'
pickle.dump(model, open(filename, 'wb'))

# loading the saved model
loaded_model = pickle.load(open('D:/MTech/Semester VI/MTP/saved models/mt19aie323_heart_disease_model.sav', 'rb'))

for column in X.columns:
  print(column)
  



