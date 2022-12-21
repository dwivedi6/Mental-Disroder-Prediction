## Program for diabetes desease prediction on given parameters
################### MT19AIE323 #################


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# loading the diabetes dataset to a pandas DataFrame
dia_db = pd.read_csv('D:/MTech/Semester VI/MTP/dataset/diabetes.csv') 


# printing the first 5 rows of the dataset
dia_db.head()


dia_db['Outcome'].value_counts()

dia_db.groupby('Outcome').mean()

# separating the data and labels
X = dia_db.drop(columns = 'Outcome', axis=1)
Y = dia_db['Outcome']


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)
classifier = svm.SVC(kernel='linear')
#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)
#SVC(kernel='linear')

# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy)

input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = classifier.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
  
import pickle

filename = 'D:/MTech/Semester VI/MTP/saved models/mt19aie323_diabetes_model.sav'
pickle.dump(classifier, open(filename, 'wb'))



# loading the saved model
loaded_model = pickle.load(open('D:/MTech/Semester VI/MTP/saved models/mt19aie323_diabetes_model.sav', 'rb'))

input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
  
for column in X.columns:
  print(column)
  

  
  