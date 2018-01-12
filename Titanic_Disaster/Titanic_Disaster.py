#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:12:18 2018

@author: dinhthanhan
"""

import pandas as pd
import numpy as np

# get dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
PassengerId = test['PassengerId']

#clean data
full_data = [train, test]

train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    
# Feature selection
    
# drop elements
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
train = train.drop(drop_elements, axis = 1)
test = test.drop(drop_elements, axis = 1)

# encoder categorical element
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_sex = LabelEncoder()
train['Sex'] = labelencoder_sex.fit_transform(train['Sex'])
test['Sex'] = labelencoder_sex.fit_transform(test['Sex'])

labelencoder_embark = LabelEncoder()
train['Embarked'] = labelencoder_embark.fit_transform(train['Embarked'])
test['Embarked'] = labelencoder_embark.fit_transform(test['Embarked'])

train_onehotencoder = OneHotEncoder(categorical_features = [5])
train = train_onehotencoder.fit_transform(train).toarray()
test_onehotencoder = OneHotEncoder(categorical_features = [4])
test = test_onehotencoder.fit_transform(test).toarray()

train = train.drop(['5'], axis = 1)
test = test.drop(['4'], axis = 1)


X_train = train.iloc[:, 1:7].values
y_train = train.iloc[:, 0].values

X_test = test.values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_train, y_train, test_size = 0.3, random_state = 1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu', input_dim = 6))
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = build_classifier('adam')

# Fitting the ANN to the Training Test set
classifier.fit(X_train_2, y_train_2, batch_size = 25, epochs = 500)
# Predicting the Test set results
y_test_2_pred = classifier.predict(X_test_2)
y_test_2_pred = (y_test_2_pred > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_2, y_test_2_pred)

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 25, epochs = 500)

# Predict result
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Generate Submission File 
AnnSubmission = pd.DataFrame({ 'PassengerId': PassengerId, 'Survived':  pd.Series(y_pred.ravel())})
AnnSubmission.to_csv("AnnSubmission.csv", index=False)