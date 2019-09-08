
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv(r"C:\Users\employee data.csv")

#dataset['experience'].fillna(0, inplace=True)

#dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

X = dataset.iloc[:, :-1]

#Converting words to integer values
def convert_to_int(word):
    word_dict = {'A':1, 'B':2, 'AB':3,'O':4}
    return word_dict[word]

X['groups'] = X['groups'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor123 = LinearRegression()

#Fitting model with trainig data
regressor123.fit(X, y)

# Saving model to disk
pickle.dump(regressor123, open('amartya.pkl','wb'))


# Loading model to compare the results
model = pickle.load(open('amartya.pkl','rb'))
model

