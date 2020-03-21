from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# load dataset
dataset = pd.read_csv("diabetes.csv", header=None).values

# Split the dataset into train and test data
X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,0:8], dataset[:,8],
                                                    test_size=0.25, random_state=87)
np.random.seed(155)
my_first_nn = Sequential() # create model
my_first_nn.add(Dense(20, input_dim=8, activation='relu')) # hidden layer
my_first_nn.add(Dense(30, )) # hidden layer
my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam')
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100, verbose=0,
                                     initial_epoch=0)O
print(my_first_nn.summary())
print(my_first_nn.evaluate(X_test, Y_test, verbose=0))