import numpy as np
import tensorflow as tf
import keras


model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

#Define the loss function and the optimizer
#sgd = stocastic gradient descent
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')

#Specify the x's and y's, it must be a numpy array
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype = float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype = float)


#Define the relationship (fitting) and do 5000 epochs
model.fit(xs, ys, epochs = 5000)


print(model.predict([10.0]))