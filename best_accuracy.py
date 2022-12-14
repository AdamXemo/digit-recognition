import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, BatchNormalization, Flatten, Dropout
from keras.callbacks import TensorBoard
from sklearn import model_selection
import matplotlib.pyplot as plt
import numpy as np
# Importing tensorflow
import tensorflow as tf
# Importing keras from tensorflow
from tensorflow import keras
# Importing keras dataset for digit recognition
from keras.datasets import mnist
# Importing matplotlib.pyplot to show images of numbers
import matplotlib.pyplot as plt
# Importing numpy just for 1 line of code
import numpy as np

# We loading our data, x_train - value of pixel of our images 
# y_train - labels of our images (what number is shown, from 0 to 9)
# x/y_test is the same, just they are for model testing, not learning
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# We divide value of each pixel by 255
x_train = x_train/255.0
# Because we need value 0 to 1, not 0 to 255
x_test = x_test/255.0

# Loading our model
#model = keras.models.load_model('digit_recognition/model.h5')

# Our Neural Network Model
model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape = (28,28,1)),
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same' ),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same' ),
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same' ),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        
        Flatten(),
          
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(10, activation = "softmax")])

# Compiling our model

# Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.

# Use this crossentropy loss function when there are two or more label classes. We expect labels to be provided as integers.

# Metric [accuracy] creates two local variables, total and count that are used to compute the frequency with which y_pred matches y_true.
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Training our model with train imahes and labels with 5 epochs (nn will train using all images 5  times)
model.fit(x_train, y_train, epochs=5, batch_size=128)

# Saving our model learning progress
model.save("main.h5")
# name it whatever you want but end with .h5

# Getting test accuracy and loss by testing our model with test images and labels
test_loss, test_acc = model.evaluate(x_test, y_test)
# Printing accuracy after testing our model
print(f'\nTest accuracy: {round((test_acc*100), 1)}\n')

# Predictions of our model about number on image
predictions = model.predict(x_test)

print(predictions.shape)
# Changing size of our graph
plt.figure(figsize=(5,5))

# Right predictions
right_pred = 0

# Wrong predictions
wrong_pred = 0

model.save('best_accuracy.h5')

# Looping through every image
for i in range(len(x_test)):
    # Checking if NN prediction is not right
    if np.argmax(predictions[i]) != y_test[i]:
        # Image showing
        # Configure the grid lines
        '''plt.grid(False)
        # Showing image
        plt.imshow(x_test[i], cmap=plt.cm.binary)
        # Writing number label
        plt.xlabel(f'Number shown: {y_test[i]}')
        # Writing Neural Network predict
        plt.title(f'Neural Network predict: {np.argmax(predictions[i])}')
        # Showing all of that stuff
        plt.show()'''
        # Counting wrong predictions
        wrong_pred += 1
    # Checking if right
    else:
        # Counting right predictions
        right_pred += 1

# Printing out amount of right and wrong predictions
print(f'\nNeural network predicted right {right_pred} times, and wrong {wrong_pred} times.\n')




        
