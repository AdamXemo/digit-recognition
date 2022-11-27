import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train/255.0
x_test = x_test/255.0

model = keras.models.load_model('model.h5')

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)),
	keras.layers.Dense(128, activation="relu"),
	keras.layers.Dense(128, activation="relu"),
	keras.layers.Dense(10, activation="softmax")
	])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5)

model.save("model.h5")

test_loss, test_acc = model.evaluate(x_test, y_test)

print(f'\nTest accuracy: {round((test_acc*100), 1)}\n')

predictions = model.predict(x_test)

plt.figure(figsize=(5,5))
for i in range(5):
        plt.grid(False)
        plt.imshow(x_test[i], cmap=plt.cm.binary)
        plt.xlabel(f'Number shown: {y_test[i]}')
        plt.title(f'Neural Network predict: {np.argmax(predictions[i])}')
        plt.show()