import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
model = keras.models.load_model('model.h5')
test_loss, test_acc = model.evaluate(x_test, y_test)

# Raw NN, loading model, without image showing, saving learning progress and so on. BUT only 6 lines.