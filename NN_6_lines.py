import tensorflow as tf 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
model = tf.keras.models.load_model('model.h5')
test_loss, test_acc = model.evaluate(x_test, y_test)

# Raw NN, loading model, without image showing, saving learning progress and so on. BUT only 4 lines.