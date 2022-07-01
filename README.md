import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_train.reshape(-1, 28*28).astype("float32") / 255.0
print(x_train.shape)

