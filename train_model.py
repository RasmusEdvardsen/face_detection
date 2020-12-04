import numpy
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense
from keras.callbacks import TensorBoard
from utils import img_to_mat
from sklearn.model_selection import train_test_split
from os import listdir

batch_size = 32
epochs = 10

def model(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=input_shape,activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
    model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=256,kernel_size=(3,3),activation='relu'))
    model.add(Conv2D(filters=256,kernel_size=(3,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Flatten())

    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))

    # Only use dropout when train acc/loss is much better than val acc/loss
    # When training, a percentage of the features are set to zero 
    # When testing, all features are used (and are scaled appropriately). 
    # So the model at test time is more robust - and can lead to higher testing accuracies.
    # model.add(Dropout(rate=0.35))
    model.add(Dense(num_classes,activation='sigmoid'))

    return model

model = model((200, 300, 3), 1)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

path_1 = 'images\\101_ObjectCategories\\Faces\\'
X_1 = np.array([img_to_mat((path_1 + i), (300, 200)) for i in listdir(path_1)], ndmin=4)
y_1 = np.full(X_1.shape[0], 1)

path_2 = 'images\\101_ObjectCategories\\BACKGROUND_Google\\'
X_2 = np.array([img_to_mat((path_2 + i), (300, 200)) for i in listdir(path_2)], ndmin=4)
y_2 = np.full(X_2.shape[0], 0)

X = np.append(X_1, X_2, axis=0)
y = np.append(y_1, y_2, axis=0)

# important to do this, model.fit won't shuffle randomly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", update_freq="batch")
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test), callbacks=[tensorboard_callback])
model.save('models/base_model_v5.h5')
# model evaluate

# Note:
# Keras means the train acc/loss of all batches per epoch.
# val acc/loss is therefore higher than reported train acc/loss