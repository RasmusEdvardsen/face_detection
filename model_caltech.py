import keras
import numpy
from os import listdir
from PIL import Image
import numpy as np
from utils import img_to_mat
from sklearn.model_selection import train_test_split
import pandas as pd

batch_size = 64
epochs = 2

def model(input_shape, num_classes):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        keras.layers.experimental.preprocessing.Rescaling(scale=1./255),
        keras.layers.convolutional.Conv2D(64, 3, padding="same", activation='relu'),
        keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.convolutional.Conv2D(32, 3, padding="same", activation='relu'),
        keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.core.Flatten(),
        keras.layers.core.Dropout(0.5),
        keras.layers.core.Dense(num_classes, activation="softmax")
    ])

    model.summary()
    return model

model = model((200, 300, 3), 2)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

path_1 = 'images\\101_ObjectCategories\\Faces\\'
X_1 = np.array([img_to_mat((path_1 + i), (300, 200)) for i in listdir(path_1)], ndmin=4)
y_1 = np.full(X_1.shape[0], 1)

path_2 = 'images\\101_ObjectCategories\\BACKGROUND_Google\\'
X_2 = np.array([img_to_mat((path_2 + i), (300, 200)) for i in listdir(path_2)], ndmin=4)
y_2 = np.full(X_2.shape[0], 0)

X = np.append(X_1, X_2, axis=0)
y = np.append(y_1, y_2, axis=0)
y = keras.utils.to_categorical(y, 2)

print(X.shape, y.shape)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.3)

# model evaluate