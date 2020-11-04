import keras
import numpy
from os import listdir
from PIL import Image
import numpy as np
from utils import img_to_mat
from sklearn.model_selection import train_test_split
import pandas as pd

batch_size = 128
epochs = 5

def model(input_shape, num_classes):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        keras.layers.experimental.preprocessing.Rescaling(scale=1./255),
        keras.layers.convolutional.Conv2D(8, 3, padding="same", activation='relu'),
        keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.convolutional.Conv2D(8, 3, padding="same", activation='relu'),
        keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.core.Flatten(),
        keras.layers.core.Dropout(0.5),
        keras.layers.core.Dense(num_classes, activation="softmax")
    ])

    model.summary()
    return model

model = model((450, 350, 3), 1)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

img_path = 'images\\normalized\\'
X = np.array([img_to_mat(img_path + i) for i in listdir(img_path)], ndmin=4)
label_path = 'annotations\\transformed_normalized\\'
y = pd.read_csv(label_path + 'annotations_transformed_normalized.csv')
y = y.set_index('id')
y = y['confidence']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# print(X_train, X_train[0])

model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# model evaluate