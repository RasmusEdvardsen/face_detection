import keras
import numpy
from os import listdir
from PIL import Image
import numpy as np
from utils import img_to_mat
from sklearn.model_selection import train_test_split
import pandas as pd

batch_size = 64
epochs = 5

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

model = model((32, 32, 3), 2)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

img_path = 'images\\makeml\\normalized\\'
X = np.array([img_to_mat(img_path + i) for i in listdir(img_path)], ndmin=4)

label_path = 'annotations\\makeml\\transformed_normalized\\'
y = pd.read_csv(label_path + 'annotations_transformed_normalized.csv')
# y = y.set_index('id')
y = y['confidence']

print(X.shape, y.shape)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train[0:500]
y_train = y_train[0:500]
y_train[:,] = 0
x_test = x_test[0:500]
y_test = y_test[0:500]
y_test[:,] = 0

print(x_train.shape, y_train.shape)

X = np.append(X, x_train, axis=0)
X = np.append(X, x_test, axis=0)

y = y.append(pd.DataFrame(y_train))
y = y.append(pd.DataFrame(y_test))

y = keras.utils.to_categorical(y, 2)

print(X.shape, y.shape)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.3)

# model evaluate