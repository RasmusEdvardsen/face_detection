from PIL import Image
from tensorflow import keras
from os import listdir
import numpy as np

global WIDTH_HEIGHT
WIDTH_HEIGHT = (300, 200)

def img_to_mat(path, w_h=None):
    image = Image.open(path).convert('RGB')
    image = image.resize(w_h if w_h is not None else WIDTH_HEIGHT)
    return np.asarray(image)

def get_data_from_directory(path):
    X_1 = np.array([img_to_mat((path + i), (300, 200)) for i in listdir(path)], ndmin=4)
    y_1 = np.full(X_1.shape[0], 1)
    return (X_1, y_1)