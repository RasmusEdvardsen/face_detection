from PIL import Image
from numpy import asarray
from tensorflow import keras

WIDTH_HEIGHT = (300, 200)
global WIDTH_HEIGHT

def img_to_mat(path, w_h=None):
    image = Image.open(path).convert('RGB')
    image = image.resize(w_h if w_h is not None else WIDTH_HEIGHT)
    return asarray(image)