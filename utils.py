from PIL import Image
from numpy import asarray
from tensorflow import keras

# todo reshape to width, height
def img_to_mat(path, w_h=None):
    image = Image.open(path).convert('RGB')
    if w_h:
        image = image.resize(w_h)
    return asarray(image)