from PIL import Image
from numpy import asarray
from tensorflow import keras

# todo reshape to width, height
def img_to_mat(path):
    image = Image.open(path)
    return asarray(image)

# def load_mnist_data():
#     (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
#     'annotations\\transformed\\annotations_transformed.csv'
#     'annotations\\transformed_normalized\\annotations_transformed_normalized.csv'