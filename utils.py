from PIL import Image
from numpy import asarray

# todo reshape to width, height
def img_to_mat(path):
    image = Image.open(path)
    return asarray(image)