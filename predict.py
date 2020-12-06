import argparse
import tensorflow as tf
import keras
import numpy as np
from utils import img_to_mat

if __name__ == '__main__':
    # Get args
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    args = parser.parse_args()

    # Careful of \\ slashes

    model = keras.models.load_model(args.model_path)
    image = np.array(img_to_mat(args.image_path), ndmin=4)

    pred = model.predict(image)
    print('Chances of the given image matching the pattern model was trained for:\n', pred[0])