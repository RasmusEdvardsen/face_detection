import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from model import model
from keras.callbacks import TensorBoard
from utils import img_to_mat, get_data_from_directory
from sklearn.model_selection import train_test_split

batch_size=32
epochs=10

if __name__ == '__main__':
    # Get args
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('--positives_path', type=str, help='input path(s) for positive images', required=True)
    parser.add_argument('--negatives_path', type=str, help='input path(s) for negative images', required=True)
    args = parser.parse_args()
    
    # Init model structure
    model = model((200, 300, 3), 1)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Get data
    (X_1, y_1) = get_data_from_directory(args.positives_path)
    (X_2, y_2) = get_data_from_directory(args.negatives_path)
    X = np.append(X_1, X_2, axis=0)
    y = np.append(y_1, y_2, axis=0)

    # important to do this, model.fit won't shuffle randomly
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Add metrics callback, train model and save on complete
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", update_freq="batch")
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test), callbacks=[tensorboard_callback])
    model.save('models/base_model_v7.h5')
    # model evaluate

    # Note:
    # Keras aggregates the train acc/loss of all batches per epoch.
    # val acc/loss is therefore higher than reported train acc/loss