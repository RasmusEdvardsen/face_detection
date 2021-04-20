import numpy as np
from glob import glob
from model import model
from utils import get_data_from_list
from sklearn.model_selection import train_test_split

import wandb
from wandb.keras import WandbCallback
wandb.init(project="tech-talk")

batch_size=32
epochs=3

if __name__ == '__main__':
    # Get data
    (X_1, y_1) = get_data_from_list(glob('images/positive/*'), 1)
    (X_2, y_2) = get_data_from_list(glob('images/negative/**/*.*'), 0)
    X = np.append(X_1, X_2, axis=0)
    y = np.append(y_1, y_2, axis=0)

    # important to do this, model.fit won't shuffle randomly
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)

    # Init model structure
    model = model((200, 300, 3), 1)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks=[WandbCallback(log_batch_frequency=10)])
    model.save('models/base_model_v13.h5')
    
    # Note:
    # Keras aggregates the train acc/loss of all batches per epoch.
    # val acc/loss is therefore better than reported train acc/loss