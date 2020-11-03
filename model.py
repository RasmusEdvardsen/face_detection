import keras
def model(input_shape, num_classes):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        keras.layers.convolutional.Conv2D(32, 3, activation='relu'),
        keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.convolutional.Conv2D(64, 3, activation='relu'),
        keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.core.Flatten(),
        keras.layers.core.Dropout(0.5),
        keras.layers.core.Dense(num_classes, activation="softmax")
    ])

    model.summary()

m = model(350*450, 1)