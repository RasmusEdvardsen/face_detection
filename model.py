from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense

def model(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=input_shape,activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dropout(rate=0.5))
    model.add(Dense(num_classes,activation='sigmoid'))

    return model