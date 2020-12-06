from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense

def model(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=input_shape,activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
    model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=256,kernel_size=(3,3),activation='relu'))
    model.add(Conv2D(filters=256,kernel_size=(3,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Flatten())

    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))

    # Only use dropout when train acc/loss is much better than val acc/loss
    # When training, a percentage of the features are set to zero 
    # When testing, all features are used (and are scaled appropriately). 
    # So the model at test time is more robust - and can lead to higher testing accuracies.
    model.add(Dropout(rate=0.25))
    model.add(Dense(num_classes,activation='sigmoid'))

    return model