import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_model(input_shape = (256,256,3)) -> keras.Model:
    model = Sequential()
    model.add(Conv2D(16, (3,3),1, activation = 'relu', input_shape=(256,256,3)))
    model.add(MaxPooling2D())

    model.add(Conv2D(32, (3,3),1, activation = 'relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(16, (3,3),1, activation = 'relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))  # Add this
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    return model