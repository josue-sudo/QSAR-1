from keras import models
from keras.layers import Dense, Dropout
from keras.regularizers import l2


def load_model(input_shape):
    model = models.Sequential()

    model.add(Dense(4000, activation='relu', input_shape=input_shape,
                    kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.25))
    model.add(Dense(2000, activation='relu', input_shape=input_shape,
                    kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.25))
    model.add(Dense(1000, activation='relu', input_shape=input_shape,
                    kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.25))
    model.add(Dense(1000, activation='relu', input_shape=input_shape,
                    kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.10))
    model.add(Dense(1, activation=None, use_bias=True,
                    kernel_regularizer=l2(0.0001)))
    return model
