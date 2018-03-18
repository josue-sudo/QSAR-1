from keras import models
import keras.backend as K
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.regularizers import l2


def r2(x, y):

    x = K.batch_flatten(x)
    y = K.batch_flatten(y)

    avx = K.mean(x)
    avy = K.mean(y)

    num = K.sum((x-avx) * (y-avy))
    num = num * num

    denom = K.sum((x-avx)*(x-avx)) * K.sum((y-avy)*(y-avy))

    return num / denom


def generate_model(input_shape):
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
    optimizer = SGD(lr=0.05, momentum=0.9, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mean_squared_error',
                  metrics=[r2])
    return model
