#!C:\Users\asus\Anaconda3\envs\DL\python.exe
import pandas as pd
import numpy as np
import sys
from keras.utils import np_utils
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
import time

start_time = time.clock()

train_file = sys.argv[1]
size = 48
num_classes = 7

batch_size = 64
epochs = 1000
dropout_rate = 0.2

steps_per_epoch = 400  # int(28710 / split / batch_size)


def standardize(x):
	x -= np.mean(x, axis=0)
	x /= np.std(x, axis=0)
	return x


data = pd.read_csv(train_file)

x = np.array([np.array(r[1].split(sep=' '), dtype=float) for r in data.values])
y = np.array(data.values[:, 0], dtype=int)

# preprocess
x = standardize(x)

x = x.reshape((x.shape[0], size, size, 1))
y = np_utils.to_categorical(y)

model = Sequential()

model.add(Conv2D(64, (5, 5), padding='valid', input_shape=(size, size, 1)))
model.add(PReLU(alpha_initializer='zero', weights=None))

model.add(ZeroPadding2D(padding=(2, 2)))
model.add(MaxPooling2D(pool_size=(5, 5),strides=(2, 2)))

model.add(ZeroPadding2D(padding=(1, 1))) 
model.add(Conv2D(64, (3, 3)))
model.add(PReLU(alpha_initializer='zero', weights=None))

model.add(ZeroPadding2D(padding=(1, 1))) 
model.add(Conv2D(64, (3, 3)))
model.add(PReLU(alpha_initializer='zero', weights=None))

model.add(AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))

model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Conv2D(128, (3, 3)))
model.add(PReLU(alpha_initializer='zero', weights=None))

model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Conv2D(128, (3, 3)))
model.add(PReLU(alpha_initializer='zero', weights=None))

model.add(ZeroPadding2D(padding=(1, 1)))
model.add(AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))

model.add(Flatten())

model.add(Dense(1024))
model.add(PReLU(alpha_initializer='zero', weights=None))
model.add(Dropout(dropout_rate))

model.add(Dense(1024))
model.add(PReLU(alpha_initializer='zero', weights=None))
model.add(Dropout(dropout_rate))

model.add(Dense(7, activation='softmax'))

ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=ada, metrics=['accuracy'])

# model.summary()

X_train = x
Y_train = y

datagen = ImageDataGenerator(
   	featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(X_train)	

model.fit_generator(datagen.flow(X_train, Y_train,
					batch_size=batch_size),
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs)

end_time = time.clock() - start_time
print(end_time)
