import numpy as np
import macro
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adadelta
from keras.models import load_model
import sys
import time

start_time = time.clock()

batch_size = 64
epochs = 1000
dropout_rate = 0.2


class Parameters:
	pass


def loss(y_true, y_pred):
	return np.abs(np.log(y_true) - np.log(y_pred))


def get_data(split=0.1):
	
	data = np.loadtxt(macro.training_data_file_name)

	np.random.shuffle(data)

	features = data[:, 1:]
	labels = data[:, 0].reshape((data.shape[0]))

	watershed = int(data.shape[0] * split)

	return features[watershed:], labels[watershed:], features[:watershed], labels[:watershed]


def get_model():

	if len(sys.argv) > 1:
		return load_model(sys.argv[1])

	model = Sequential()

	model.add(Dense(32, input_shape=(6,)))
	model.add(PReLU(alpha_initializer='zero', weights=None))
	model.add(Dropout(dropout_rate))
	
	model.add(Dense(64))
	model.add(PReLU(alpha_initializer='zero', weights=None))
	model.add(Dropout(dropout_rate))
	
	model.add(Dense(1, activation='linear'))
		
	ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
	model.compile(loss="mean_squared_logarithmic_error", optimizer=ada, metrics=['msle'])
	
	return model

	
X_train, Y_train, X_test, Y_test = get_data()

model = get_model()
model.summary()

model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, Y_test))

model.save(macro.model_file_name)

print(time.clock() - start_time)
