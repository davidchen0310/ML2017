from keras.callbacks import Callback
from keras.utils import np_utils
from keras.models import load_model
import numpy as np
import pandas as pd
import os
import sys

size = 48
num_classes=7
train_file = "train.csv"
unlabeled_data_file = "test.csv"
original_model = "65534.h5"

trained_model = original_model[:-3] + "_trained.h5"
logs_folder = "./" + original_model[:-3] + "_logs"

epochs = 50
batch_size = 64
confidence = 0.75


class History(Callback):
	
	def initialize(self,logs={}):
		self.tr_losses=[]
		self.val_losses=[]
		self.tr_accs=[]
		self.val_accs=[]

	def on_epoch_end(self,epoch,logs={}):
		self.tr_losses.append(logs.get('loss'))
		self.val_losses.append(logs.get('val_loss'))
		self.tr_accs.append(logs.get('acc'))
		self.val_accs.append(logs.get('val_acc'))


def dump_history(store_path,logs):
   	
	if not os.path.exists(store_path):
		os.makedirs(store_path, exist_ok=True)  # to avoid race condition 
	
	with open(os.path.join(store_path,'train_loss'),'w') as f:
		for loss in logs.tr_losses:
			f.write('{}\n'.format(loss))
	with open(os.path.join(store_path,'train_accuracy'),'w') as f:
		for acc in logs.tr_accs:
			f.write('{}\n'.format(acc))
	with open(os.path.join(store_path,'valid_loss'),'w') as f:
		for loss in logs.val_losses:
			f.write('{}\n'.format(loss))
	with open(os.path.join(store_path,'valid_accuracy'),'w') as f:
		for acc in logs.val_accs:
			f.write('{}\n'.format(acc))


def get_data():
	
	data = pd.read_csv(train_file)
	
	x = np.array([np.array(r[1].split(sep=' '), dtype=float) for r in data.values])
	y = np.array(data.values[:, 0], dtype=int)
	
	x = x.reshape((x.shape[0], size, size, 1))
	y = np_utils.to_categorical(y, num_classes=num_classes)
	
	return x[2871:], y[2871:], x[:2871], y[:2871]


def get_unlabeled_data():
	
	data = pd.read_csv(unlabeled_data_file)

	x = np.array([np.array(r[1].split(sep=' '), dtype=float) for r in data.values])

	return x.reshape((x.shape[0], size, size, 1))


X_train, Y_train, X_test, Y_test = get_data()
unlabeled_data = get_unlabeled_data()

model = load_model(original_model)
history = History()
history.initialize();

for index in range(epochs):

	print("epoch ", str(index))

	model.fit(X_train, Y_train,
			  batch_size=batch_size,
			  epochs=1,
			  validation_data=(X_test, Y_test),
			  callbacks=[history])

	probs = model.predict(unlabeled_data)
	predictions = np.argmax(probs, axis=1)  # slower -> model.predict_classes(unlabeled_data)

	selected_indices = []
	for i, p in enumerate(probs):
		if np.max(p) > confidence:
			selected_indices.append(i)
	
	if selected_indices:  # if this list is not empty

		X_train = np.append(X_train, unlabeled_data[selected_indices], axis=0)
		Y_train = np.append(Y_train, np_utils.to_categorical(predictions[selected_indices], num_classes=num_classes), axis=0)

		unlabeled_data = np.delete(unlabeled_data, selected_indices, axis=0)

dump_history(logs_folder, history)

model.save(trained_model)
