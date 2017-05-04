from keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import sys
import pandas as pd
import numpy as np
import itertools
import csv

size = 48
model_file = "65534.h5"
data_file = "train.csv"
error_index_file = "error_index.txt"
no_validation_elements = 2871
classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

def get_features():
	data = pd.read_csv(data_file)
	x = np.array([np.array(r[1].split(sep=' '), dtype=float) for r in data.values])
	return x.reshape(x.shape[0], size, size, 1)[:no_validation_elements]
	

def get_answers():
	data = pd.read_csv(data_file)
	return np.array(data.values[:,0], dtype=int)[:no_validation_elements]


def get_predictions(x):
	model = load_model(model_file)
	return model.predict_classes(x)[:no_validation_elements]


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def save_prediction_error_index(ys, y_hats):
	
	tuples = []
	for i, (y, y_hat) in enumerate(zip(ys, y_hats)):
		if y != y_hat:
			tuples.append((i, classes[y], classes[y_hat]))
	
	with open(error_index_file, "w") as the_file:
	    csv.register_dialect("custom", delimiter=" ", skipinitialspace=True)
	    writer = csv.writer(the_file, dialect="custom")
	    for tup in tuples:
	        writer.writerow(tup)

features = get_features()

predictions = get_predictions(features)
answers = get_answers()

conf_mat = confusion_matrix(predictions, answers)

plot_confusion_matrix(conf_mat, classes=classes)
plt.show()

save_prediction_error_index(predictions, answers)
