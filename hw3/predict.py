import sys
from keras.models import load_model
import pandas as pd
import numpy as np

test_file = sys.argv[1]
model_file = "65534.h5"
res_file = sys.argv[2]


def standardize(x):
	x -= np.mean(x, axis=0)
	x /= np.std(x, axis=0)
	return x


def save_labels(labels):
    with open(res_file, "w") as f:
        title = "id" + "," + "label" + '\n'
        f.write(title)
        for i, l in enumerate(labels):
            row = str(i) + "," + str(l) + '\n'
            f.write(row)

data = pd.read_csv(test_file, header=None, skiprows=1, sep=' |,', engine='python')

x_test = np.array(data.values[:, 1:], dtype=float).reshape((data.shape[0], 48, 48, 1))
x_test = standardize(x_test)

model = load_model(model_file)

labels = np.argmax(model.predict(x_test), axis=1)

save_labels(labels)

# model = load_model(model_file)
# model.summary()
