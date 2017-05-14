import numpy as np
from sklearn.decomposition import PCA
import macro
from keras.models import load_model
import time
import sys
import os

start_time = time.clock()


data_file_name = sys.argv[1]
num_datasets = 200
res_file = sys.argv[2]


def save_predictions(predictions):
    with open(res_file, "w") as f:
        title = "SetId" + "," + "LogDim" + '\n'
        f.write(title)
        for i, prediction in enumerate(predictions):
            row = str(i) + "," + str(prediction) + '\n'
            f.write(row)

model = load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), macro.model_file_name))

data = np.load(data_file_name)
pca = PCA(svd_solver="full")

features = np.empty((200, macro.max_step + 1))

for i in range(num_datasets):
   
	x = data[str(i)]
	pca.fit(x)
	features[i] = macro.generate_dimension_vector(pca.explained_variance_ratio_, max_step=macro.max_step)
	
model = load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), macro.model_file_name))

predictions = model.predict(features)

save_predictions(np.log(np.squeeze(predictions)))

print(time.clock() - start_time)
