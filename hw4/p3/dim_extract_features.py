import numpy as np
from sklearn.decomposition import PCA
import macro
import time

start_time = time.clock()


features_all = []
labels_all = []

for file_index in range(macro.num_dataset_files):
	
	file_name = macro.base_dataset_file_name + "_" + str(file_index) + ".npz"
	
	data = np.load(file_name)
	datasets = data["datasets"]
	labels = data["labels"]

	pca = PCA(svd_solver="full")

	num_datasets = len(labels)
	features = np.empty((num_datasets, macro.max_step + 1))

	for i in range(num_datasets):
	   
		x = datasets[i]
		pca.fit(x)
		features[i] = macro.generate_dimension_vector(pca.explained_variance_ratio_, max_step=macro.max_step)
		
	features_all.append(features)
	labels_all.append(labels)

features_all = np.concatenate(features_all, axis=0)
labels_all = np.concatenate(labels_all, axis=0)

res = np.c_[labels_all, features_all]

np.savetxt(macro.training_data_file_name, res, fmt='%d')

print(time.clock() - start_time)
