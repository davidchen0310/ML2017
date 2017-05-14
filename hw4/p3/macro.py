import numpy as np

num_dataset_files = 2
base_dataset_file_name = "training_dataset"
training_data_file_name = "train.txt"
max_step = 5
model_file_name = "model_2000.h5"


def generate_dimension_vector(eigenvalues, lower_bound=0.70, max_step=5, step_size=0.05):

	dimensions = np.empty((max_step + 1))
	current_step = 0
	cumulative_eigenvalue = 0

	for i, eigenvalue in enumerate(eigenvalues):
		cumulative_eigenvalue += eigenvalue
		while current_step <= max_step:
			if cumulative_eigenvalue >= lower_bound + current_step * step_size:
				dimensions[current_step] = i + 1
				current_step += 1
			else:
				break;
		if current_step > max_step:
			break;

	return dimensions


def preprocess(features):
	
	features = np.c_[features, np.sum(features, axis=1).reshape((features.shape[0], 1))]
	# features = np.c_[np.ones((features.shape[0], 1)), features]  # intercept

	return features


def predict(features, w):
	res = np.dot(features, w)
	return np.where(res >= 1, res, 1)
