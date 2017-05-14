import numpy as np
from skimage.measure import block_reduce
from scipy import misc
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import macro
from keras.models import load_model

def get_images():
	
	path = "hand"
	num_images = 481
	height = 480
	width = 512
	images = np.empty((num_images, height, width))
	
	for i in range(num_images):
		file_name = "hand.seq" + str(i+1) + ".png"
		images[i] = misc.imread(os.path.join(path, file_name))
		
	return images


def downsample(images, ratio):
	return block_reduce(images, block_size=(1, ratio, ratio), func=np.mean)


def get_num_dimension(dataset):
	
	pca = PCA(svd_solver="full")
	pca.fit(dataset)
	feature = macro.generate_dimension_vector(pca.explained_variance_ratio_, max_step=macro.max_step)
	feature = feature[np.newaxis, :]

	model = load_model(macro.model_file_name)
	return model.predict(feature)

n = 6
res = np.empty((6))
for i in range(n):
	images = get_images()
	images = downsample(images, 2 ** i)
	predicted_num_dimension = get_num_dimension(images.reshape((images.shape[0], -1)))
	res[i] = predicted_num_dimension

print(res)
