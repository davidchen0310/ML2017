import os
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from vis.utils import utils

size = 64
original_path = "faces_all"
processed_path = "faces"


# extract the first 10 faces of each expression
def extract_faces():

	global original_path
	global processed_path
	
	for i, expression_index in enumerate(range(65, 75)):  # A~I
		for image_index in range(10):  # 0~9
			
			original_file_name = chr(expression_index) + "0" + str(image_index) + ".bmp"
			image = misc.imread(os.path.join(original_path, original_file_name))
			
			processed_file_name = str(i * 10 + image_index) + ".bmp"
			with open(os.path.join(processed_path, processed_file_name), "wb") as f:
				misc.imsave(f, image)


def get_flatten_images():
	path = "faces"
	images = np.empty((100, size * size))
	for i in range(100):
		file_name = str(i) + ".bmp"
		image = misc.imread(os.path.join(path, file_name))
		images[i] = image.flatten()
	return images


def center(images):
	mean = images.mean(axis=0, keepdims=True)
	return images - mean


def get_eigenvectors(images, num_eigen_vectors=9):
	U, S, V = np.linalg.svd(images, full_matrices=False)
	return V[:num_eigen_vectors]


def save_faces(file_name, faces, cols=3):
	res = utils.stitch_images(list(faces), cols=cols)
	plt.imsave(file_name, res.reshape((res.shape[0], res.shape[1])), cmap='gray')


def plot_average_face():
	images = get_flatten_images()
	average_face = images.mean(axis=0, keepdims=True).reshape((size, size))
	plt.imsave("average_face.png", average_face, cmap='gray')	


def plot_eigenfaces():
	images = get_flatten_images()
	images = center(images)
	eigenvectors =  get_eigenvectors(images)
	eigenfaces = eigenvectors.reshape((eigenvectors.shape[0], size, size, 1))
	save_faces("eigenfaces.png", eigenfaces, cols=3)


def reduce_and_reconstruct(images, eigenvectors):
	reduced_images = np.dot(center(images), eigenvectors.T)
	return images.mean(axis=0, keepdims=True) + np.dot(reduced_images, eigenvectors)



def plot_original_and_reconstructed_faces():
	images = get_flatten_images()
	eigenvectors = get_eigenvectors(images, num_eigen_faces=5)
	reconstructed_images = reduce_and_reconstruct(images, eigenvectors)
	save_faces("original_faces.png", images.reshape((images.shape[0], size, size, 1)), cols=10)
	save_faces("reconstructed_faces.png", reconstructed_images.reshape((reconstructed_images.shape[0], size, size, 1)), cols=10)


def RMSE(mat1, mat2):
	return np.sqrt(np.average((mat1 - mat2) ** 2))

def plot_reconstruction_error():
	images = get_flatten_images()
	eigenvectors = get_eigenvectors(images, num_eigen_vectors=100)
	errors = np.empty((100,))
	for k in range(1, 101):
		reconstructed_images = reduce_and_reconstruct(images, eigenvectors[:k])
		error = RMSE(images, reconstructed_images) / 256
		print(str(k) + " " + str(error))
		errors[k-1] = error
	print(errors)
	plt.plot(errors)
	plt.show()

plot_reconstruction_error()
# images = get_flatten_images()
# v = get_eigenvectors(images)
# print(v.shape)