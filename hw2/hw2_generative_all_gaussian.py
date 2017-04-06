import numpy as np
import sys
# import itertools
# import time

train_file = sys.argv[1]  # "X_train.csv"
train_Y_file = sys.argv[2]  # "Y_train.csv"
test_file = sys.argv[3]  # "X_test.csv"
res_file = sys.argv[4]  #"res.txt"

ordinal_indices = [1,5,6,7,8,9,13]
numerical_indices = [0,2,4,10,11,12]

ordinal_indices = [5, 6]
numerical_indices = [4, 10, 11, 12]

numerical_indices = list(range(106))

# rlr, tol = 1e-3, selection_threshold = 0.5
# numerical_indices = [0, 2, 3, 4, 5, 6, 10, 11, 15, 16, 19, 20, 21, 24, 25, 26, 27, 29, 33, 35, 41, 42, 44, 45, 47, 49, 50, 56, 58, 63, 89, 102]

# rlr, tol = 1e-4, selection_threshold = 0.6
numerical_indices = [0, 3, 4, 5, 6, 10, 15, 20, 21, 24, 25, 26, 27, 29, 33, 35, 41, 42, 45, 47, 49, 50, 58]

# numerical_indices = range(14)

class Accuracy:
	pass

def get_train_Y():
	return np.loadtxt(train_Y_file)

def calculate_prior():
	y = get_train_Y()
	class_1_prior = np.sum(y) / len(y)
	class_0_prior = 1 - class_1_prior
	return np.array([class_0_prior, class_1_prior])

def get_ordinal_data():
	class_0 = []
	class_1 = []

	ys = get_train_Y()
	xs = np.genfromtxt(train_file, delimiter=',', usecols=ordinal_indices, dtype=str)

	for x, y in zip(xs, ys):
		if y == 0:
			class_0.append(x)
		else:
			class_1.append(x)

	return (np.array(class_0, dtype=str), np.array(class_1, dtype=str))

def get_frequencies(ordinal_data_tuple):
	list_of_dics = []
	for ordinal_data in ordinal_data_tuple:
		dics = [dict() for x in range(len(ordinal_data[0]))]
		for row in ordinal_data:
			for i, e in enumerate(row):  # e for element
				if e != " ?":
					dics[i][e] = dics[i].get(e, 0) + 1
		
		list_of_dics.append(dics)
	
	return tuple(list_of_dics)

def get_probabilities(frequencies_tuple):
	list_of_dics = []

	for frequencies in frequencies_tuple:
		dics = [dict() for x in range(len(frequencies))]
		for i, frequency in enumerate(frequencies):
			frequency_sum = np.sum(list(frequency.values()))
			for k in frequency.keys():
				dics[i][k] = frequency[k] / frequency_sum
		
		list_of_dics.append(dics)
	
	return tuple(list_of_dics)

def get_numerical_data():
	class_0 = []
	class_1 = []

	ys = get_train_Y()
	xs = np.loadtxt(train_file, delimiter=',', usecols=numerical_indices, skiprows=1, dtype=float)

	for x, y in zip(xs, ys):
		if y == 0:
			class_0.append(x)
		else:
			class_1.append(x)

	return (np.array(class_0), np.array(class_1))	


def MLE(numerical_data_tuple, prior_probs):
	mus = []
	sigmas = []

	for numerical_data in numerical_data_tuple:
		mus.append(np.mean(numerical_data.T, axis=1))
		sigmas.append(np.cov(numerical_data.T))

	sigma_product = [sigma * prior_prob for sigma, prior_prob in zip(sigmas, prior_probs)]
	sigma_sum = np.array(sigma_product[0]) + np.array(sigma_product[1])
	return (mus, sigma_sum)

def test_mean(i, class_id):
	ys = get_train_Y()
	xs = np.loadtxt(train_file, delimiter=',', usecols=[i], dtype=float)

	return np.mean([x for x, y in zip(xs, ys) if y == class_id])


def get_test_ordinal_data():
	return np.genfromtxt(test_file, delimiter=',', usecols=ordinal_indices, skip_header=1, dtype=str)


def calculate_binomial_probs(test_ordinal_data, probabilities):
	list_of_products = []
	for i in range(len(probabilities)):
		products = np.empty((len(test_ordinal_data)))
		for j, row in enumerate(test_ordinal_data):
			product = 1
			for k, e in enumerate(row):
				if e != ' ?':
					try:
						product = product * probabilities[i][k][e]
					except:
						product = 0  # some poor people
						break;
			products[j] = product
		list_of_products.append(products)
	return np.array(list_of_products)


def get_test_numerical_data():
	return np.loadtxt(test_file, delimiter=',', usecols=numerical_indices, skiprows=1, dtype=float)


def calculate_gaussian_probs(test_numerical_data, mus, sigma):
	list_of_probs = []
	for mu in mus:
		probs = np.empty((len(test_numerical_data)))
		for i, row in enumerate(test_numerical_data):
			denominator = ((2 * np.pi) ** (len(row) / 2)) * np.sqrt(np.linalg.det(sigma))
			numerator = np.e ** (((-1) / 2) * np.dot(np.dot(row - mu, np.linalg.inv(sigma)), row - mu))
			try:
				probs[i] = numerator / denominator
			except:
				print(numerator)
				print(denominator)
				print(probs[i].shape)
				exit()
		list_of_probs.append(probs)		
	return np.array(list_of_probs)


def calculate_likelihood(binomial_probs, gaussian_probs):
	return np.array([b * g for b, g in zip(binomial_probs, gaussian_probs)])


def calculate_posterior(prior_probs, likelihood_probs, class_id):
	n = len(likelihood_probs[0])
	posterior_probs = np.empty((n))
	for i in range(n):
		marginal_likelihood = np.sum([likelihood_probs[j][i] * prior_probs[j] for j in range(len(prior_probs))])
		if marginal_likelihood != 0:
			posterior_probs[i] = likelihood_probs[class_id][i] * prior_probs[class_id] / marginal_likelihood
		else:
			posterior_probs[i] = 0

	return posterior_probs


def transform(posterior_probs):
	res = np.empty(len(posterior_probs), dtype=int)
	for i, posterior_prob in enumerate(posterior_probs):
		if posterior_prob > 0.5:
			res[i] = 1
		else:
			res[i] = 0
	return res


def save_labels(labels):
    with open(res_file, "w") as f:
        title = "id" + "," + "label" + '\n'
        f.write(title)
        for i, l in enumerate(labels):
        	row = str(i+1) + "," + str(l) + '\n'
        	f.write(row)


def get_train_accuracy(prior_probs, mu, sigma):

	# test_ordinal_data = np.genfromtxt(train_file, delimiter=',', usecols=ordinal_indices, dtype=str)
	# binomial_probs = calculate_binomial_probs(test_ordinal_data, probabilities)

	test_numerical_data = np.loadtxt(train_file, delimiter=',', usecols=numerical_indices, skiprows=1, dtype=float)
	gaussian_probs = calculate_gaussian_probs(test_numerical_data, mu, sigma)

	likelihood_probs = gaussian_probs
	posterior_probs = calculate_posterior(prior_probs, likelihood_probs, 1)

	labels = transform(posterior_probs)

	return (np.sum([1 for l, y in zip(labels, get_train_Y()) if l == y]) / len(labels))


def print_probabilities_in_each_category(frequencies):
	for f1, f2 in zip(frequencies[0], frequencies[1]):
		for k in f1.keys():
			print(k, f1.get(k, 0) / f2.get(k, 1))


def print_accuracies(accuracies):
    for accuracy in accuracies:
        print(accuracy.features, accuracy.rate)

# start_time = time.clock()

# # test data correlation, 6840 combinations, 2017/3/27 13:00

# prior_probs = calculate_prior()

# original_ordinal_indices = ordinal_indices
# original_numerical_indices = numerical_indices

# accuracies = []

# for ordinal_length in range(2, len(original_ordinal_indices) + 1):

# 	ordinal_combinations = itertools.combinations(original_ordinal_indices, ordinal_length)

# 	for ordinal_c in ordinal_combinations:

# 		ordinal_indices = list(ordinal_c)

# 		ordinal_data = get_ordinal_data()
# 		frequencies = get_frequencies(ordinal_data)
# 		probabilities = get_probabilities(frequencies)

# 		for numerical_length in range(2, len(original_numerical_indices) + 1):

# 			numerical_combinations = itertools.combinations(original_numerical_indices, numerical_length)

# 			for numerical_c in numerical_combinations:

# 				numerical_indices = list(numerical_c)

# 				numerical_data = get_numerical_data()
# 				mu, sigma = MLE(numerical_data, prior_probs)  # maximum likelihood estimation

# 				accuracy = Accuracy()
# 				accuracy.features = ordinal_indices + numerical_indices
# 				accuracy.rate = get_train_accuracy(prior_probs, mu, sigma, probabilities)
# 				accuracies.append(accuracy)

# print_accuracies(sorted(accuracies, key = lambda accuracy: accuracy.rate))

# test ordinal data correlation

# prior_probs = calculate_prior()

# numerical_data = get_numerical_data()
# mu, sigma = MLE(numerical_data, prior_probs)  # maximum likelihood estimation

# original_ordinal_indices = ordinal_indices

# accuracies = []

# for length in range(2, len(original_ordinal_indices) + 1):

# 	combinations = itertools.combinations(original_ordinal_indices, length)

# 	for c in combinations:
# 		ordinal_indices = list(c)

# 		ordinal_data = get_ordinal_data()
# 		frequencies = get_frequencies(ordinal_data)
# 		probabilities = get_probabilities(frequencies)

# 		accuracy = Accuracy()
# 		accuracy.features = ordinal_indices
# 		accuracy.rate = get_train_accuracy(prior_probs, mu, sigma, probabilities)
# 		accuracies.append(accuracy)

# print_accuracies(sorted(accuracies, key = lambda accuracy: accuracy.rate))

# test numerical data correlation

# prior_probs = calculate_prior()

# ordinal_data = get_ordinal_data()
# frequencies = get_frequencies(ordinal_data)
# probabilities = get_probabilities(frequencies)

# original_numerical_indices = numerical_indices

# accuracies = []

# for length in range(2, len(original_numerical_indices) + 1):

# 	combinations = itertools.combinations(original_numerical_indices, length)

# 	for c in combinations:

# 		numerical_indices = list(c)

# 		numerical_data = get_numerical_data()
# 		mu, sigma = MLE(numerical_data, prior_probs)  # maximum likelihood estimation

# 		accuracy = Accuracy()
# 		accuracy.features = numerical_indices
# 		accuracy.rate = get_train_accuracy(prior_probs, mu, sigma, probabilities)
# 		accuracies.append(accuracy)

# print_accuracies(sorted(accuracies, key = lambda accuracy: accuracy.rate))

prior_probs = calculate_prior()

# ordinal_data = get_ordinal_data()
# frequencies = get_frequencies(ordinal_data)
# probabilities = get_probabilities(frequencies)

numerical_data = get_numerical_data()
mu, sigma = MLE(numerical_data, prior_probs)  # maximum likelihood estimation

# test_ordinal_data = get_test_ordinal_data()
# binomial_probs = calculate_binomial_probs(test_ordinal_data, probabilities)

test_numerical_data = get_test_numerical_data()
gaussian_probs = calculate_gaussian_probs(test_numerical_data, mu, sigma)

likelihood_probs = gaussian_probs
posterior_probs = calculate_posterior(prior_probs, likelihood_probs, 1)

labels = transform(posterior_probs)

save_labels(labels)

# print_probabilities_in_each_category(frequencies)

print(get_train_accuracy(prior_probs, mu, sigma))

# print(time.clock() - start_time)
