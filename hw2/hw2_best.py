import numpy as np
import math
# import time
import pandas as pd
# from sklearn.linear_model import RandomizedLasso
# from sklearn.model_selection import train_test_split
import sys

training_data_file = sys.argv[1]  # "X_train.csv"
training_targets_file = sys.argv[2]  # "Y_train.csv"
test_data_file = sys.argv[3]  # "X_test.csv"
res_file = sys.argv[4]  # "res.csv"

use_colons = list(range(106))

class Seed:
    pass

# tol = 1e-3, selection_threshold = 0.5
# use_colons = [0, 2, 3, 4, 5, 6, 10, 11, 15, 16, 19, 20, 21, 24, 25, 26, 27, 29, 33, 35, 41, 42, 44, 45, 47, 49, 50, 56, 58, 63, 89, 102]

# tol = 1e-4
# use_colons = [0, 2, 3, 4, 5, 6, 10, 11, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 32, 33, 35, 41, 42, 43, 44, 45, 47, 48, 49, 50, 52, 55, 56, 58, 63, 89, 102]

# tol = 1e-3
# use_colons = [0, 2, 3, 4, 5, 6, 10, 11, 14, 15, 16, 18, 19, 20, 21, 23, 24, 25, 26, 27, 29, 32, 33, 35, 41, 42, 43, 44, 45, 47, 48, 49, 50, 52, 53, 56, 58, 63, 89, 102]

# use_colons = use_colons + [33, 53, 0, 5, 3, 2, 41, 47, 24, 27, 29, 4, 10, 25, 58, 63] * 6

# use_colons = [33, 53, 0, 5, 3, 2, 41, 47, 24, 27, 29, 4, 10, 25, 43, 61, 38, 31, 26, 57, 45, 54, 56, 35]

# incomplete_columns = [14, 52, 105]

def standardize(mat):
    res = mat.T
    n = len(res)
    # n = 6
    for i in range(n):
        try:
            res[i] = (res[i] - np.mean(res[i])) / np.std(res[i]) 
        except:
            res[i] = res[i] - np.mean(res[i])
    return res.T


def normalize(mat):
    res = mat.T
    n = len(res)
    for i in range(n):
        if max(res[i]) != min(res[i]):
            res[i] = (res[i] - min(res[i])) / (max(res[i]) - min(res[i]))
    return res.T


def adjust(mat):
    features = mat.T
    # numerical_indices = [0,1,3,4,5]
    features = np.insert(features, len(features), features[0] ** 2, axis=0)
    features = np.insert(features, len(features), np.log(features[1]), axis=0)
    features = np.insert(features, len(features), np.sqrt(features[3]), axis=0)
    features = np.insert(features, len(features), np.sqrt(features[4]), axis=0)
    features = np.insert(features, len(features), features[5] ** 2, axis=0)
    return features.T

def get_mu_and_sd(mat):
    mat = mat.T
    mu = np.empty((len(mat)))
    sd = np.empty((len(mat)))
    for i in range(len(mat)):
        mu[i] = np.mean(mat[i])
        sd[i] = np.std(mat[i])
    return (mu, sd)


def normalize_depend_on_training_data(mat, mu, sd):
    res = mat.T
    for i in range(len(res)):
        try:
            res[i] = (res[i] - mu[i]) / sd[i] 
        except:
            res[i] = res[i] - mu[i]
    return res.T


def get_training_data():
    data = np.loadtxt(training_data_file, delimiter=',', skiprows=1, usecols=use_colons, dtype=float)
    data = adjust(data)
    data = normalize(data)
    return np.insert(data, 0, np.ones((len(data))), axis=1)  # insert bias


def get_training_targets():
    return np.loadtxt(training_targets_file, delimiter=',', dtype=float)


def sigmoid(z):
    res = (1 / (1 + np.exp(-z)))
    return np.clip(res, 1e-15, 1 - 1e-15)


def train(data, targets):
    
    w = np.zeros(len(data[0])).flatten() 
    lr = np.ones(len(data[0])).flatten()  # learning rate
    _lambda = 0
    batch_size = 1000
    iteration = 3000

    # lr = np.random.rand(len(data[0]))

    grad_sum = np.full(len(data[0]), 1e-30).flatten()
    # grad_sum = np.zeros(len(data[0])).flatten()

    for i in range(iteration):
        
        bound = math.ceil(len(data) / batch_size)
        
        for j in range(bound):

            start_index = j * batch_size
            end_index = (j + 1) * batch_size

            if j == bound - 1:
                x = data[start_index:]
                y_hat = targets[start_index:]
            else:
                x = data[start_index:end_index]
                y_hat = targets[start_index:end_index]  

            z = np.dot(x, w)
            y = sigmoid(z)
            grad = np.dot(x.T, y - y_hat) + 2 * _lambda * w

            # print(j, grad)

            grad_sum += grad ** 2
            ada = np.sqrt(grad_sum)
            w = w - lr / ada * grad           

    return w


def get_training_accuracy(w, x, y_hat):
    y = predict(x, w)
    return calculate_accuracy(y, y_hat)


def predict(x, w):
    z = np.dot(x, w)
    res = np.empty((len(z)), dtype=int)
    for i in range(len(z)):
        if z[i] >= 0:
            res[i] = 1
        else:
            res[i] = 0
    return res


def calculate_accuracy(Y, Y_hat):

    # res = np.zeros(1, dtype=[('TP', float), ('FP', float), ('TN', float), ('FN', float)])
    # indices = [[] for i in range(4)]
    # for i, (y, y_hat) in enumerate(zip(Y, Y_hat)):
    #     if y == y_hat and y == 1:
    #         res[0]['TP'] += 1
    #         indices[0].append(i)
    #     if y != y_hat and y == 1:
    #         res[0]['FP'] += 1
    #         indices[1].append(i)
    #     if y == y_hat and y == 0:
    #         res[0]['TN'] += 1
    #         indices[2].append(i)
    #     if y != y_hat and y == 0:
    #         res[0]['FN'] += 1
    #         indices[3].append(i)

    # raw_data = np.loadtxt(training_data_file, delimiter=',', skiprows=1, usecols=use_colons, dtype=float)
    # mats = [raw_data[index, :] for index in indices]
    # sums = [np.sum(mat[:, 6:], axis=0) for mat in mats]
    # lens = [len(mat) for mat in mats]

    # for i in range(len(sums[0])):
    #     print(i+6, end=' ')
    #     for l, s in zip(lens, sums):
    #         print(s[i]/l, end=' ')
    #     print()

    # return res

    return np.sum([1 for l, y in zip(Y, Y_hat) if l == y])/len(Y)


def get_test_data():
    data = np.loadtxt(test_data_file, delimiter=',', skiprows=1, usecols=use_colons, dtype=float)
    # mu, sd = get_mu_and_sd(training_data)
    # data = normalize_depend_on_training_data(raw_data, mu, sd)
    data = adjust(data)
    data = normalize(data)
    return np.insert(data, 0, np.ones((len(data))), axis=1)  # insert bias


def get_test_targets():
    return get_training_targets()


def save_predictions(predictions):
    with open(res_file, "w") as f:
        title = "id" + "," + "label" + '\n'
        f.write(title)
        for i, l in enumerate(predictions):
            row = str(i+1) + "," + str(l) + '\n'
            f.write(row)


def set_use_colons(sel_thre):
    X = pd.read_csv(training_data_file, header=0, index_col=False)
    Y = np.ravel(pd.read_csv(training_targets_file, header=None, index_col=False, names=['target']))

    rlr = RandomizedLasso(normalize=True, selection_threshold=sel_thre)
    rlr = rlr.fit(X, Y)

    selected = rlr.get_support()

    global use_colons
    use_colons = [i for i, s in zip(list(range(len(X.columns))), selected) if s == True]

# start_time = time.clock()

# np.seterr(all='raise')

# seeds = []

# for i in range(300):

# training_data = get_training_data()
# training_targets = get_training_targets()

# seed = np.random.randint(0, high=4294967295, dtype=np.int64)
# seed = 2728749324
# X_train, X_test, y_train, y_test = train_test_split(training_data, training_targets, test_size=0.33, random_state=seed)

# X_train = X_test = get_training_data()
# y_train = y_test = get_training_targets()

# w = train(training_data, training_targets)

w = np.loadtxt("model.txt")

# s = Seed()
# s.value = seed
# s.train_err = get_training_accuracy(w, X_train, y_train)
# s.test_err = get_training_accuracy(w, X_test, y_test)
# seeds.append(s)

# print(seed, get_training_accuracy(w, X_train, y_train), get_training_accuracy(w, X_test, y_test))

predictions = predict(get_test_data(), w)
save_predictions(predictions)

# def print_errs(errs):
#     for err in errs:
#         print(str(err.value) + " " + str(err.train_err) + " " + str(err.test_err))


# print_errs(sorted(seeds, key = lambda seed: seed.test_err, reverse=True))

# for i in range(10):
#     sel_thre = 0.7 + i * 0.05
#     print(sel_thre)
#     set_use_colons(sel_thre)
#     print(use_colons)

#     training_data = get_training_data()
    
#     # for j in range(10):     
#     w = train(training_data, training_targets)
#     print_training_accuracy(w, training_data, training_targets)

# predictions = predict(get_test_data(), w)
# save_predictions(predictions)

# predictions = predict(get_test_data(), w)
# print(predictions.shape)
# print(predictions[:100])
# accuracy = calculate_accuracy(predictions, get_test_targets())
# print(data[0])
# print(data.shape)

# print(time.clock() - start_time)
