import numpy as np
import macro
import time


start_time = time.clock()

N = 10000  # how many data elements in a dataset
num_datasets_per_file = 500


def elu(arr):
    return np.where(arr > 0, arr, np.exp(arr) - 1)


def make_layer(in_size, out_size):
    w = np.random.normal(scale=0.5, size=(in_size, out_size))
    b = np.random.normal(scale=0.5, size=out_size)
    return (w, b)


def forward(inpd, layers):
    out = inpd
    for layer in layers:
        w, b = layer
        out = elu(out @ w + b)

    return out


def gen_data(dim, layer_dims, N):
    layers = []
    data = np.random.normal(size=(N, dim))

    nd = dim
    for d in layer_dims:
        layers.append(make_layer(nd, d))
        nd = d

    w, b = make_layer(nd, nd)
    gen_data = forward(data, layers)
    gen_data = gen_data @ w + b
    return gen_data


if __name__ == '__main__':
    
    for file_index in range(macro.num_dataset_files):
        
        datasets = np.empty((num_datasets_per_file, N, 100))
        labels = np.empty((num_datasets_per_file, 1))
        
        for i in range(num_datasets_per_file):
            
            # if we want to generate data with intrinsic dimension of 10
            dim = np.random.randint(1, 61)

            # the hidden dimension is randomly chosen from [60, 79] uniformly
            layer_dims = [np.random.randint(60, 80), 100]
            data = gen_data(dim, layer_dims, N)

            # (data, dim) is a (question, answer) pair
            datasets[i] = data
            labels[i] = dim

        file_name = macro.base_dataset_file_name + "_" + str(file_index) + ".npz"
        np.savez(file_name, datasets=datasets, labels=labels)

print(time.clock() - start_time)