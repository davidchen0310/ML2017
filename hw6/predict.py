import pandas as pd
from keras.models import load_model
import numpy as np
import pickle
import argparse
import os
from keras.layers import Input, Embedding, Reshape, Concatenate, Dropout, Dense, Dot, Add
from keras.models import Model
from keras.layers.advanced_activations import PReLU


# Define constants
USERS_EMBEDDING_FILE = "users_embedding_ids.pkl"
MOVIES_EMBEDDING_FILE = "movies_embedding_ids.pkl"
STD_FILE = "ratings_std.pkl"
MEAN_FILE = "ratings_mean.pkl"
LATENT_DIMENSIONS = 120
dropout_rate = 0.2

# Construct command line argument parser
parser = argparse.ArgumentParser()

parser.add_argument('--base-folder', type=str, default='.',
                    help='Base folder for data')

parser.add_argument('--res-file', type=str, default='res.csv',
                    help='Path to resulting prediction file')

parser.add_argument('--model-weights-path', type=str, default='model.h5',
                    help='Path to model weights')

parser.add_argument('--use-DNN', action='store_true',
                    help='Set this flag to use DNN model, otherwise use MF model')

parser.add_argument('--normalize', action='store_true',
                    help='Set this flag when normalization was performed before training')


# Get command line argument values
args = parser.parse_args()

test_data_file = os.path.join(args.base_folder, "test.csv")
res_file = args.res_file
model_weights_path = args.model_weights_path
use_DNN = args.use_DNN
use_normalize = args.normalize


def get_embedding_ids(df, col_name, embedding_dict_path):
	ids = df[col_name].values
	
	with open(embedding_dict_path, 'rb') as f:
		embedding_dict = pickle.load(f)

	embedding_ids = np.empty(ids.shape)
	for i, _id in enumerate(ids):
		embedding_ids[i] = embedding_dict[_id]

	print("get embedding", col_name)
	return embedding_ids, len(embedding_dict)


def normalize(df, col_name):

	with open(MEAN_FILE, 'wb') as f:
		pickle.dump(df[col_name].mean(), f)
	with open(STD_FILE, 'wb') as f:
		pickle.dump(df[col_name].std(), f)
	
	df[col_name] = (df[col_name] - df[col_name].mean()) / df[col_name].std()
	return df


def get_MF_model(num_total_users, num_total_movies):
  
  	# Input layers
	users_input = Input(shape=(1,))
	movies_input = Input(shape=(1,))

	# Dot operation
	users_embedding = Embedding(num_total_users, LATENT_DIMENSIONS, input_length=1)(users_input)
	users_embedding = Reshape((LATENT_DIMENSIONS,))(users_embedding)
	
	movies_embedding = Embedding(num_total_movies, LATENT_DIMENSIONS, input_length=1)(movies_input)
	movies_embedding = Reshape((LATENT_DIMENSIONS,))(movies_embedding)
	
	dot = Dot(1)([users_embedding, movies_embedding])

	# Handle bias
	user_bias = Embedding(num_total_users, 1, input_length=1)(users_input)
	user_bias = Reshape((1,))(user_bias)

	movie_bias = Embedding(num_total_movies, 1, input_length=1)(movies_input)
	movie_bias = Reshape((1,))(movie_bias)

	# Add dot and biases
	_sum = Add()([dot, user_bias, movie_bias])

	model = Model([users_input, movies_input], _sum)
	
	model.compile(loss='mse', optimizer='adamax')
	
	model.summary()

	return model


def get_DNN_model(num_total_users, num_total_movies):
  
  	# Input layers
	users_input = Input(shape=(1,))
	movies_input = Input(shape=(1,))

	# Ebedding operation
	users_embedding = Embedding(num_total_users, LATENT_DIMENSIONS, input_length=1)(users_input)
	users_embedding = Reshape((LATENT_DIMENSIONS,))(users_embedding)
	
	movies_embedding = Embedding(num_total_movies, LATENT_DIMENSIONS, input_length=1)(movies_input)
	movies_embedding = Reshape((LATENT_DIMENSIONS,))(movies_embedding)
	
	# Concatenate two lists of embedding tensors
	concatenate = Concatenate()([users_embedding, movies_embedding])
	dropout_0 = Dropout(dropout_rate)(concatenate)

	# Add a Dense layer
	dense_1 = Dense(LATENT_DIMENSIONS)(dropout_0)
	activation_1 = PReLU()(dense_1)
	dropout_1 = Dropout(dropout_rate)(activation_1)

	# Output layer
	out = Dense(1)(dropout_1)

	model = Model([users_input, movies_input], out)
	
	model.compile(loss="mse", optimizer='adam')
	
	model.summary()
	
	return model


def predict(model, users, movies):
	
	predictions = np.empty(len(users))
	for i, (u, m) in enumerate(zip(users, movies)):
		predictions[i] = model.predict([np.array([u]), np.array([m])])[0][0]

	if use_normalize == True:
		
		with open(MEAN_FILE, 'rb') as f:
			mean = pickle.load(f)
		with open(STD_FILE, 'rb') as f:
			std = pickle.load(f)

		predictions = predictions * std + mean

	predictions = np.clip(predictions, 1, 5)

	return predictions


def save_predictions(predictions):
	df = pd.DataFrame()
	df['TestDataID'] = range(1, len(predictions) + 1)
	df['Rating'] = predictions
	df.to_csv(res_file, index=False)


if __name__ == "__main__":
	
	df = pd.read_csv(test_data_file, sep=',')
	embedding_user_ids, num_total_users = get_embedding_ids(df, "UserID", USERS_EMBEDDING_FILE)
	embedding_movie_ids, num_total_movies = get_embedding_ids(df, "MovieID", MOVIES_EMBEDDING_FILE)
	
	if use_DNN == True:
		model = get_DNN_model(num_total_users, num_total_movies)
	else:
		model = get_MF_model(num_total_users, num_total_movies)

	model.load_weights(model_weights_path)
	
	predictions = predict(model, embedding_user_ids, embedding_movie_ids)

	save_predictions(predictions)

