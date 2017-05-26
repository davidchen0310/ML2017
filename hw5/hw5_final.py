import sys
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import numpy as np
from keras import backend as K

test_data_path = sys.argv[1]
res_path = sys.argv[2]
X_tokenizer_path = "X_tokenizer.pkl"
Y_tokenizer_path = "Y_tokenizer.pkl"
model_path = "best.hdf5"
# model_weights_path = "model_weights.hdf5"
# best_thresholds_path = "best_thresholds.npy"
# aux_path = "aux.pkl"


def f1_score(y_true,y_pred):
    thresh = 0.25
    # thresh = np.load("thresh.npy")
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)
    
    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))


def get_test_data():
	
	X_test_raw = []
	with open(test_data_path, 'r', encoding = "utf-8") as f:
		for row in f:
			splits = row.split(',',1)
			X_test_raw.append(splits[1])  

	with open(X_tokenizer_path, 'rb') as f:
		X_tokenizer = pickle.load(f)
	
	X_test_sequences = X_tokenizer.texts_to_sequences(X_test_raw[1:])
	X_test = pad_sequences(X_test_sequences)

	return X_test


def get_model():
	
	# with open(model_path, 'r') as json_file:
	# 	model_json = json_file.read()
	# 	model = model_from_json(model_json)

	# model.load_weights(model_weights_path)

	return load_model(model_path, custom_objects={"f1_score": f1_score})


def transform(raw_predictions):

	with open(Y_tokenizer_path, 'rb') as f:
		Y_tokenizer = pickle.load(f)
		# class_dict = {v: k for k, v in Y_tokenizer.word_index.items()}

	tag_list = 38 * [None]
	for tag, index in Y_tokenizer.word_index.items():
		tag_list[index-1] = tag
		# print(tag, " ", tag_list[index-1])
	tag_list = [tag.upper() for tag in tag_list]

	# # with open(best_thresholds_path, 'rb') as f:
	# best_thresholds = np.load(best_thresholds_path)

	res = []
	for i, row in enumerate(raw_predictions):
		temp = []
		for j, class_prob in enumerate(row):
			if class_prob >= 0.25:
				temp.append(tag_list[j])
		if not temp:
			distance_to_threshold = 0.25 - row
			temp.append(tag_list[np.argmin(distance_to_threshold)])
		
		res.append(temp)

	# X_test_raw = []
	# with open(test_data_path, 'r', encoding = "utf-8") as f:
	# 	for row in f:
	# 		splits = row.split(',',1)
	# 		X_test_raw.append(splits[1])  

	# X_test_raw = X_test_raw[1:]

	# with open(aux_path, 'rb') as f:
	# 	classes_dict = pickle.load(f)

	# predictions = []
	# for article in X_test_raw:
	# 	tags = classes_dict.get(article)
	# 	predictions.append(tags)

	return res


def save_predictions(predictions):
	
	with open(res_path, 'w') as f:
		
		f.write("\"id\",\"tags\"\n")
		
		for i, prediction in enumerate(predictions):
			
			f.write("\"" + str(i) + "\",\"")
			
			for j, p in enumerate(prediction):
				if j != 0:
					f.write(" ")
				f.write(str(p))	
			
			f.write("\"\n")


features = get_test_data()
model = get_model()
raw_predictions = model.predict_proba(features)
predictions = transform(raw_predictions)
save_predictions(predictions)
