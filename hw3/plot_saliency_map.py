from keras.models import load_model, Model
import numpy as np
import pandas as pd
from vis.visualization import visualize_saliency
from vis.utils import utils
import cv2

model_file = "65534.h5"
data_file = "train.csv"
saliency_img_file = "saliency_map.png"
original_img_file = "p4_original.png"
size = 48
indices = [6, 8, 21, 38, 416, 423, 425]

data = pd.read_csv(data_file, nrows = np.max(indices) + 1)
x = np.array([r[1].split() for r in data.values], dtype=float).reshape((data.shape[0], size, size, 1))
imgs = x[indices]

model = load_model(model_file)
pred_class = model.predict_classes(imgs)

heatmaps = []

for img, pc in zip(imgs, pred_class):
	heatmap = visualize_saliency(model, len(model.layers) - 1, [pc], img)
	heatmaps.append(heatmap)

cv2.imwrite(saliency_img_file, utils.stitch_images(heatmaps, cols=7))
cv2.imwrite(original_img_file, utils.stitch_images(list(imgs), cols=7))
