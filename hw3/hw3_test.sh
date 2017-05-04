cat 65534* > 65534.h5
cat model_tf* > model_tf.h5
CUDA_VISIBLE_DEVICES=0 python3 predict.py $1 $2
