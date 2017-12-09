import os
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def main():
	train_d, train_l, eval_d, eval_l = split_data(0.8)
	
	np.save("./train_d", train_d)
	np.save("./train_l", train_l)
	np.save("./eval_d", eval_d)
	np.save("./eval_l", eval_l)
	
	#train_d = np.asarray(np.load("./train_d.npy"), dtype=np.float32)
	#train_l = np.asarray(np.load("./train_l.npy"), dtype=np.int32)
	#eval_d = np.asarray(np.load("./eval_d.npy"), dtype=np.float32)
	#eval_l = np.asarray(np.load("./eval_l.npy"), dtype=np.int32)
	
	with tf.device('/device:GPU:0'):
		yale_classifier = tf.estimator.Estimator(
		model_fn=create_cnn, model_dir="./yale_conv_model")

		tensors_to_log = {"probabilities": "softmax_tensor"}
		logging_hook = tf.train.LoggingTensorHook(
		tensors=tensors_to_log, every_n_iter=100)

		train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": train_d},
		y=train_l,
		batch_size=100,
		num_epochs=None,
		shuffle=True)
	
		
		yale_classifier.train(
			input_fn=train_input_fn,
			steps=2000,
			hooks=[logging_hook])
		
		
		eval_input_fn = tf.estimator.inputs.numpy_input_fn(
			x={"x": eval_d},
			y=eval_l,
			num_epochs=1,
			shuffle=False)
	        eval_results = yale_classifier.evaluate(input_fn=eval_input_fn)
	

def create_cnn(features, labels, mode):
	#Images are 168*168 pixels with one color channel
	print(features["x"])
	input_layer = tf.reshape(features["x"], [-1, 168, 168, 1])
	
	#Input [batch_size, 168, 168, 1]
	#Output [batch_size, 168, 168, 350]
	conv1 = tf.layers.conv2d(
			inputs=input_layer,
			filters= 350,
			kernel_size=[7,7],
			padding="same",
			activation=tf.nn.relu)
	print(conv1)
	#Input [batch_size, 168, 168, 350]
	#Output [batch_size, 42, 42, 350]
	pool1 = tf.layers.max_pooling2d(
			inputs=conv1, 
			pool_size=[4,4], 
			strides=4)
	print(pool1)
	#Input [batch_size, 42, 42, 350]
	#Output [batch_size, 42, 42, 350]
	norm1 = tf.layers.batch_normalization(
			inputs=pool1)
	print(norm1)
	#Input [batch_size, 42, 42, 350]
	#Output [batch_size, 42, 42, 60]
	conv2 = tf.layers.conv2d(
			inputs=norm1,
			filters=60,
			kernel_size=[5,5],
			padding="same",
			activation=tf.nn.relu)
	print(conv2)
	#Input [batch_size, 42, 42, 60]
	#Output [batch_size, 42, 42, 60]
	norm2 = tf.layers.batch_normalization(
			inputs=conv2)
	print(norm2)
	#Input [batch_size, 42, 42, 60]
	#Output [batch_size, 14, 14, 196]
	pool2 = tf.layers.max_pooling2d(
			inputs=norm2,
			pool_size=[3,3],
			strides=3)
	print(pool2)
	pool2_flat = tf.reshape(pool2, [-1,14*14*60])
	print(pool2_flat)
	dense = tf.layers.dense(inputs=pool2_flat, units=14000, activation=tf.nn.relu)
	print(dense)
	dropout = tf.layers.dropout(
			inputs=dense,
			rate=0.4,
			training=mode == tf.estimator.ModeKeys.TRAIN)

	logits = tf.layers.dense(inputs=dropout, units=38)
	print(dropout)
	print(logits)
	predictions = {
		"classes": tf.argmax(input=logits, axis=1),
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	} 
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	onehot_labels = tf.one_hot(indices=labels, depth=38)
	print(labels)
	print(onehot_labels)
	loss = tf.losses.softmax_cross_entropy(
		onehot_labels=onehot_labels, logits=logits)

	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	eval_metric_ops = {
		"accuracy": tf.metrics.accuracy(
			labels=labels, predictions=predictions["classes"])
		}
	return tf.estimator.EstimatorSpec(
		mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def split_data(percent_train):
	indir = os.path.dirname(os.path.realpath(__file__))
	data_names = np.array([])
	i = 0
	for root, idrs, filenames in os.walk(indir+"/ImageData"):
		for f in filenames:
			data_names = np.append(data_names, f)

	data_perm = np.random.permutation(data_names.shape[0])
	
	training = np.array(data_perm[:int(len(data_names)*percent_train)])
	evaluation = np.array(data_perm[int(len(data_names)*percent_train):])
	
	tra_label = np.array([int(data_names[x][6:8]) for x in training])
	eva_label = np.array([int(data_names[x][6:8]) for x in evaluation])
	
	tra_data = np.array([collect_image_bytes(x, filenames) for x in training])
	eva_data = np.array([collect_image_bytes(x, filenames) for x in evaluation])

	return [tra_data, tra_label, eva_data, eva_label]


def collect_image_bytes(index, filenames):
	indir = os.path.dirname(os.path.realpath(__file__))
	myFile = open(indir+"/ImageData/"+filenames[index], 'r+b')

	image_data = myFile.read()[15:]
	final = np.array([])

	even = 0
	i = 0
	for byte in image_data:
		i += 1
		final = np.append(final, byte)
	
	return final

if __name__ == "__main__":
	tf.app.run()
