import numpy as np 
import tensorflow as tf
from PIL import Image
import sys
import os
import skimage.io
import time

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, create_pairwise_gaussian, softmax_to_unary

import vgg

import dataset

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

input_fn_images = dataset.input_fn_images
input_fn_np = dataset.input_fn_np
slim = tf.contrib.slim

num_classes = 0 # 21 for pascal
mask_out_class_label = 0 # 255 for pascal
resized_height = 0 # 512 usually
resized_width = 0 # 512 usually
ckpt_path = '' # path for saving checkpoint
vgg16_ckpt_path = '' # checkpoint file of vgg16
dataset.resized_height = 0 # same as resized_height
dataset.resized_width = 0 # same as resized_height

#mask out class 255, and flat.
def valid(logits, annotation):
	#logits [batch, H, w, num_classes]
	#annotation [batch, H, w]
	#valid_logits [num_valid_eintries, num_classes]
	#valid_labels [num_valid_eintries]
    
	# mask_out_class_label = list(class_labels)[-1] #255
	# mask_out_class_label=255
	
	#[batch,H,W]
	valid_labels_mask = tf.not_equal(annotation, mask_out_class_label)
	
	#doc: If both x and y are None, then this operation returns the coordinates of true elements of condition. 
	#[num_valid_eintries, ranks_of_annotation]. Ranks_of_annotation is usually 3. Elements in axis 1 are coordinates.
	valid_labels_indices = tf.where(valid_labels_mask) 
	valid_indices = tf.to_int32(valid_labels_indices)
	
	#[num_valid_eintries]
	valid_labels = tf.gather_nd(params=annotation, indices=valid_indices)
	#[num_valid_eintries, num_classes]
	valid_logits = tf.gather_nd(params=logits, indices=valid_indices)

	return valid_logits, valid_labels
	
# get bilinear weights	
def get_deconv_filter(f_shape):
	width = f_shape[0]
	height = f_shape[1]
	f = ceil(width/2.0)
	c = (2 * f - 1 - f % 2) / (2.0 * f)
	bilinear = np.zeros([f_shape[0], f_shape[1]])
	for x in range(width):
		for y in range(height):
			value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
			bilinear[x, y] = value
	weights = np.zeros(f_shape)
	for i in range(f_shape[2]):
		weights[:, :, i, i] = bilinear

	init = tf.constant_initializer(value=weights,
								   dtype=tf.float32)
	var = tf.get_variable(name="convt_weights", initializer=init,
						  shape=weights.shape)
	return var

# FCN network according to paper Fully Convolutional Networks for Semantic Segmentation
# train both fcn and vgg
# result logits is upsampled fused layers of pool3, pool4 and pool5	
def fcn_paper(inputs_32s, inputs_16s, inputs_8s, img_height, img_width, is_training = True):
	#inputs: [batch,h,w,channels]. And can be (1, any_size, any_size, channels).
	#logits: [batch,h,w,classes]
	#upsampled_logits: [batch,H,W,classes]
	#annotation: [batch,H,W]
	#loss: [batch,H,W]
	#((11, 15, 512), (23, 31, 512), (46, 62, 256), (375, 500), b'2007_000645', '2007_000645')
		
	with tf.variable_scope('fcn/logits/32s') as scope:
		weights = tf.get_variable('weights', shape = [1, 1, 4096, num_classes], dtype = tf.float32, initializer=tf.glorot_normal_initializer() )
		biases = tf.get_variable('biases', shape=[num_classes], dtype=tf.float32, initializer=tf.constant_initializer(0.0) )
		conv = tf.nn.conv2d(inputs_32s, weights, strides=[1,1,1,1], padding='SAME')
		logits_32s = tf.nn.bias_add(conv, biases)	
		
		convt_weights = get_deconv_filter([4,4,num_classes, num_classes])
		inputs_16s_shape = tf.shape(inputs_16s)
		logits_32s_upsampled = tf.nn.conv2d_transpose(
			value=logits_32s,
			filter=convt_weights,
			output_shape=[inputs_16s_shape[0], inputs_16s_shape[1], inputs_16s_shape[2], num_classes],
			strides=[1, 2, 2, 1],
			padding='SAME',
			data_format='NHWC',
			name=None
		)
		
	with tf.variable_scope('fcn/logits/16s') as scope:
		weights = tf.get_variable('weights', shape = [1, 1, 512, num_classes], dtype = tf.float32, initializer=tf.glorot_normal_initializer() )
		biases = tf.get_variable('biases', shape=[num_classes], dtype=tf.float32, initializer=tf.constant_initializer(0.0) )
		conv = tf.nn.conv2d(inputs_16s, weights, strides=[1,1,1,1], padding='SAME')
		logits_16s = tf.nn.bias_add(conv, biases)	
		
		fused_logits_16s = logits_16s + logits_32s_upsampled
		
		convt_weights = get_deconv_filter([4,4,num_classes, num_classes])
		inputs_8s_shape = tf.shape(inputs_8s)
		logits_16s_upsampled = tf.nn.conv2d_transpose(
			value=fused_logits_16s,
			filter=convt_weights,
			output_shape=[inputs_8s_shape[0], inputs_8s_shape[1], inputs_8s_shape[2], num_classes],
			strides=[1, 2, 2, 1],
			padding='SAME',
			data_format='NHWC',
			name=None
		)
		
	with tf.variable_scope('fcn/logits/8s') as scope:
		weights = tf.get_variable('weights', shape = [1, 1, 256, num_classes], dtype = tf.float32, initializer=tf.glorot_normal_initializer() )
		biases = tf.get_variable('biases', shape=[num_classes], dtype=tf.float32, initializer=tf.constant_initializer(0.0) )
		conv = tf.nn.conv2d(inputs_8s, weights, strides=[1,1,1,1], padding='SAME')
		logits_8s = tf.nn.bias_add(conv, biases)	
		
		fused_logits_8s = logits_8s + logits_16s_upsampled
	
		convt_weights = get_deconv_filter([16,16,num_classes, num_classes])
		logits_8s_upsampled = tf.nn.conv2d_transpose(
			value=fused_logits_8s,
			filter=convt_weights,
			output_shape=[inputs_8s_shape[0], img_height, img_width, num_classes],
			strides=[1, 8, 8, 1],
			padding='SAME',
			data_format='NHWC',
			name=None
		)
	
	return logits_8s_upsampled
	
# FCN network wont't train vgg
# result logits is upsampled fused layers of pool3, pool4 and pool5	
def fcn_light(inputs_32s, inputs_16s, inputs_8s, img_height, img_width, is_training = True):
	#inputs: [batch,h,w,channels]. And can be (1, any_size, any_size, channels).
	#logits: [batch,h,w,classes]
	#upsampled_logits: [batch,H,W,classes]
	#annotation: [batch,H,W]
	#loss: [batch,H,W]
	#((11, 15, 512), (23, 31, 512), (46, 62, 256), (375, 500), b'2007_000645', '2007_000645')
	
	with tf.variable_scope('fcn/32s/fc6') as scope:
		weights = tf.get_variable('weights', shape = [7, 7, 512, 1024], dtype = tf.float32, initializer=tf.glorot_normal_initializer() )
		biases = tf.get_variable('biases', shape=[1024], dtype=tf.float32, initializer=tf.constant_initializer(0.0) )
		conv = tf.nn.conv2d(inputs_32s, weights, strides=[1,1,1,1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)	
		conv1 = tf.nn.relu(pre_activation)
		conv1 = tf.layers.dropout(inputs=conv1, rate=0.3, training=is_training)
		
	with tf.variable_scope('fcn/32s/fc8') as scope:
		weights = tf.get_variable('weights', shape = [1, 1, 1024, num_classes], dtype = tf.float32, initializer=tf.glorot_normal_initializer() )
		biases = tf.get_variable('biases', shape=[num_classes], dtype=tf.float32, initializer=tf.constant_initializer(0.0) )
		conv = tf.nn.conv2d(conv1, weights, strides=[1,1,1,1], padding='SAME')
		logits_32s = tf.nn.bias_add(conv, biases)	
		
	with tf.variable_scope('fcn/32s/convt') as scope:	
		convt_weights = get_deconv_filter([4,4,num_classes, num_classes])
		inputs_16s_shape = tf.shape(inputs_16s)
		logits_32s_upsampled = tf.nn.conv2d_transpose(
			value=logits_32s,
			filter=convt_weights,
			output_shape=[inputs_16s_shape[0], inputs_16s_shape[1], inputs_16s_shape[2], num_classes],
			strides=[1, 2, 2, 1],
			padding='SAME',
			data_format='NHWC',
			name=None
		)
	
	with tf.variable_scope('fcn/16s/fc6') as scope:
		weights = tf.get_variable('weights', shape = [7, 7, 512, 1024], dtype = tf.float32, initializer=tf.glorot_normal_initializer() )
		biases = tf.get_variable('biases', shape=[1024], dtype=tf.float32, initializer=tf.constant_initializer(0.0) )
		conv = tf.nn.conv2d(inputs_16s, weights, strides=[1,1,1,1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)	
		conv1 = tf.nn.relu(pre_activation)
		conv1 = tf.layers.dropout(inputs=conv1, rate=0.3, training=is_training)
		
	with tf.variable_scope('fcn/16s/fc8') as scope:
		weights = tf.get_variable('weights', shape = [1, 1, 1024, num_classes], dtype = tf.float32, initializer=tf.glorot_normal_initializer() )
		biases = tf.get_variable('biases', shape=[num_classes], dtype=tf.float32, initializer=tf.constant_initializer(0.0) )
		conv = tf.nn.conv2d(conv1, weights, strides=[1,1,1,1], padding='SAME')
		logits_16s = tf.nn.bias_add(conv, biases)	
		
		fused_logits_16s = logits_16s + logits_32s_upsampled
	
	with tf.variable_scope('fcn/16s/convt') as scope:	
		convt_weights = get_deconv_filter([4,4,num_classes, num_classes])
		inputs_8s_shape = tf.shape(inputs_8s)
		logits_16s_upsampled = tf.nn.conv2d_transpose(
			value=fused_logits_16s,
			filter=convt_weights,
			output_shape=[inputs_8s_shape[0], inputs_8s_shape[1], inputs_8s_shape[2], num_classes],
			strides=[1, 2, 2, 1],
			padding='SAME',
			data_format='NHWC',
			name=None
		)

	with tf.variable_scope('fcn/8s/fc6') as scope:
		weights = tf.get_variable('weights', shape = [7, 7, 256, 512], dtype = tf.float32, initializer=tf.glorot_normal_initializer() )
		biases = tf.get_variable('biases', shape=[512], dtype=tf.float32, initializer=tf.constant_initializer(0.0) )
		conv = tf.nn.conv2d(inputs_8s, weights, strides=[1,1,1,1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)	
		conv1 = tf.nn.relu(pre_activation)
		conv1 = tf.layers.dropout(inputs=conv1, rate=0.3, training=is_training)
		
	with tf.variable_scope('fcn/8s/fc8') as scope:
		weights = tf.get_variable('weights', shape = [1, 1, 512, num_classes], dtype = tf.float32, initializer=tf.glorot_normal_initializer() )
		biases = tf.get_variable('biases', shape=[num_classes], dtype=tf.float32, initializer=tf.constant_initializer(0.0) )
		conv = tf.nn.conv2d(conv1, weights, strides=[1,1,1,1], padding='SAME')
		logits_8s = tf.nn.bias_add(conv, biases)	
		
		fused_logits_8s = logits_8s + logits_16s_upsampled
	
	with tf.variable_scope('fcn/8s/convt') as scope:	
		convt_weights = get_deconv_filter([16,16,num_classes, num_classes])
		logits_8s_upsampled = tf.nn.conv2d_transpose(
			value=fused_logits_8s,
			filter=convt_weights,
			output_shape=[inputs_8s_shape[0], img_height, img_width, num_classes],
			strides=[1, 8, 8, 1],
			padding='SAME',
			data_format='NHWC',
			name=None
		)
	
	return logits_8s_upsampled

def vgg_16(inputs, no_fc=False):
	with slim.arg_scope(vgg.vgg_arg_scope()):
		net, end_points = vgg.vgg_16(inputs, None, is_training=False, spatial_squeeze=False, fc_conv_padding='SAME', no_fc=no_fc)
	
	if no_fc:
		return end_points['vgg_16/pool5'], end_points['vgg_16/pool4'], end_points['vgg_16/pool3']
	else:
		return net, end_points['vgg_16/pool4'], end_points['vgg_16/pool3']

def train(image_tfrecords_file, epochs, batch_size, retrain=False, restore_step=None, use_fcn_paper=False):
	#inputs: [batch,h,w,channels]. And can be (1, any_size, any_size, channels).
	#annotations [batch, H, W]
	#logits: [batch,h,w,classes]
	#upsampled_logits: [batch,H,W,classes]
	#valid_logits [num_valid_eintries, num_classes]
	#valid_labels [num_valid_eintries]

	global_step = tf.get_variable('global_step', [], dtype = tf.int32, initializer=tf.constant_initializer(0), trainable=False)

	images, annotations, _ = input_fn_images(image_tfrecords_file, epochs, batch_size)
	
	if use_fcn_paper:
		inputs_32s, inputs_16s, inputs_8s = vgg_16(images, no_fc=False)
	else:
		inputs_32s, inputs_16s, inputs_8s = vgg_16(images, no_fc=True)
	
	if use_fcn_paper:
		upsampled_logits = fcn_paper(inputs_32s, inputs_16s, inputs_8s, tf.shape(annotations)[1], tf.shape(annotations)[2])
	else:
		upsampled_logits = fcn_light(inputs_32s, inputs_16s, inputs_8s, tf.shape(annotations)[1], tf.shape(annotations)[2])
	
	valid_logits, valid_labels = valid(upsampled_logits, annotations)
	
	top_k = tf.nn.in_top_k(valid_logits, valid_labels, 1)
	corrects = tf.cast(top_k, tf.float16)
	accuracy = tf.reduce_mean(corrects, name='accuracy')

	loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=valid_logits, labels=valid_labels)
	loss_mean = tf.reduce_mean(loss)
	loss_sum = tf.reduce_sum(loss)
	
	if use_fcn_paper:
		trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
	else:
		trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fcn')
	
	with tf.variable_scope("opt_vars"): #会根据var_list建立一些variables
		# train_step = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(loss=loss_mean)
		train_step = tf.train.MomentumOptimizer(0.00001, 0.9).minimize(
			loss=loss_mean, global_step=global_step, var_list=trainable_variables)
	
	# opt_vars won't be in TRAINABLE_VARIABLES, but vgg will.
	
	saver = tf.train.Saver()
	
	sess=tf.Session()

	if retrain:
		restorer = tf.train.Saver()
		
		ckpt_path_local_var = ckpt_path
		if restore_step != None:
			ckpt_path_local_var = ckpt_path_local_var+'-'+str(restore_step)
		
		restorer.restore(sess, ckpt_path_local_var)
	else:
		sess.run(tf.local_variables_initializer())
		sess.run(tf.global_variables_initializer())
		
		variables_to_restore = slim.get_variables_to_restore(exclude=['global_step', 'opt_vars', 'fcn'])
		restorer = tf.train.Saver(variables_to_restore)
		restorer.restore(sess, vgg16_ckpt_path)
	
	steps=0
	while True:
		try:
			loss_mean_v, loss_sum_v, _, global_step_v, accuracy_v = sess.run([
				loss_mean, loss_sum, train_step, global_step, accuracy
			])

		except tf.errors.OutOfRangeError:
			break
			
		if steps%20 == 0:
			cur_time = time.strftime('%H:%M:%S', time.localtime())
			print('{} steps: {}, loss_mean: {}, loss_sum: {}, global_step: {}, accuracy: {}'.format(
				cur_time, 
				steps, 
				round(float(loss_mean_v),4), 
				round(float(loss_sum_v),4), 
				global_step_v, 
				round(float(accuracy_v), 4)))

		if steps%1000 == 0 and steps!=0:
			save_path = saver.save(sess, ckpt_path)
			print('ckpt saved in ' + save_path)
			
		steps+=1
			
	save_path = saver.save(sess, ckpt_path)
	print('finally ckpt saved in ' + save_path)
	print('completed. steps: {}, loss_mean: {}, loss_sum: {}, global_step: {}, accuracy: {}'.format(
		steps, 
		round(float(loss_mean_v),4), 
		round(float(loss_sum_v),4), 
		global_step_v, 
		round(float(accuracy_v), 4)))

def eval(image_tfrecords_file, batch_size=1, restore_step=None, use_fcn_paper=False):

	images, annotations, _ = input_fn_images(image_tfrecords_file, epochs=None, batch_size=batch_size, is_training=False)
	
	if use_fcn_paper:
		inputs_32s, inputs_16s, inputs_8s = vgg_16(images, no_fc=False)
	else:
		inputs_32s, inputs_16s, inputs_8s = vgg_16(images, no_fc=True)
	
	if use_fcn_paper:
		upsampled_logits = fcn_paper(inputs_32s, inputs_16s, inputs_8s, tf.shape(annotations)[1], tf.shape(annotations)[2], False)
	else:
		upsampled_logits = fcn_light(inputs_32s, inputs_16s, inputs_8s, tf.shape(annotations)[1], tf.shape(annotations)[2], False)
	
	
	valid_logits, valid_labels = valid(upsampled_logits, annotations)
	
	top_k = tf.nn.in_top_k(valid_logits, valid_labels, 1)
	corrects = tf.cast(top_k, tf.float16)
	accuracy = tf.reduce_mean(corrects, name='accuracy')
	
	pred = tf.argmax(input=valid_logits, axis=1) #[num_valid_eintries]
	mean_iou, iou_update_op = tf.metrics.mean_iou(labels=valid_labels, predictions=pred, num_classes=num_classes, weights=None)
	mean_acc, acc_update_op = tf.metrics.accuracy(labels=valid_labels, predictions=pred)
	
	# variables_to_restore = slim.get_variables_to_restore(include=['vgg_16/fc6', 'vgg_16/fc7'])
	restorer = tf.train.Saver()
	
	sess=tf.Session()

	sess.run(tf.local_variables_initializer())
	sess.run(tf.global_variables_initializer())
	
	ckpt_path_local_var = ckpt_path
	if restore_step != None:
		ckpt_path_local_var = ckpt_path_local_var+'-'+str(restore_step)
		
	restorer.restore(sess, ckpt_path_local_var)
	
	all_acc=[]
	steps=0
	# for i in range(3):
	while True:
		try:
			acc_v, _, __ = sess.run([accuracy, iou_update_op, acc_update_op])
			all_acc.append(acc_v)
			
		except tf.errors.OutOfRangeError:
			break
			
		if steps%20 == 0:
			cur_time = time.strftime('%H:%M:%S', time.localtime())
			print('{} eval steps: {}'.format(cur_time, steps))
		
		steps+=1
	
	mean_iou_v, mean_acc_v =  sess.run([mean_iou, mean_acc]) #不会调动一个batch
	mean_acc_basic = np.mean(all_acc)

	print('Eval done. mean_iou: {}, mean_acc: {}, mean_acc_np: {}'.format(mean_iou_v,mean_acc_v,mean_acc_basic))
	return mean_iou_v, mean_acc_v

# predict eval tfrecords	
def infer(image_tfrecords_file, images_dir, start=0, end=1, restore_step=None, use_fcn_paper=False):
	images, annotations, name = input_fn_images(image_tfrecords_file, epochs=None, batch_size=1, is_training=False)
	
	annotations = tf.cast(annotations, tf.uint8)
	
	if use_fcn_paper:
		inputs_32s, inputs_16s, inputs_8s = vgg_16(images, no_fc=False)
	else:
		inputs_32s, inputs_16s, inputs_8s = vgg_16(images, no_fc=True)
	
	if use_fcn_paper:
		upsampled_logits = fcn_paper(inputs_32s, inputs_16s, inputs_8s, tf.shape(annotations)[1], tf.shape(annotations)[2], False)
	else:
		upsampled_logits = fcn_light(inputs_32s, inputs_16s, inputs_8s, tf.shape(annotations)[1], tf.shape(annotations)[2], False)
	
	#[batch,H,W]
	pred = tf.argmax(input=upsampled_logits, axis=3)
	
	#[batch,H,W,classes]
	probabilities = tf.nn.softmax(upsampled_logits)
	
	restorer = tf.train.Saver()
	
	sess=tf.Session()
	
	ckpt_path_local_var = ckpt_path
	if restore_step != None:
		ckpt_path_local_var = ckpt_path_local_var+'-'+str(restore_step)
		
	restorer.restore(sess, ckpt_path_local_var)
	
	for i in range(end):
		pred_v, names_v, probabilities_v, annotations_v = sess.run([
			pred, 
			name[0],
			probabilities, 
			annotations
		])
		
		if i not in range(start, end):
			continue
		
		image_name = str(names_v, encoding='utf8')
		annotation = annotations_v.squeeze()
		print('image name: {}'.format(image_name))
		image_np = np.array(Image.open(images_dir+'/'+image_name+'.jpg').resize((resized_width,resized_height)))	
		optimized_pred = crf(image_np, probabilities_v.squeeze())
		skimage.io.imshow_collection([
			image_np, 
			annotation, 
			optimized_pred, 
			pred_v.squeeze()
		])
		
		skimage.io.show()

# predict own images
def indef_np(images_list, start=0, end=1, restore_step=None, use_fcn_paper=False):
	images, images_ori = input_fn_np(images_list, True)
	
	if use_fcn_paper:
		inputs_32s, inputs_16s, inputs_8s = vgg_16(images, no_fc=False)
	else:
		inputs_32s, inputs_16s, inputs_8s = vgg_16(images, no_fc=True)
	
	if use_fcn_paper:
		upsampled_logits = fcn_paper(inputs_32s, inputs_16s, inputs_8s, tf.shape(images)[1], tf.shape(images)[2], False)
	else:
		upsampled_logits = fcn_light(inputs_32s, inputs_16s, inputs_8s, tf.shape(images)[1], tf.shape(images)[2], False)
	
	
	#[batch,H,W]
	pred = tf.argmax(input=upsampled_logits, axis=3)
	
	#[batch,H,W,classes]
	probabilities = tf.nn.softmax(upsampled_logits)
	
	restorer = tf.train.Saver()
	
	sess=tf.Session()
	
	ckpt_path_local_var = ckpt_path
	if restore_step != None:
		ckpt_path_local_var = ckpt_path_local_var+'-'+str(restore_step)
		
	restorer.restore(sess, ckpt_path_local_var)
	
	pred_v, probabilities_v = sess.run([pred, probabilities])
	
	for i in range(end):
		if i not in range(start, end):
			continue
		
		skimage.io.imshow_collection([pred_v[i], images_ori[i]])
		skimage.io.show()

# fully connected CRF
# image: original resized(if needed) image np, shape(H, W, 3)
# softmax: probabilities np, shape(H, W, num_classes)
def crf(image, softmax):

	softmax = softmax.transpose((2, 0, 1))

	# The input should be the negative of the logarithm of probability values
	# Look up the definition of the softmax_to_unary for more information
	unary = softmax_to_unary(softmax)

	# The inputs should be C-continious -- we are using Cython wrapper
	unary = np.ascontiguousarray(unary)

	d = dcrf.DenseCRF(image.shape[0] * image.shape[1], num_classes)

	d.setUnaryEnergy(unary)

	# This potential penalizes small pieces of segmentation that are
	# spatially isolated -- enforces more spatially consistent segmentations
	feats = create_pairwise_gaussian(sdims=(10, 10), shape=image.shape[:2])

	d.addPairwiseEnergy(feats, compat=3,
		            kernel=dcrf.DIAG_KERNEL,
		            normalization=dcrf.NORMALIZE_SYMMETRIC)

	# This creates the color-dependent features --
	# because the segmentation that we get from CNN are too coarse
	# and we can use local color features to refine them
	feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20),
		                           img=image, chdim=2)

	d.addPairwiseEnergy(feats, compat=10,
		             kernel=dcrf.DIAG_KERNEL,
		             normalization=dcrf.NORMALIZE_SYMMETRIC)
	Q = d.inference(5)

	optimized_pred = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))
	
	return optimized_pred

# print the information of checkpoint	
def chkpt():
	from tensorflow.python.tools import inspect_checkpoint as chkp
	chkp.print_tensors_in_checkpoint_file(ckpt_path, tensor_name='', all_tensors=False,all_tensor_names=True)
		
if __name__ == '__main__':
	num_classes = 21
	mask_out_class_label = 255
	resized_height = 384
	resized_width = 384
	ckpt_path = 'fcn8s.ckpt'
	vgg16_ckpt_path='vgg_16.ckpt'
	dataset.resized_height = 384
	dataset.resized_width = 384
	
	# chkpt()
	
		
	
	

		




