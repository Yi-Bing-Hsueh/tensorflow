
import numpy as np 
import tensorflow as tf
from PIL import Image
import skimage.io

resized_height = 0 # 512 usually
resized_width = 0 # 512 usually

def _random_flip(image, annotation):
	#annotation: (H, W)

	#(H, W, 1)
	image1,image2,image3 = tf.split(image, [1,1,1], 2)
	
	#(H, W)
	image1 = tf.squeeze(image1, 2)
	image2 = tf.squeeze(image2, 2)
	image3 = tf.squeeze(image3, 2)
	
	#(H, W, 4)
	stackimg = tf.stack([image1,image2,image3,annotation], 2)
	
	stackimg = tf.image.random_flip_left_right(stackimg)
	
	stackimg = tf.image.random_flip_up_down(stackimg)
	
	#(H, W, 3), (H, W, 1)
	image, annotation = tf.split(stackimg, [3,1], 2)
	annotation = tf.squeeze(annotation, 2)
	
	return image, annotation

# image: (H,W,3)
# annotation: (H,W)
def _resize_img(image, annotation):
	annotation = tf.expand_dims(annotation, 2)

	image = tf.expand_dims(image, 0)
	annotation = tf.expand_dims(annotation, 0) #(1,H,W,1)
	
	image_dtype = image.dtype
	image = tf.image.resize_bilinear(image, (resized_height, resized_width)) #to be float32
	image = tf.cast(image, image_dtype)
	
	# image = tf.image.resize_nearest_neighbor(image, (resized_height, resized_width))
	annotation = tf.image.resize_nearest_neighbor(annotation, (resized_height, resized_width))
	
	image = tf.squeeze(image, [0])
	annotation = tf.squeeze(annotation, [0, 3])
	
	return image, annotation
		
#some code to check the tfrecords
def chk_imges_ds(images_tfrecords_file):
	def _chk_imges_ds_map(image):
		feature = {
			'image': tf.FixedLenFeature([], tf.string),
			'annotation': tf.FixedLenFeature([], tf.string),
			"height": tf.FixedLenFeature([], tf.int64),
			"width": tf.FixedLenFeature([], tf.int64),
			"name": tf.FixedLenFeature([], tf.string),
		}
		features = tf.parse_single_example(image, features=feature)
		
		image = tf.decode_raw(features['image'], tf.uint8)
		annotation = tf.decode_raw(features['annotation'], tf.uint8)
		# image = tf.cast(image, tf.float32) # comment for image viewing
		# annotation = tf.cast(annotation, tf.int32) # comment for image view
		height = tf.cast(features['height'], tf.int32)
		width = tf.cast(features['width'], tf.int32)
		
		image = tf.reshape(image, (height, width, 3))
		annotation = tf.reshape(annotation, (height, width))
		
		image, annotation = _resize_img(image, annotation)
		
		# image, annotation = _random_flip(image, annotation)
		
		# image = tf.image.random_brightness(image, 0.1)
		
		# image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
		
		# image = tf.image.adjust_hue(image, -0.1) #center is 0. max[-1, 1]. 变其反色，正往暖色方向变，负往冷色方向变。
		# image = tf.image.random_hue(image, 0.1) #0.1 or 0.15 is good
		
		# image = tf.image.adjust_saturation(image, 2) #center=1. 0.0-2.0 is good. 越大越艳，越小越黑白。可负。
		# image = tf.image.random_saturation(image, 0.5, 1.5)
		
		
		# image = image- [123.68, 116.78, 103.94] # comment for image viewing

		return image, annotation, features['name']

	ds=tf.data.TFRecordDataset(images_tfrecords_file)
	ds=ds.map(_chk_imges_ds_map)
	iterator = ds.make_one_shot_iterator()
	image, annotation, name = iterator.get_next()
	
	sess=tf.Session()

	
	#no batch
	
	step = 0
	# while True:
	for i in range(10):
		try:
			image_v, annotation_v, name_v = sess.run([image, annotation, name])

			print((image_v.dtype, annotation_v.dtype, annotation_v.shape, name_v))

			# Image.fromarray(image_v).save('chkimds3/'+str(name_v, encoding = "utf8")+'_'+str(step)+'.jpg')

			skimage.io.imshow_collection([image_v, annotation_v])
			skimage.io.show()

			step+=1

		except tf.errors.OutOfRangeError:
			break
	print('final step: {}'.format(step))
	

def _input_fn_map_train(image):
	feature = {
		'image': tf.FixedLenFeature([], tf.string),
		'annotation': tf.FixedLenFeature([], tf.string),
		"height": tf.FixedLenFeature([], tf.int64),
		"width": tf.FixedLenFeature([], tf.int64),
		"name": tf.FixedLenFeature([], tf.string),
	}
	features = tf.parse_single_example(image, features=feature)
	
	image = tf.decode_raw(features['image'], tf.uint8)
	annotation = tf.decode_raw(features['annotation'], tf.uint8)
	
	height = tf.cast(features['height'], tf.int32)
	width = tf.cast(features['width'], tf.int32)
	
	image = tf.reshape(image, (height, width, 3))
	annotation = tf.reshape(annotation, (height, width))
	
	image, annotation = _resize_img(image, annotation)
	
	image, annotation = _random_flip(image, annotation)
	
	image = tf.image.random_brightness(image, 0.1)
	
	image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
	
	image = tf.image.random_hue(image, 0.05)
	
	image = tf.image.random_saturation(image, 0.5, 1.5)
	
	image = tf.cast(image, tf.float32)
	annotation = tf.cast(annotation, tf.int32)

	image = image- [123.68, 116.78, 103.94] 

	return image, annotation, features['name']
	
def _input_fn_map_eval(image):
	feature = {
		'image': tf.FixedLenFeature([], tf.string),
		'annotation': tf.FixedLenFeature([], tf.string),
		"height": tf.FixedLenFeature([], tf.int64),
		"width": tf.FixedLenFeature([], tf.int64),
		"name": tf.FixedLenFeature([], tf.string),
	}
	features = tf.parse_single_example(image, features=feature)
	
	image = tf.decode_raw(features['image'], tf.uint8)
	annotation = tf.decode_raw(features['annotation'], tf.uint8)
	
	height = tf.cast(features['height'], tf.int32)
	width = tf.cast(features['width'], tf.int32)
	
	image = tf.reshape(image, (height, width, 3))
	annotation = tf.reshape(annotation, (height, width))
	
	image, annotation = _resize_img(image, annotation)
	
	image = tf.cast(image, tf.float32)
	annotation = tf.cast(annotation, tf.int32)

	image = image- [123.68, 116.78, 103.94] 

	return image, annotation, features['name']
	
def input_fn_images(image_tfrecords_file, epochs=None, batch_size=1, is_training=True):
	ds=tf.data.TFRecordDataset(image_tfrecords_file)
	if is_training:
		ds=ds.map(_input_fn_map_train)
	else:
		ds=ds.map(_input_fn_map_eval)
	ds=ds.batch(batch_size)
	if epochs != None:
		ds = ds.repeat(epochs)
	iterator = ds.make_one_shot_iterator()
	inputs, annotations, names = iterator.get_next()
	return inputs, annotations, names

def input_fn_np(images_list, resize=False):
	images=[]
	for image_file in images_list:
		image=Image.open(image_file)
		if resize:
			image=image.resize((resized_width, resized_height))
		images.append(np.array(image))	
	
	images_ori = np.array(images)
	
	inputs=np.array(images)
	inputs = inputs - [123.68, 116.78, 103.94] #_R_MEAN = 123.68 _G_MEAN = 116.78 _B_MEAN = 103.94
	inputs=np.float32(inputs)
	
	return inputs, images_ori
	
	
if __name__ == '__main__':
	resized_height = 384
	resized_width = 384

	images_tfrecords_file_eval = 'images_eval.tfrecords'
	images_tfrecords_file_train = 'images_train.tfrecords'

	chk_imges_ds(images_tfrecords_file_eval)
	
	

