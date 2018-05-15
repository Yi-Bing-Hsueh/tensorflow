
import numpy as np 
import tensorflow as tf
from PIL import Image


images_list_file = '' #file of All images list. in pascal, it's trainval.txt
images_dir_voc = '' #images dir of pascal dataset, likes 'VOC2012/JPEGImages'
annotations_dir_voc = '' #annotations dir of pascal dataset, likes 'VOC2012/SegmentationClass'

#images_tfrecords_file_*: file to save tfrecords
def make_images_ds(images_tfrecords_file_train, images_tfrecords_file_eval):
	images_list_file_f = open(images_list_file, 'r')
	images_filename_list = [line for line in images_list_file_f]
	np.random.shuffle(images_filename_list)
	print(images_filename_list[0:10])
	val_images_filename_list = images_filename_list[:int(0.1*len(images_filename_list))]
	train_images_filename_list = images_filename_list[int(0.1*len(images_filename_list)):]

	def _dataset(images_filename_list, images_tfrecords_file):
		print(len(images_filename_list))	
		writer = tf.python_io.TFRecordWriter(images_tfrecords_file)
		for i in range(len(images_filename_list)):
			if not i % 500:
				print( 'set {}: {}/{}'.format(images_tfrecords_file, i, len(images_filename_list)))
			
			file_name = images_filename_list[i].strip()
			
			
			img = np.array(Image.open(images_dir_voc + '/' + file_name + ".jpg"))
			annotation = np.array(Image.open(annotations_dir_voc+'/'+file_name+'.png'))
			
			#for specific classes training
			# annotation[annotation==12]=2
			# annotation[annotation==8]=1
				
			height = img.shape[0]
			width = img.shape[1]
			img = img.tostring()
			annotation = annotation.tostring()
			file_name = bytes(file_name, encoding = "utf8")
				
			# Create a feature
			feature = {
				'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
				'annotation': tf.train.Feature(bytes_list=tf.train.BytesList(value=[annotation])),
				'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
				'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
				'name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[file_name])),
			}
			# Create an example protocol buffer
			example = tf.train.Example(features=tf.train.Features(feature=feature))
			# Serialize to string and write on the file
			writer.write(example.SerializeToString())
		writer.close()
		print('completed {}'.format(i))
	
	if images_tfrecords_file_train != None:
		_dataset(train_images_filename_list, images_tfrecords_file_train)

	if images_tfrecords_file_eval != None:
		_dataset(val_images_filename_list, images_tfrecords_file_eval)
		
if __name__ == '__main__':
	images_list_file = 'VOC2012/ImageSets/Segmentation/trainval.txt'
	images_dir_voc = 'VOC2012/JPEGImages'
	annotations_dir_voc = 'VOC2012/SegmentationClass'

	images_tfrecords_file_eval = 'images_eval.tfrecords'
	images_tfrecords_file_train = 'images_train.tfrecords'
	
	make_images_ds(images_tfrecords_file_train, images_tfrecords_file_eval)	
	
	