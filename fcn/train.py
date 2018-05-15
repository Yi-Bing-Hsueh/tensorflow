
import tensorflow as tf
import time
import os
import fcn

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

fcn.num_classes = 21
fcn.mask_out_class_label = 255
fcn.resized_height = 384
fcn.resized_width = 384
fcn.ckpt_path = 'fcn8s.ckpt'
fcn.vgg16_ckpt_path='vgg_16.ckpt'
fcn.dataset.resized_height = 384
fcn.dataset.resized_width = 384


ckpt_dir = '' # dir which ckpt_path is in
image_tfrecords_file_train = 'images_train.tfrecords'
image_tfrecords_file_eval = 'images_eval.tfrecords'
use_fcn_paper=False
batch_size = 4

cur_iou=0.0
cur_acc=0.0
i =0	
while True:
	if not os.listdir(ckpt_dir):
		retrain=False
		epochs = 2
	else:
		retrain=True
		epochs = 1
		
	fcn.train(image_tfrecords_file_train, epochs=epochs, batch_size=batch_size, retrain=retrain, restore_step=None, use_fcn_paper=use_fcn_paper)
	
	tf.reset_default_graph()
	iou, acc = fcn.eval(image_tfrecords_file_eval, batch_size=batch_size, use_fcn_paper=use_fcn_paper)
	tf.reset_default_graph()
	
	file=open('result.txt','a') 
	cur_time = time.strftime('%H:%M:%S', time.localtime())
	file.write("{}, {}, {}, {}, {}, {}\n".format(i, cur_time, 'light', epochs, iou, acc))
	file.close()
	
	if iou<cur_iou and acc<cur_acc:
		break
	
	cur_iou=iou
	cur_acc=acc
	i +=1
	
print('Training Done.')
	