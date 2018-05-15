import fcn

fcn.num_classes = 21
fcn.mask_out_class_label = 255
fcn.resized_height = 384
fcn.resized_width = 384
fcn.ckpt_path = 'fcn8s.ckpt'
fcn.vgg16_ckpt_path='vgg_16.ckpt'
fcn.dataset.resized_height = 384
fcn.dataset.resized_width = 384

use_fcn_paper=False
image_tfrecords_file_eval = 'images_eval.tfrecords'
images_dir = 'VOC2012/JPEGImages'

fcn.infer(image_tfrecords_file_eval, images_dir, start=0, end=10, restore_step=None, use_fcn_paper=use_fcn_paper)