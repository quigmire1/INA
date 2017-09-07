import sys
import os
import time

import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging


sys.path.append("D:/Lab/workspace/INA/models/slim")
from datasets import imagenet
from nets import inception_resnet_v2
from preprocessing import inception_preprocessing

# from models.slim.datasets import imagenet
# from models.slim.nets import inception_resnet_v2
# from models.slim.preprocessing import inception_preprocessing

slim = tf.contrib.slim

init_time = time.time()

def measure():
    global init_time
    after_time = time.time()
    dif_time = after_time - init_time
    hour = int(dif_time / 3600)
    min = int((dif_time - hour * 3600) / 60)
    sec = dif_time - hour * 3600 - min * 60
    print('Processing Time: ' + str(hour) + "hour " + str(min) + "min " + str(sec) + "sec ")

# set your .ckpt file
checkpoints_dir = 'models/slim'

batch_size = 3
image_size = 299

with tf.Graph().as_default():
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        # set your image
        imgPath = 'mountain.jpg'
        testImage_string = tf.gfile.FastGFile(imgPath, 'rb').read()
        testImage = tf.image.decode_jpeg(testImage_string, channels=3)
        processed_image = inception_preprocessing.preprocess_image(testImage, image_size, image_size, is_training=False)
        processed_images = tf.expand_dims(processed_image, 0)

        logits, _ = inception_resnet_v2.inception_resnet_v2(processed_images, num_classes=1001, is_training=False)
        probabilities = tf.nn.softmax(logits)

        init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, 'inception_resnet_v2_2016_08_30.ckpt'),
            slim.get_model_variables('InceptionResnetV2'))

        with tf.Session() as sess:
            init_fn(sess)

            np_image, probabilities = sess.run([processed_images, probabilities])
            probabilities = probabilities[0, 0:]
            sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]

            names = imagenet.create_readable_names_for_imagenet_labels()
            for i in range(15):
                index = sorted_inds[i]
                print((probabilities[index], names[index]))

measure()
