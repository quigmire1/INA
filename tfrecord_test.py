import sys
import os
import time
import random
import tensorflow as tf
from models.slim.datasets.dataset_utils import _dataset_exists, _get_filenames_and_classes, write_label_file, _convert_dataset
init_time = time.time()


def measure():
    global init_time
    after_time = time.time()
    dif_time = after_time - init_time
    hour = int(dif_time / 3600)
    min = int((dif_time - hour * 3600) / 60)
    sec = dif_time - hour * 3600 - min * 60
    print
    'Processing Time: ' + str(hour) + "hour " + str(min) + "min " + str(sec) + "sec "


def writeOutput(listString, strOutputName):
    manipulatedData = open(strOutputName, 'w+');
    strNewRow = '\n'.join(listString);
    manipulatedData.write(strNewRow);
    manipulatedData.close();


def stringClensing(string):
    string = string.replace("\n", "");
    string = string.replace("\"", "");
    string = string.replace("\r", "");
    string = string.strip();
    # string = string.lower();
    return string;


def tfrecord_test():
    flags = tf.app.flags

    # State your dataset directory
    flags.DEFINE_string('dataset_dir', None, 'String: Your dataset directory')

    # Proportion of dataset to be used for evaluation
    flags.DEFINE_float('validation_size', 0.3,
                       'Float: The proportion of examples in the dataset to be used for validation')

    # The number of shards to split the dataset into.
    flags.DEFINE_integer('num_shards', 2, 'Int: Number of shards to split the TFRecord files into')

    # Seed for repeatability.
    flags.DEFINE_integer('random_seed', 0, 'Int: Random seed to use for repeatability.')

    # Output filename for the naming the TFRecord file
    flags.DEFINE_string('tfrecord_filename', None, 'String: The output filename to name your TFRecord file')

    FLAGS = flags.FLAGS

if __name__ == '__main__':
    measure()
