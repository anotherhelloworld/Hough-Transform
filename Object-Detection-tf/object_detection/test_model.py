import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import ntpath
import scipy.misc
import time

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

sys.path.append("..")
from object_detection.utils import ops as utils_ops

from utils import label_map_util

from utils import visualization_utils as vis_util

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

flags = tf.app.flags
flags.DEFINE_string('path_to_ckpt', '',
                    'path to .pb')
flags.DEFINE_string('path_to_labels', '',
                    'path to .pbtxt')
flags.DEFINE_string('path_to_images_dir', '',
                    'path to images dir')
flags.DEFINE_string('output_dir', '',
                    'output directory')
flags.DEFINE_integer('num_classes', 3,
                     'number of classes')

FLAGS = flags.FLAGS


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def main(unused_argv):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(FLAGS.path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(FLAGS.path_to_labels)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=FLAGS.num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    TEST_IMAGE_PATHS =   [ os.path.join(FLAGS.path_to_images_dir, i) for i in os.listdir(FLAGS.path_to_images_dir) ]

    IMAGE_SIZE = (12, 8)

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            for image_path in TEST_IMAGE_PATHS:
                image = Image.open(image_path)
                image_np = load_image_into_numpy_array(image)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=1)
                scipy.misc.toimage(image_np).save(os.path.join(FLAGS.output_dir, ntpath.basename(image_path)))

    print(TEST_IMAGE_PATHS)


if __name__ == '__main__':
    tf.app.run()