from os import path
import tensorflow as tf
from lxml import etree
import io
import PIL.Image
import hashlib

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecords')
flags.DEFINE_string('label_map_path', '', 'Path to label map proto')
flags.DEFINE_string('train_set', '', 'Path to train folder')
flags.DEFINE_string('validation_set', '', 'Path to validation folder')

FLAGS = flags.FLAGS


def create_tf_example(image, dataset_path):

    image_path = path.join(dataset_path, image.attrib['file'])
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_img = fid.read()
    encoded_img_io = io.BytesIO(encoded_img)
    img = PIL.Image.open(encoded_img_io)
    if img.format not in ['JPEG', 'PNG']:
        raise ValueError('Image format not JPEG or PNG')
    image_format = None
    if img.format == 'JPEG':
        image_format = b'jpeg'
    if img.format == 'PNG':
        image_format = b'png'
    key = hashlib.sha256(encoded_img).hexdigest()

    height = img.size[1]
    width = img.size[0]

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    boxes = image.xpath('.//box')
    for box in boxes:
        label = box.xpath('.//label')[0]
        if label.text == 'car':
            classes_text.append('car'.encode('utf8'))
            classes.append(1)
            xmins.append(float(box.attrib['left']) / width)
            xmaxs.append((float(box.attrib['left']) + float(box.attrib['width'])) / width)
            ymins.append(float(box.attrib['top']) / width)
            ymaxs.append((float(box.attrib['top']) + float(box.attrib['height'])) / height)

    if not classes:
        return None

    filename = image_path
    encoded_image_data = encoded_img

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def create_tf_record(dataset, record):
    writer = tf.python_io.TFRecordWriter(record)
    with tf.gfile.GFile(path.join(dataset, 'dataset.xml'), 'rb') as fid:
        xml_string = fid.read()
    xml_tree = etree.fromstring(xml_string)
    images = xml_tree.xpath('.//image')

    for image in images:
        example = create_tf_example(image, dataset)
        if example is not None:
            writer.write(example.SerializeToString())
    writer.close()


def main(_):

    create_tf_record(FLAGS.train_set, path.join(FLAGS.output_path, 'traffic_train.record'))
    create_tf_record(FLAGS.validation_set, path.join(FLAGS.output_path, 'traffic_validation.record'))


if __name__ == '__main__':
    tf.app.run()
