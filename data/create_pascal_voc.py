"""Create tfrecord files for PASCAL VOC 2007 and 2012.

Note the `pascal_home_dir` is the home directory of PASCAL VOC dataset

Example:
`pascal_home_dir` = "." 

            annotation files                          jpg images

2007 train  ./TrainVal/VOCdevkit/VOC2007/Annotations  ./TrainVal/VOCdevkit/VOC2007/JPEGImages  
2007 val    ./TrainVal/VOCdevkit/VOC2007/Annotations  ./TrainVal/VOCdevkit/VOC2007/JPEGImages
2012 train  ./TrainVal/VOCdevkit/VOC2012/Annotations  ./TrainVal/VOCdevkit/VOC2012/JPEGImages
2012 val    ./TrainVal/VOCdevkit/VOC2012/Annotations  ./TrainVal/VOCdevkit/VOC2012/JPEGImages
2007 test   ./Test/VOCdevkit/VOC2007/Annotations      ./Test/VOCdevkit/VOC2007/JPEGImages   

The text files containing image IDs (train.txt, val.txt, test.txt) for 
2007 trainval, 2012 trainval, 2007 test should be located in
./TrainVal/VOCdevkit/VOC2007/ImageSets/Main
./TrainVal/VOCdevkit/VOC2012/ImageSets/Main
./Test/VOCdevkit/VOC2007/ImageSets/Main


Make sure you download and decompress the files according to directory layout. 
"""
import os

import tensorflow as tf
import lxml.etree


ignore_difficult = False 


flags = tf.app.flags
flags.DEFINE_string('pascal_home_dir', None, 'path to home directory of PASCAL VOC.')
flags.DEFINE_bool('ignore_difficult', True, 'whether to ignore difficult instances.')

#pascal_home_dir = '/home/chaoji/Desktop/dense_prediction_data/PASCAL'

FLAGS = flags.FLAGS

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
"dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

label_map = {c: i + 1 for i, c in enumerate(classes)}

def normalize_bbox(ymin, xmin, ymax, xmax, height, width):
  ymin = ymin / height
  xmin = xmin / width
  ymax = ymax / height
  xmax = xmax / width

  return ymin, xmin, ymax, xmax

def _bytes_feature(value):
  if isinstance(value, bytes):
    value = [value]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
  if isinstance(value, float):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  if isinstance(value, int):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def main(_):

  label_dirs = [
      "TrainVal/VOCdevkit/VOC2007/Annotations",
      "TrainVal/VOCdevkit/VOC2007/Annotations",
      "TrainVal/VOCdevkit/VOC2012/Annotations",
      "TrainVal/VOCdevkit/VOC2012/Annotations",
      "Test/VOCdevkit/VOC2007/Annotations"]
  image_dirs = [
      "TrainVal/VOCdevkit/VOC2007/JPEGImages",
      "TrainVal/VOCdevkit/VOC2007/JPEGImages",
      "TrainVal/VOCdevkit/VOC2012/JPEGImages",
      "TrainVal/VOCdevkit/VOC2012/JPEGImages",
      "Test/VOCdevkit/VOC2007/JPEGImages"]
  image_sets_dirs = [
      "TrainVal/VOCdevkit/VOC2007/ImageSets/Main",
      "TrainVal/VOCdevkit/VOC2007/ImageSets/Main",
      "TrainVal/VOCdevkit/VOC2012/ImageSets/Main",
      "TrainVal/VOCdevkit/VOC2012/ImageSets/Main",
      "Test/VOCdevkit/VOC2007/ImageSets/Main"]
  split_fnames = ["train.txt",
                  "val.txt",
                  "train.txt",
                  "val.txt",
                  "test.txt"]

  for i in range(len(label_dirs)):
    label_dirs[i] = os.path.join(FLAGS.pascal_home_dir, label_dirs[i])

  for i in range(len(image_dirs)):
    image_dirs[i] = os.path.join(FLAGS.pascal_home_dir, image_dirs[i])

  for i in range(len(image_sets_dirs)):
    image_sets_dirs[i] = os.path.join(FLAGS.pascal_home_dir, image_sets_dirs[i])

  filenames = [ ('VOC2007_train', ([], [])), 
                ('VOC2007_val', ([], [])), 
                ('VOC2012_train', ([], [])), 
                ('VOC2012_val', ([], [])), 
                ('VOC2007_test', ([], []))]


  for i, (label_dir, image_dir, image_sets_dir, split_fname, filename_tuple) in (
      enumerate(zip(label_dirs, image_dirs, image_sets_dirs, split_fnames, filenames))):
    basenames = [line.strip() for line in open(os.path.join(image_sets_dir, split_fname))]
    label_filenames = [os.path.join(label_dir, bn + '.xml') for bn in basenames]
    image_filenames = [os.path.join(image_dir, bn + '.jpg') for bn in basenames]
    filenames[i][1][0].extend(label_filenames)
    filenames[i][1][1].extend(image_filenames)

  filenames = dict(filenames)

  for k, v in filenames.items():
    output_filename = k + '.tfrecord' if ignore_difficult else k + '_w_difficult.tfrecord'

    with tf.python_io.TFRecordWriter(output_filename) as writer:
      for label_fname, image_fname in list(zip(v[0], v[1])):
        doc = lxml.etree.parse(label_fname)
        objects = doc.xpath('/annotation/object')

        height = float(doc.xpath('/annotation/size/height')[0].text)
        width = float(doc.xpath('/annotation/size/width')[0].text)

        ymin_list = []
        xmin_list = []
        ymax_list = []
        xmax_list = []
        labels = []

        for obj in objects:
          if ignore_difficult and obj.xpath('difficult')[0].text == '1':
            continue

          label = obj.xpath('name')[0].text
          ymin = float(obj.xpath('bndbox/ymin')[0].text)
          xmin = float(obj.xpath('bndbox/xmin')[0].text)
          ymax = float(obj.xpath('bndbox/ymax')[0].text)
          xmax = float(obj.xpath('bndbox/xmax')[0].text)

          ymin, ymax = ymin / height, ymax / height
          xmin, xmax = xmin / width, xmax / width

          ymin_list.append(ymin)
          xmin_list.append(xmin)
          ymax_list.append(ymax)
          xmax_list.append(xmax)
          labels.append(label_map[label])

        with open(image_fname, 'rb') as fid:
          image_encoded = fid.read()

        example = tf.train.Example(features=tf.train.Features(feature={
            'image/encoded': _bytes_feature(image_encoded),
            'image/object/bbox/ymin': _float_feature(ymin_list),
            'image/object/bbox/xmin': _float_feature(xmin_list),
            'image/object/bbox/ymax': _float_feature(ymax_list),
            'image/object/bbox/xmax': _float_feature(xmax_list),
            'image/object/class/label': _int64_feature(labels)}))

        writer.write(example.SerializeToString())
      print(len(v[0]))
      
      
if __name__ == '__main__':
  tf.flags.mark_flag_as_required('pascal_home_dir')

  tf.app.run()
