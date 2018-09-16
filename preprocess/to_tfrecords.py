from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import lxml.etree
import glob
from hp import *
from scipy.misc import imread
import io

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

label_dirs = hparams.label_dirs
image_dirs = hparams.image_dirs
image_sets_dirs = hparams.image_sets_dirs
split_fnames = hparams.split_fnames

splits = [(label_dir, image_dir, os.path.join(image_sets_dir, split_fname))
    for label_dir, image_dir, image_sets_dir, split_fname in
    zip(label_dirs, image_dirs, image_sets_dirs, split_fnames)]

train_label_image_fnames = []
for label_dir, image_dir, fname, in splits:
  with open(fname) as f:
    for line in f:
      bn = line.strip()
      train_label_image_fnames.append((
          os.path.join(label_dir, bn + '.xml'),
          os.path.join(image_dir, bn + '.jpg')))


val_label_image_fnames = sorted(glob.glob(os.path.join(splits[1][0], '*.xml'))), \
    sorted(glob.glob(os.path.join(splits[1][1], '*.jpg')))
val_label_image_fnames = zip(*val_label_image_fnames)

train_unique_fnames = set(list(zip(*train_label_image_fnames))[0])
val_label_image_fnames = [(label_fname, image_fname) 
    for label_fname, image_fname in val_label_image_fnames 
    if label_fname not in train_unique_fnames]
  



for label_image_fnames, tfrecords_fname in (
    (train_label_image_fnames, 'train.tfrecords'),
    (val_label_image_fnames, 'val.tfrecords')):
  with tf.python_io.TFRecordWriter(tfrecords_fname) as writer:

    for label_fname, image_fname in label_image_fnames:
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
    print(len(label_image_fnames))

