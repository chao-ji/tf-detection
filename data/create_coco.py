"""Create tfrecord files for COCO 2017 dataset.
"""

import os
import json

import io
from pycocotools import mask
from PIL import Image
import numpy as np

import tensorflow as tf



#val2017_images_dir = '/home/chaoji/Desktop/dense_prediction_data/COCO/val2017'
#val2017_annotation_file = '/home/chaoji/Desktop/dense_prediction_data/COCO/annotations/instances_val2017.json'
#train2017_images_dir = '/home/chaoji/Desktop/dense_prediction_data/COCO/train2017'
#train2017_annotation_file = '/home/chaoji/Desktop/dense_prediction_data/COCO/annotations/instances_train2017.json'


flags = tf.app.flags

flags.DEFINE_string('val2017_images_dir', None, 
    'directory to the images from COCO 2017 val split.')
flags.DEFINE_string('val2017_annotation_file', None, 
    'path to json file holding annotations of COCO 2017 val split.')
flags.DEFINE_string('train2017_images_dir', None,
    'directory to the images from COCO 2017 train split.')
flags.DEFINE_string('train2017_annotation_file', None,
    'path to json file holding annotations of COCO 2017 train split.')
flags.DEFINE_bool('include_masks', False, 'whether to include instance masks.')

FLAGS = flags.FLAGS


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


def process_per_image(image, annotation_list, images_dir, include_masks=False):
  image_height = image['height']
  image_width = image['width']
  filename = image['file_name']

  ymin_list = []
  xmin_list = []
  ymax_list = []
  xmax_list = []
  labels = []
  is_crowd = []
  encoded_mask_png = []
  area_list = []

  num_annotations_skipped = 0
  for annotation in annotation_list:
    x, y, width, height = tuple(annotation['bbox'])
    if width <= 0 or height <= 0:
      num_annotations_skipped += 1
      continue
    if x + width > image_width or y + height > image_height:
      num_annotations_skipped += 1
      continue
    ymin_list.append(float(y) / image_height)
    xmin_list.append(float(x) / image_width)
    ymax_list.append(float(y + height) / image_height)
    xmax_list.append(float(x + width) / image_width)
    labels.append(int(annotation['category_id']))
    is_crowd.append(annotation['iscrowd'])
    area_list.append(annotation['area'])

    if include_masks:
      run_len_encoding = mask.frPyObjects(annotation['segmentation'], image_height, image_width) 
      binary_mask = mask.decode(run_len_encoding)
      if not annotation['iscrowd']:
        binary_mask = np.amax(binary_mask, axis=2)
      pil_image = Image.fromarray(binary_mask)
      output_io = io.BytesIO()
      pil_image.save(output_io, format='PNG')
      tmp = output_io.getvalue()
      encoded_mask_png.append(tmp)


  filename = os.path.join(images_dir, filename)
  with open(filename, 'rb') as fid:
    image_encoded = fid.read()


  feature = {
      'image/height': _int64_feature(image_height),
      'image/width': _int64_feature(image_width),
      'image/encoded': _bytes_feature(image_encoded),
      'image/object/bbox/ymin': _float_feature(ymin_list),
      'image/object/bbox/xmin': _float_feature(xmin_list),
      'image/object/bbox/ymax': _float_feature(ymax_list),
      'image/object/bbox/xmax': _float_feature(xmax_list),
      'image/object/class/label': _int64_feature(labels),
      'image/object/is_crowd': _int64_feature(is_crowd),
      'image/object/area': _float_feature(area_list)}

  if include_masks:
    feature['image/object/mask'] = _bytes_feature(encoded_mask_png)

  example = tf.train.Example(features=tf.train.Features(feature=feature))

  return example, num_annotations_skipped


def create_split(images_dir, annotation_file, output_path, num_per_shard, include_masks=False):
  data = json.load(open(annotation_file))

  images = data['images']
  annotations = data['annotations']
  annotations_index = {}

  for annotation in annotations:
    image_id = annotation['image_id']
    if image_id not in annotations_index:
      annotations_index[image_id] = []
    annotations_index[image_id].append(annotation)

  missing_annotation_count = 0
  for image in images:
    image_id = image['id']
    if image_id not in annotations_index:
      missing_annotation_count += 1
      annotations_index[image_id] = []

  print('missing_annotation_count', missing_annotation_count)


  file_index = 0
  num_examples = len(images)
  num_shards = int(np.ceil(num_examples / num_per_shard))
  print('num_shards', num_shards)


  for shard_id in range(num_shards):
    start_index = shard_id * num_per_shard
    end_index = min((shard_id + 1) * num_per_shard, num_examples)
    print('start_index', start_index, 'end_index', end_index)

    writer = tf.python_io.TFRecordWriter('%s-%05d-%05d.tfrecord' % (output_path, shard_id, num_shards))
    images_shard = images[start_index:end_index]  

 
    total_num_annotations_skipped = 0
    for i, image in enumerate(images_shard):
      if i % 100 == 0:
        print('%d of %d' % (i, len(images_shard)))
      annotation_list = annotations_index[image['id']]
      example, num_annotations_skipped = process_per_image(image, annotation_list, images_dir, include_masks)
      total_num_annotations_skipped += num_annotations_skipped
      writer.write(example.SerializeToString())
    print('total_num_annotations_skipped', total_num_annotations_skipped)
    print('Done processing shard %d out of %d.' % (shard_id + 1, num_shards), '\n')
    writer.close()


def main(_):
  create_split(FLAGS.val2017_images_dir, FLAGS.val2017_annotation_file, 'val2017', 1000, True)
  create_split(FLAGS.train2017_images_dir, FLAGS.train2017_annotation_file, 'train2017', 11829, True)
  

if __name__ == '__main__':
  tf.flags.mark_flag_as_required('val2017_images_dir')
  tf.flags.mark_flag_as_required('val2017_annotation_file')
  tf.flags.mark_flag_as_required('train2017_images_dir')
  tf.flags.mark_flag_as_required('train2017_annotation_file')

  tf.app.run()
