import tensorflow as tf
from google.protobuf import text_format
from detection.protos import anchor_generator_pb2
import anchor_generator_builder 
from scipy.misc import imread 
import matplotlib.pyplot as plt
import numpy as np

def tile_anchors(grid_height,
                 grid_width,
                 scales,
                 aspect_ratios,
                 base_anchor_size,
                 anchor_stride,
                 anchor_offset):
  ratio_sqrts = np.sqrt(aspect_ratios)
  heights = scales / ratio_sqrts * base_anchor_size[0]
  widths = scales * ratio_sqrts * base_anchor_size[1]

  y_centers = np.arange(grid_height)
  y_centers = y_centers * anchor_stride[0] + anchor_offset[0]
  x_centers = np.arange(grid_width)
  x_centers = x_centers * anchor_stride[1] + anchor_offset[1]
  x_centers, y_centers = np.meshgrid(x_centers, y_centers)

  widths_grid, x_centers_grid = np.meshgrid(widths, x_centers)
  heights_grid, y_centers_grid = np.meshgrid(heights, y_centers)

  bbox_centers = np.stack([y_centers_grid, x_centers_grid], axis=2)
  bbox_sizes = np.stack([heights_grid, widths_grid], axis=2)

  bbox_centers = bbox_centers.reshape((-1, 2))
  bbox_sizes = bbox_sizes.reshape((-1, 2))


  return np.hstack([bbox_centers - 0.5 * bbox_sizes, bbox_centers + 0.5 * bbox_sizes]).astype(np.float32)


config = anchor_generator_pb2.AnchorGenerator()


config = text_format.Merge(open('../protos/anchor_generator.config').read(), config)

generator = anchor_generator_builder.build(config)

al = generator.generate([(19, 19), (10, 10), (5, 5), (3, 3), (2, 2), (1, 1)], im_height=300, im_width=300)


h, w = 10., 10.
layer = 1
num_boxes_per_cell = 6

result = tile_anchors(h, w, generator._scales[layer], generator._aspect_ratios[layer], [1., 1.], [1/h, 1/w], [1/h/2, 1/w/2])




img = imread('/home/chaoji/Desktop/009782.jpg')

img = tf.image.resize_images(img, [300, 300], method=tf.image.ResizeMethod.BILINEAR)


#img = tf.image.draw_bounding_boxes(tf.expand_dims(img, axis=0), tf.expand_dims(al[4].get(), axis=0))

sess = tf.InteractiveSession()

img, boxes = sess.run([img, al[layer].get()])

y = (boxes[:, 2] + boxes[:, 0]) / 2
x = (boxes[:, 3] + boxes[:, 1]) / 2

print()
print(len(y))
print(len(x))

y = [int(y[i] * 300) for i in np.arange(0, len(y), num_boxes_per_cell)]
x = [int(x[i] * 300) for i in np.arange(0, len(x), num_boxes_per_cell)]

for yy, xx in zip(y, x):
  img[yy, xx] = np.array([255, 0, 0])

plt.imshow(img.astype(np.uint8))
