import numpy as np
import matplotlib.pyplot as plt


def visualize_detections(detection_dict,
                         label_map,
                         color_map,
                         text_color='red',
                         score_thresh=.5):
  """Visualizes detected boxes, class, and score.

  Args:

  Returns:
      
  """
  image = np.copy(detection_dict['image'])
  height, width, _ = image.shape
  scores = detection_dict['scores']
  boxes = detection_dict['boxes']
  classes = detection_dict['classes']

  detection_indices = scores >= score_thresh
  scores = scores[detection_indices]
  boxes = boxes[detection_indices]
  classes = classes[detection_indices]

  num_valid_detections = scores.shape[0]

  for i in range(num_valid_detections):
    ymin, xmin, ymax, xmax = boxes[i].astype(np.int32)
    ymax = np.minimum(height - 1, ymax)
    xmax = np.minimum(width - 1, xmax)

    color = color_map[classes[i] - 1]

    image[ymin:ymax, xmin, :] = color
    image[ymin:ymax, xmax, :] = color
    image[ymin, xmin:xmax, :] = color
    image[ymax, xmin:xmax, :] = color

    detection_label_text = '%s, %.2f' % (
        label_map[classes[i]], int(scores[i] * 100) / 100)

    plt.text(xmin, ymin - 5, detection_label_text, color=text_color)

  plt.imshow(image.astype(np.uint8))


def get_color_map(num_colors=256, normalized=False):
  """Gets color map.

  Args:
    
  Returns:
    
  """
  def bitget(byteval, idx):
    return ((byteval & (1 << idx)) != 0)

  dtype = 'float32' if normalized else 'uint8'
  color_map = np.zeros((num_colors, 3), dtype=dtype)

  for i in range(num_colors):
    r = g = b = 0
    c = i
    for j in range(8):
      r = r | (bitget(c, 0) << 7 - j)
      g = g | (bitget(c, 1) << 7 - j)
      b = b | (bitget(c, 2) << 7 - j)
      c = c >> 3

    color_map[i] = np.array([r, g, b])

  color_map = color_map/255 if normalized else color_map
  return color_map

