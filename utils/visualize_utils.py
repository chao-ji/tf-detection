import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

TEXT_COLOR = 255, 255, 255


def visualize_detections(detection_result,
                         label_map,
                         font_path=None,
                         font_size=10,
                         line_width=1,
                         score_thresh=.5,
                         max_num_viz=None):
  """Draw detected boxes in the input image, annotated with class and 
  confidence score.

  Args:
    detection_result: a dict mapping from strings to numpy arrays, holding
      the following entries:
      { 'image': uint8 numpy array of shape [height, width, depth],
        'boxes': float numpy array of shape [num_detections, 4],
        'scores': float numpy array of shape [num_detections],
        'classes': int numpy array of shape [num_detections]}
    label_map: a dict mapping from int (class index) to string (class name).
    font_path: string scalar, path to the '*.ttf' font file. If None, a the
      default ImageFont is used.
    font_size: int scalar, font size.
    line_width: int scalar, line width.
    score_thresh: float scalar, only detections with score >= `score_thresh`
      will be drawn.
    max_num_viz: int scalar, max num of boxes to be visualized.

  Returns:
    image: uint8 numpy array of shape [height, width, depth], same as input
      image except with annotatation. 
  """
  image = np.copy(detection_result['image'])
  height, width = image.shape[:2]

  scores = detection_result['scores']
  boxes = detection_result['boxes']
  classes = detection_result['classes']

  detection_indices = scores >= score_thresh
  scores = scores[detection_indices]
  boxes = boxes[detection_indices]
  classes = classes[detection_indices]

  num_detections = scores.shape[0]
  if max_num_viz is not None:
    num_detections = np.minimum(num_detections, max_num_viz)
 
  color_map = get_color_map(len(label_map) + 1)
 
  for i in range(num_detections):
    ymin, xmin, ymax, xmax = boxes[i].astype(np.int32)
    ymin = np.maximum(0, ymin)
    xmin = np.maximum(0, xmin)
    ymax = np.minimum(height - 1, ymax)
    xmax = np.minimum(width - 1, xmax)
    color = color_map[classes[i]]

    image[ymin : ymax, 
          np.maximum(xmin - line_width // 2, 0) : 
          np.minimum(xmin + line_width - line_width // 2, width)] = color
    image[ymin : ymax, 
          np.maximum(xmax - line_width // 2, 0) : 
          np.minimum(xmax + line_width - line_width // 2, width)] = color
    image[np.maximum(ymin - line_width // 2, 0) : 
          np.minimum(ymin + line_width - line_width // 2, height), 
          xmin : xmax] = color
    image[np.maximum(ymax - line_width // 2, 0) : 
          np.minimum(ymax + line_width - line_width // 2, height), 
          xmin : xmax] = color

    detection_label_text = '%s: %.2f' % (
        label_map[classes[i]], int(scores[i] * 100) / 100)

    if font_path is not None: 
      font = ImageFont.truetype(font_path, size=font_size)
    else:
      font = ImageFont.load_default()
    text_width, text_height = font.getsize(detection_label_text)

    x = xmin
    y = np.maximum(ymin - text_height, 0)
    
    image[y : np.minimum(y + text_height, height), 
          x : np.minimum(x + text_width, width)] = color 

    img_obj = Image.fromarray(image)
    draw = ImageDraw.Draw(img_obj)
    draw.text((x, y), detection_label_text, TEXT_COLOR, font=font)
    image = np.array(img_obj)

  return image


def get_color_map(num_colors=256, normalized=False):
  """Creates color map.

  Args:
    num_colors: int scalar, total num of colors.
    normalized: bool scalar, whether RGB channel values are in the range of 
      [0, 1] float (True) or [0, 255] uint8 (False).
    
  Returns:
    color_map: numpy array of shape [num_colors, 3].    
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
