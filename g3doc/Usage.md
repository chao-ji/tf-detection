
# Usage

* [Preparing the input](#preparing-the-input)
* [Configuring the models](#configuring-the-models)
* [Run training, evaluation, inference, and visualization](#run-training-evaluation-inference-and-visualization)

## Preparing the input

This implementation assumes that the images and object instance annotations (groundtruth boxes, labels and masks) are in [TFRecord](https://medium.com/ymedialabs-innovation/how-to-use-tfrecord-with-datasets-and-iterators-in-tensorflow-with-code-samples-ffee57d298af) format (for training and evaluation purposes). If you only want to make inferences, you can leave your image in its original format (e.g. jpg).

Python scripts ([data/create_pascal_voc.py](../data/create_pascal_voc.py) and [data/create_coco.py](../data/create_coco.py)) are provided for converting Pascal VOC dataset and COCO dataset into TFRecord files.

### Pascal VOC dataset
The Pascal VOC dataset can be downloaded at:
* [Pascal VOC 2007 train/val splits](http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)
* [Pascal VOC 2007 test split](http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)
* [Pascal VOC 2012 train/val splits](http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)

After untar'ing these files, your folders should be organized like this:

```
pascal_data
+---------- TrainVal
|           +------- VOCdevkit
|                    +-------- VOC2007
|                    |         +------ Annotations
|                    |         |
|                    |         +------ JPEGImages
|                    |         |
|                    |         +------ ImageSets
|                    |
|                    +-------- VOC2012
|                              +------ Annotations
|                              |
|                              +------ JPEGImages
|                              |
|                              +------ ImageSets
|
+---------- Test
|           +--- VOCdevkit
|                +-------- VOC2007
|                          +------ Annotations
|                          |
|                          +------ JPEGImages
|                          |
|                          +------ ImageSets
```

Suppose that the full path to the parent directory `pascal_data` is `/path_to_pascal_data/pascal_data`. Run

```bash
python create_pascal_voc.py \
  --ignore_difficult=False \
  --pascal_home_dir=/path_to_pascal_data/pascal_data
```
to convert the train/val splits of Pascal VOC 2007 and 2012, and test split of Pascal VOC 2007 into `.tfrecord` files. Set the flag `--ignore_difficult=True` to exclude difficult instances.


### COCO dataset
The COCO dataset can be downloaded at

* [COCO 2017 train images](http://images.cocodataset.org/zips/train2017.zip)
* [COCO 2017 val images](http://images.cocodataset.org/zips/val2017.zip)
* [COCO 2017 trainval annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)


Suppose the paths to the images in train split and val split are `train2017` and `val2017`. Run

```
python data/create_coco.py \
  --include_masks=False \
  --train2017_annotation_file=instances_train2017.json \
  --train2017_images_dir=train2017 \
  --val2017_annotation_file=instances_val2017.json \
  --val2017_images_dir=val2017
```
to convert the train/val splits of COCO 2017 into `.tfrecord` files. Set the flag `--include_masks=True` to include instance masks to train and evaluate Mask R-CNN. 

## Configuring the models

The specifications for different modules of a object detection systesm are organized as protocol buffer messages. Their schemas are stored in `.proto` files. Once you compile these `.proto` files (see [this](Installation.md#protocol-buffer)), you need to prepare the following `.config` files for training, evaluating, inference and visualization:
* `label_map.config`: the mapping between object class index and object class name (e.g. aeroplane -> 1, bicycle -> 2, etc.). You must use the corresponding label map file for each dataset ([pascal_voc](../configs/pascal_label_map.config) and [COCO](../configs/coco_label_map.config))
* `model.config` (e.g. `faster_rcnn_model.config`, `ssd_model.config`): contains specifications for the neural network (i.e backbone network for feature extraction, hyperparameters for additional convolutional layers, etc.), box encoder, image resizer and normalizer, anchor generator, post processing, target assigner, losses
* `dataset.config`: contains specification for data transformation pipeline -- decoder that transform `.tfrecord` files in to tensor dicts, data augmentation options, data batching and padding.
* `train_config.config`: contains specifications for the optimizer, learning rate schedule, the list of input `.tfrecord` files for training, and the checkpoint of a classification model to restore variables from.
* `test_config.config`: contains the specifications for the evaluation metrics (i.e. PASCAL VOC, COCO box, COCO mask), the checkpoint of a trained object detector to restore variables from, also the options for visualization.

Examples can be found [here](../configs). You can modify the hyperparameters in these `.config` files for your specific setting.


## Run training, evaluation, inference, and visualization

### Training 
To train an object detector, run
```
  python run_trainer.py \
    --label_map_config_path=label_map.config \
    --model_config_path=model.config \
    --dataset_config_path=dataset.config \
    --train_config_path=train_config.config \
    --model_arch=MODEL_ARCH
```
`MODEL_ARCH` can be `ssd_model` or `faster_rcnn_model`. **Note**: Mask R-CNN essentially use the same architecture as Faster R-CNN (besides the mask prediction branch), so for Mask R-CNN `MODEL_ARCH` should be set to `faster_rcnn_model`.

* You need to specify the paths to the `.tfrecord` files for training using the `input_file` option in the `train_config.config` file. 
* The path to the checkpoint file holding the weights of backbone network pretrained for classification tasks must be specified in the `load_ckpt_path` field in `train_config.config`. 
* The checkpoint files holding the trained weights can be found in the folder specified by the `save_ckpt_path` option in the `model.config` file.


#### Resume training from a checkpoint of object detector
Most of the time, you want to start training from a backbone network pretrained for classification (e.g. ResNet). However, sometimes you may want to resume training from an existing network that has been trained for object detection, which contains additional layers compared to the classification network. In this case, you must set the `checkpoint_type` to `detection` rather than `classification`.

#### Bucketed batching
In the [original implementation](https://github.com/rbgirshick/py-faster-rcnn) of Faster R-CNN by Ross Girshick, the minibatch size is set to one image per minibatch. I think part of the reason is that the preprocessing pipeline does not reshape images into a fixed spatial extent (i.e. height and width), so we can't place images of different shapes in the same batch. 

In this implementation, however, batching of images of different shapes is made possible by padding the images to the same size. In addition, you can optionally use **bucketed batching**, which send images of similar size into the same bucket, so we can minimize the amount of padding needed to make images the same size. 

To use bucketed batching, set `bucketed_batching` of the `trainer_dataset` in `dataset.config` to `True`, and set `height_boundaries` and `width_boundaries` to the desired values. For example, if `height_boundaries` is set to `200` and `500`, and `width_boundaries` is set to `200` and `500`, the height and width dimension is divided into three intervals `[0, 200)`, `[200, 500)` and `[500, inf)`, which forms **nine buckets**, each of which store images of sizes that fall into that range. 
  
### Evaluation

```
  python run_evaluator.py \
    --label_map_config_path=label_map.config \
    --model_config_path=model.config \
    --dataset_config_path=dataset.config \
    --test_config_path=test_config.config \
    --model_arch=MODEL_ARCH
```
* You need to specify the paths to the `.tfrecord` files for evaluation using the `input_file` option in the `test_config.config` file. The evaluation metrics will be printed to stdout.

* The path to the checkpoint file holding the weights of a trained object detection network must be specified in the `load_ckpt_path` field in `test_config.config`. 

### Inference and Visualization

To make detection inference on input images using a trained model, run
```
  python run_inferencer.py \
    --label_map_config_path=label_map.config \
    --model_config_path=model.config \
    --dataset_config_path=dataset.config \
    --test_config_path=test_config.config \
    --model_arch=MODEL_ARCH
```
You need to place the input images (e.g. `.jpg`) under the folder specified by the `inference_directory` option in  the `test_config.config` file. The detected bounding boxes (i.e. coordinates, class labels, confidence scores, and optionally masks) are saved in `.npy` files, one for each `.jpg` image file with the same base name (i.e. `dog.npy` for `dog.jpg`).


To draw the detected boxes (and optionally masks) already inferred, run
```
  python run_visualizer.py \
    --label_map_config_path=faster_rcnn/label_map.config \
    --test_config_path=faster_rcnn/test_config.config
```
The images annotated with predicted boxes are named as `dog.jpg` for the `dog.npy` file.

