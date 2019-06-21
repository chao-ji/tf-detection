# Installation

Besides this repository, the following libraries or tools are required to be installed:
* [Protocol Buffer](#protocol-buffer)
* [COCO API](#coco-api) (if using COCO dataset) 
* [tf.slim](#installing-tf-slim)
* [PIL](#pil-library) (for drawing bounding boxes, masks and labels)


## Install this repository

You can clone this repository by running

```bash
git clone git@github.com:chao-ji/tf-detection.git detection
```

You need to make sure that the parent directory `mydir`

```
mydir
+---- detection
```

is on the Python search path list by adding
```bash
PYTHONPATH=$PYTHONPATH:/path_to_mydir/mydir
```

to file `.bashrc` or `.bash_profile` found in your home directory.

You should be able to run `import detection` in the Python interactive shell if installation was successful.


## Protocol Buffer
Object detection systems usually have a large number of hyperparameters and settings to configure (e.g. for data augmentation, anchor generation, target assignment etc.), which makes it cumbersome to pass them all as command line arguments. In this implementation, [Protocol Buffer](https://developers.google.com/protocol-buffers/) is used to manage the configuration of model settings -- You only need to prepare a few text files with extension `.config` ([examples](../configs/)) holding the parameters and settings you wish to be used, and pass the names of these text files as command line arguments.

Follow this [link](https://developers.google.com/protocol-buffers/docs/downloads) for instructions about downloading and installing Protocol Buffer compiler. You should be able to run `protoc --help` in the bash shell if installation was successful.


### Compile the Protbuf messages
Protocol Buffer specifications are text files with extension `.proto`, located under the folder `protos`. They store the schemas by which the `.config` text files are parsed. Protobuf messages must be compiled before being used.

To compile, run

```bash
mydir $ protoc detection/protos/*.proto --python_out=.
```
under the directory `mydir`.

You should get one `*_pb2.py` file for each `*.proto` file if compilation was successful.

After you compile the `.proto` files, you need to prepare `.config` files that contain proto buffer *messages* holding specific parameter settings that you want to take effect. See [this](Usage.md#configuring-the-models) for details.

## COCO API
[COCO API](https://github.com/cocodataset/cocoapi) is needed only if you wish to use the [COCO dataset](http://cocodataset.org/#download) or compute the [COCO detection metrics](http://cocodataset.org/#detection-eval). To install COCO API, run

```bash
git clone https://github.com/cocodataset/cocoapi.git
```
```bash
cd cocoapi/PythonAPI
```
```bash
PythonAPI $ make
```
Next, make a symbolic link to the folder `pycocotools` under `mydir`
```bash
ln -s pycocotools /path_to_mydir/mydir/pycocotools
```
You should be able to run `import pycocotools` in the Python interactive shell if installation was successful.


## Installing tf.slim
`tf.slim` is a library built on top of TensorFlow APIs that allows one to easily build common convolutional neural network architectures (e.g. VGG, ResNet, Inception etc.). In this repo, it is used to build feature extractors for object detection models.

Follow this [link](https://github.com/tensorflow/models/tree/master/research/slim#installation) for instructions about installing `tf.slim`.

You can download the [checkpoints](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models) holding the weights of ConvNets pre-trained for image classification tasks.

Currently the following models are supported as feature extractor:
* [resnet-v1-101](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz), for Faster R-CNN, Mask R-CNN
* [mobilenet-v2](//storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz), for SSD
* [inception-v2](http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz), for SSD, Faster R-CNN and Mask R-CNN


## PIL library

[Python Imaging Library](https://en.wikipedia.org/wiki/Python_Imaging_Library) (PIL) is a popular library that supports various operations on image files. It is recommended that you install it via [Anaconda](https://www.anaconda.com/).

If installation is successful, you should be able to run
```python
import PIL
```


