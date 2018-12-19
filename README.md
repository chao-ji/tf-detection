### TensorFlow implementation of Object Detection ModelS

### Usage
[Protocol buffer](https://developers.google.com/protocol-buffers/) is used to manage 
all configuration settings.

First you need to download & install [proto buffer compiler](https://developers.google.com/protocol-buffers/docs/downloads), and compile the `*.proto` files by running
```
  protoc --python_out=. protos/*.proto
```
which generates `*_pb2.py` files in the `protos` directory.


##### To train a detection model, run

```
  python run_trainer.py \
    --label_map_config_path=label_map.config \
    --model_config_path=model.config \
    --dataset_config_path=fdataset.config \
    --train_config_path=train_config.config \
    --model_arch=MODEL_ARCH

```

##### To evaluate a trained detection model, run
```
  python run_evaluator.py \
    --label_map_config_path=label_map.config \
    --model_config_path=model.config \
    --dataset_config_path=dataset.config \
    --test_config_path=test_config.config \
    --model_arch=MODEL_ARCH
```

##### To make detection inference on input images using a trained model, run
```
  python run_inferencer.py \
    --label_map_config_path=label_map.config \
    --model_config_path=model.config \
    --dataset_config_path=dataset.config \
    --test_config_path=test_config.config \
    --model_arch=MODEL_ARCH
```

##### To draw/visualize the detected boxes already inferred, run
```
  python run_inferencer.py \
    --label_map_config_path=label_map.config \
    --model_config_path=model.config \
    --dataset_config_path=dataset.config \
    --test_config_path=test_config.config \
    --model_arch=MODEL_ARCH

```

### Experiments

#### PASCAL VOC 2007 test

##### Faster RCNN
mean Average Precisions = 0.7772

|-|plane|bike|bird|boat|bottle|bus|car|cat|chair|cow|table|dog|horse|mbike|person|plant|sheep|sofa|train|tvmonitor|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
AP|0.8095|0.8508|0.8202|0.6911|0.6144|0.8631|0.8818|0.9181|0.5212|0.8156|0.6111|0.9122|0.9036|0.8273|0.8531|0.4848|0.8275|0.7391|0.8552|0.7441|


Sample detections
<p align="center "><img src="g3doc/images/dining_room.jpg" width="500"></p>
