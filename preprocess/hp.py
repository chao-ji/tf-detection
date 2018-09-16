import tensorflow as tf

hparams = tf.contrib.training.HParams(
    label_dirs = (
        "/home/chaoji/Desktop/dense_prediction_data/PASCAL/TrainVal/VOCdevkit/VOC2007/Annotations",
        "/home/chaoji/Desktop/dense_prediction_data/PASCAL/TrainVal/VOCdevkit/VOC2012/Annotations",
        "/home/chaoji/Desktop/dense_prediction_data/PASCAL/Test/VOCdevkit/VOC2007/Annotations"),
    image_dirs = (
        "/home/chaoji/Desktop/dense_prediction_data/PASCAL/TrainVal/VOCdevkit/VOC2007/JPEGImages",
        "/home/chaoji/Desktop/dense_prediction_data/PASCAL/TrainVal/VOCdevkit/VOC2012/JPEGImages",
        "/home/chaoji/Desktop/dense_prediction_data/PASCAL/Test/VOCdevkit/VOC2007/JPEGImages"),
    image_sets_dirs = (
        "/home/chaoji/Desktop/dense_prediction_data/PASCAL/TrainVal/VOCdevkit/VOC2007/ImageSets/Main",
        "/home/chaoji/Desktop/dense_prediction_data/PASCAL/TrainVal/VOCdevkit/VOC2012/ImageSets/Main",
        "/home/chaoji/Desktop/dense_prediction_data/PASCAL/Test/VOCdevkit/VOC2007/ImageSets/Main"),
    split_fnames = ("trainval.txt",
                    "trainval.txt",
                    "test.txt")
)
