# [Clarifai Data Utils](https://github.com/Clarifai/clarifai-python-datautils)

Clarifai Data Utils offers various types of multimedia data utilities. Enhance your experience by seamlessly integrating these utilities with the Clarifai Python SDK.

## Installation


```bash
pip install clarifai-datautils
```

# Features
## Annotation Loader

A framework to load, export and analyze different annotated datasets.

### Supported Formats

| Annotation format                                                                                | Format       |      TASK       |
| ------------------------------------------------------------------------------------------------ | -------      | --------------- |
| [ImageNet](http://image-net.org/)                                                                | imagenet     | classification  |
| [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)                                          | cifar     | classification  |
| [MNIST](http://yann.lecun.com/exdb/mnist/)                                                       | mnist     | classification  |
| [VGGFace2](https://github.com/ox-vgg/vgg_face2)                                                  | vgg_face2     | classification  |
| [LFW](http://vis-www.cs.umass.edu/lfw/)                                                          | lfw     | classification  |
| [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/index.html)                  | voc_detection     | detection  |
| [YOLO](https://github.com/AlexeyAB/darknet#how-to-train-pascal-voc-data)                         | yolo     | detection  |
| [COCO](http://cocodataset.org/#format-data)                                                      | coco_detection     | detection  |
| [CVAT](https://opencv.github.io/cvat/docs/manual/advanced/xml_format/)                           | cvat     | detection  |
| [Kitti](http://www.cvlibs.net/datasets/kitti/index.php)                                          | kitti     | detection  |
| [LabelMe](http://labelme.csail.mit.edu/Release3.0)                                               | label_me     | detection  |
| [Open Images](https://storage.googleapis.com/openimages/web/download.html)                       | open_images     | detection  |
| [Clarifai](https://github.com/Clarifai/examples/tree/main/Data_Utils)                       | clarifai     | detection  |
| [COCO(segmentation)](http://cocodataset.org/#format-data)                                     | coco_segmentation     | segmentation  |
| [Cityscapes](https://www.cityscapes-dataset.com/)                                                | cityscapes     | segmentation  |
| [ADE](https://www.cityscapes-dataset.com/)                                                       | ade20k2017     | segmentation  |
