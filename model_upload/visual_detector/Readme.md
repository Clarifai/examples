# Visual Detection Model Examples

These can be used on the fly with minimal or no changes to test deploy visual detection models to the Clarifai platform. See the required files section for each model below and deployment instruction.

## [yolof](./yolof/)

[YOLOF](https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc3/configs/yolof) Requirements to run tests locally:

Download checkpoint and save it in `yolof/config/`:

```bash
$ wget -P yolof/config https://download.openmmlab.com/mmdetection/v2.0/yolof/yolof_r50_c5_8x8_1x_coco/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth
```

Install dependecies to test locally

```bash
$ pip install -r yolof/requirements.txt
```

## Torch serve model format  [faster-rcnn_torchserve](./faster-rcnn_torchserve/)

To utilize a Torch serve model (.mar file) created by running torch-model-archiver – essentially a zip file containing the model checkpoint, Python code, and other components – within this module, follow these steps:

1. Unzip the .mar file to obtain your checkpoint.
2. Implement your postprocess method in inference.py.

For example: [Faster-RCNN example](https://github.com/pytorch/serve/tree/master/examples/object_detector/fast-rcnn), suppose you already have .mar file following the torch serve example

unzip it to `./faster-rcnn_torchserve/model_store/hub/checkpoints` as the Torch cache is configured to use this folder in torch serve inference.py.

```bash
$ unzip faster_rcnn.mar -d ./faster-rcnn_torchserve/model_store/hub/checkpoints/
```

```bash
# in model_store/hub/checkpoints you will have
model_store/hub/checkpoints/
├── MAR-INF
│   └── MANIFEST.json
├── model.py
└── fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
```

Install dependecies to test locally

```bash
$ pip install -r faster-rcnn_torchserve/requirements.txt
```


## Deploy the model to Clarifai

Steps to deploy one of above examples after downloading weights and testing to the Clarifai platform.

>Note: set `--no-test` flag for `build` and `upload` command to disable testing

1. Build

```bash
$ clarifai build model <path/to/folder> # either `faster-rcnn_torchserve` or `yolof`
```

upload `*.clarifai` file to storage to obtain direct download url

2. Upload

```bash
$ clarifai upload model <path/to/folder> --url <your_url> 
```