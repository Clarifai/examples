
_base_ = '/mmdetection/configs/yolof/yolof_r50_c5_8x8_1x_coco.py'
runner = dict(type='EpochBasedRunner', max_epochs=10)
model=dict(
  bbox_head=dict(num_classes=0),
  )
data=dict(
  train=dict(
    ann_file='',
    img_prefix='',
    classes=''
    ),
  val=dict(
    ann_file='',
    img_prefix='',
    classes=''))
