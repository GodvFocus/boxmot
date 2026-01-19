try:
    import swattention
    from boxmot.ultralytics.nn.backbone.TransNeXt.TransNext_cuda import *
except ImportError as e:
    from boxmot.ultralytics.nn.backbone.TransNeXt.TransNext_native import *
    pass