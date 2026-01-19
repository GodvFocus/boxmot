# Ultralytics YOLO 🚀, AGPL-3.0 license

from boxmot.ultralytics.models.yolo import classify, detect, pose, segment

from .model import YOLO

__all__ = 'classify', 'segment', 'detect', 'pose', 'YOLO'
