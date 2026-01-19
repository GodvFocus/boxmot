# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = '8.0.201'

from boxmot.ultralytics.models import RTDETR, SAM, YOLO
from boxmot.ultralytics.models.fastsam import FastSAM
from boxmot.ultralytics.models.nas import NAS
from boxmot.ultralytics.utils import SETTINGS as settings
from boxmot.ultralytics.utils.checks import check_yolo as checks
from boxmot.ultralytics.utils.downloads import download

__all__ = '__version__', 'YOLO', 'NAS', 'SAM', 'FastSAM', 'RTDETR', 'checks', 'download', 'settings'
