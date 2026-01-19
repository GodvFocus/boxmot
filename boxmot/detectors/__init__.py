# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from boxmot.utils import logger as LOGGER
from boxmot.utils.checks import RequirementsChecker

checker = RequirementsChecker()

ULTRALYTICS_MODELS = {"yolov8", "yolov9", "yolov10", "yolo11", "yolo12", "sam"}
RTDETR_MODELS = {"rtdetr_v2_r50vd", "rtdetr_v2_r18vd", "rtdetr_v2_r101vd"}
YOLOX_MODELS = {"yolox_n", "yolox_s", "yolox_m", "yolox_l", "yolox_x"}


def _check_model(name, markers):
    """Check if model name contains any of the markers."""
    return any(m in str(name) for m in markers)


def is_ultralytics_model(yolo_name):
    return _check_model(yolo_name, ULTRALYTICS_MODELS)


def is_yolox_model(yolo_name):
    return _check_model(yolo_name, YOLOX_MODELS)


def is_rtdetr_model(yolo_name):
    return _check_model(yolo_name, RTDETR_MODELS)


def is_rtdetr_ultralytics(yolo_name):
    """
    Check if model is Ultralytics RT-DETR (not HuggingFace transformers RT-DETR).
    
    Identifies RT-DETR models that should use the Ultralytics RTDETR class:
    - Official models with 'rtdetr-' prefix (e.g., rtdetr-l.pt, rtdetr-x.yaml)
    - Custom local .yaml/.yml files (assumed to be RT-DETR configs)
    - Local .pt files with 'rtdetr' in path/name
    
    Args:
        yolo_name: Model path or name
        
    Returns:
        bool: True if Ultralytics RT-DETR, False otherwise
    """
    from pathlib import Path
    
    name_str = str(yolo_name)
    path = Path(name_str)
    stem = path.stem.lower()
    
    # Exclude HuggingFace transformers RT-DETR models explicitly
    if stem in {m.lower() for m in RTDETR_MODELS}:
        return False
    
    # Official Ultralytics RT-DETR models (rtdetr-l, rtdetr-x, etc.)
    if "rtdetr-" in name_str.lower():
        return True
    
    # YAML/YML files in rt-detr config directory or containing 'rtdetr'
    if path.suffix.lower() in (".yaml", ".yml"):
        if "rtdetr" in name_str.lower() or "rt-detr" in name_str:
            return True
    
    # PT files containing 'rtdetr' in path or name
    if path.suffix.lower() == ".pt":
        if "rtdetr" in name_str.lower() or "rt-detr" in name_str:
            return True
    
    return False


def resolve_rtdetr_weights(yaml_path):
    """
    Find matching weights file for a RT-DETR YAML configuration.
    
    Searches for .pt weights file with the same stem as the YAML in:
    1. Same directory as YAML
    2. boxmot/ultralytics/cfg/models/rt-detr/
    3. Project root directory
    4. Current working directory
    
    Args:
        yaml_path: Path to YAML configuration file
        
    Returns:
        Path to weights file if found, None otherwise
    """
    from pathlib import Path
    
    yaml_path = Path(yaml_path)
    if yaml_path.suffix.lower() not in (".yaml", ".yml"):
        return None
    
    stem = yaml_path.stem
    weight_name = f"{stem}.pt"
    
    # Search paths in priority order
    search_paths = [
        yaml_path.parent / weight_name,  # Same directory as YAML
        Path("boxmot/ultralytics/cfg/models/rt-detr") / weight_name,  # Config directory
        Path(weight_name),  # Project root / current working directory
    ]
    
    for weight_path in search_paths:
        if weight_path.exists():
            LOGGER.info(f"Found matching weights for {yaml_path.name}: {weight_path}")
            return weight_path
    
    return None


def default_imgsz(yolo_name):
    if is_ultralytics_model(yolo_name) or is_rtdetr_ultralytics(yolo_name):
        return [640, 640]
    elif is_yolox_model(yolo_name):
        return [800, 1440]
    else:
        return [640, 640]


def get_yolo_inferer(yolo_model):
    """
    Determines and returns the appropriate inference strategy class based on the model name.
    Handles dependency checks and imports dynamically.
    """
    model_name = str(yolo_model)

    strategies = [
        (
            is_yolox_model,
            ("yolox", "tabulate", "thop"),
            {"yolox": ["--no-deps"]},
            "boxmot.detectors.yolox",
            "YoloXStrategy",
        ),
        (
            is_ultralytics_model,
            (),
            {},
            "boxmot.detectors.ultralytics",
            "UltralyticsStrategy",
        ),
        (
            is_rtdetr_model,
            ("transformers[torch]", "timm"),
            {},
            "boxmot.detectors.rtdetr",
            "RTDetrStrategy",
        ),
    ]

    for check_func, packages, extra_args, module_path, class_name in strategies:
        if check_func(model_name):
            for package in packages:
                try:
                    # Simple import check for package name (stripping version/extras)
                    pkg_name = package.split("[")[0].split("=")[0]
                    __import__(pkg_name)
                except ImportError:
                    args = extra_args.get(pkg_name, [])
                    checker.check_packages((package,), extra_args=args)

            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)

    LOGGER.error(f"Failed to infer inference mode from yolo model name: {model_name}")
    LOGGER.error("Supported models must contain one of the following:")
    LOGGER.error(f"  Ultralytics: {ULTRALYTICS_MODELS}")
    LOGGER.error(f"  RTDetr: {RTDETR_MODELS}")
    LOGGER.error(f"  YOLOX: {YOLOX_MODELS}")
    LOGGER.error(
        "By using these names, the default COCO-trained models will be downloaded automatically. "
        "For custom models, the filename must include one of these substrings to route it to the correct package and architecture."
    )
    exit()

