# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import argparse
import subprocess
from pathlib import Path
import numpy as np
from tqdm import tqdm
import configparser
import shutil
import json
import yaml
import cv2
import re
import os
import torch
import threading
import sys
import copy
import concurrent.futures
import traceback

from boxmot.trackers.tracker_zoo import create_tracker
from boxmot.utils import NUM_THREADS, ROOT, WEIGHTS, TRACKER_CONFIGS, DATASET_CONFIGS, logger as LOGGER, TRACKEVAL
from boxmot.utils.checks import RequirementsChecker
from boxmot.utils.torch_utils import select_device
from boxmot.utils.plots import MetricsPlotter
from boxmot.utils.misc import increment_path, prompt_overwrite
from boxmot.utils.clean import cleanup_mot17
from typing import Optional, List, Dict, Generator, Union

from boxmot.utils.dataloaders.MOT17 import MOT17DetEmbDataset
from boxmot.postprocessing.gsi import gsi

from boxmot.detectors import (
    default_imgsz,
    get_ultralytics_wrapper,
    get_yolo_inferer,
    is_rtdetr_model,
    is_ultralytics_model,
    is_yolox_model,
)
from boxmot.utils.mot_utils import convert_to_mot_format, write_mot_results
from boxmot.reid.core.auto_backend import ReidAutoBackend
from boxmot.utils.download import download_eval_data, download_trackeval

checker = RequirementsChecker()
checker.check_packages(('ultralytics', ))  # install


COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

TRACKEVAL_DETECTION_AVERAGE_KEY = "cls_comb_det_av"
OVERALL_RESULT_KEY = "overall"


def load_dataset_cfg(name: str) -> dict:
    """Load the dict from boxmot/configs/datasets/{name}.yaml."""
    path = DATASET_CONFIGS / f"{name}.yaml"
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def resolve_dataset_cfg_name(args: argparse.Namespace) -> Optional[str]:
    """解析当前运行应使用的数据集配置名，避免 benchmark 被重写后丢失 YAML 入口。"""
    candidates: list[str] = []

    dataset_cfg_name = getattr(args, "dataset_cfg_name", None)
    if dataset_cfg_name:
        candidates.append(str(dataset_cfg_name))

    benchmark = getattr(args, "benchmark", None)
    if benchmark:
        candidates.append(str(benchmark))

    source = getattr(args, "source", None)
    if source is not None:
        candidates.append(str(source))

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (DATASET_CONFIGS / f"{candidate}.yaml").exists():
            return candidate

    source_str = str(source) if source is not None else ""
    for config_file in DATASET_CONFIGS.glob("*.yaml"):
        if config_file.stem in source_str:
            return config_file.stem

    return None


def load_runtime_dataset_cfg(args: argparse.Namespace) -> dict:
    """按当前运行上下文加载数据集配置，统一处理 benchmark/source 的回退逻辑。"""
    cfg_name = resolve_dataset_cfg_name(args)
    if not cfg_name:
        return {}
    return load_dataset_cfg(cfg_name)


def eval_init(args,
              trackeval_dest: Path = TRACKEVAL,
              branch: str = "master",
              overwrite: bool = False
    ) -> None:
    """
    Common initialization: download TrackEval and (if needed) the MOT-challenge
    data for ablation runs, then canonicalize args.source.
    Modifies args in place.
    """
    # 1) download the TrackEval code
    download_trackeval(dest=trackeval_dest, branch=branch, overwrite=overwrite)

    # 2) if doing MOT17/20-ablation, pull down the dataset and rewire args.source/split
    if (DATASET_CONFIGS / f"{args.source}.yaml").exists():
        args.dataset_cfg_name = str(args.source)
        cfg = load_dataset_cfg(str(args.source))
        
        # Determine dataset destination
        if cfg["download"]["dataset_url"]:
            dataset_dest = TRACKEVAL / f"{Path(cfg['benchmark']['source']).name}.zip"
        else:
            # For custom datasets without URL, use the path from config if available, or default to assets
            dataset_dest = Path(cfg["download"].get("dataset_dest", f"assets/{Path(cfg['benchmark']['source']).name}"))

        download_eval_data(
            runs_url=cfg["download"]["runs_url"],
            dataset_url=cfg["download"]["dataset_url"],
            dataset_dest=dataset_dest,
            overwrite=overwrite
        )
        args.benchmark = Path(cfg["benchmark"]["source"]).name
        args.split = cfg["benchmark"]["split"]
        if cfg["download"]["dataset_url"]:
            args.source = TRACKEVAL / f"{args.benchmark}/{args.split}"
        elif "source" in cfg["benchmark"]:
            args.source = Path(cfg["benchmark"]["source"]) / args.split
        else:
            args.source = dataset_dest / args.split

    # 3) finally, make source an absolute Path everywhere
    args.source = Path(args.source).resolve()
    args.project = Path(args.project).resolve()
    args.project.mkdir(parents=True, exist_ok=True)


def generate_dets_embs(args: argparse.Namespace, y: Path, source: Path) -> None:
    """
    Generates detections and embeddings for the specified 
    arguments, YOLO model and source.

    Args:
        args (Namespace): Parsed command line arguments.
        y (Path): Path to the YOLO model file.
        source (Path): Path to the source directory.
    """
    WEIGHTS.mkdir(parents=True, exist_ok=True)

    if args.imgsz is None:
        args.imgsz = default_imgsz(y)

    seq_name = source.parent.name if source.name == "img1" else source.name
    source_frames = sorted(list(source.glob("*.jpg")) + list(source.glob("*.png")))
    LOGGER.info(
        "Generate start | seq={} | source={} | image_files={} | detector={} | reids={} | project={}",
        seq_name,
        source,
        len(source_frames),
        y,
        [r.name for r in args.reid_model],
        args.project,
    )

    ultralytics_wrapper = get_ultralytics_wrapper(y)
    yolo = ultralytics_wrapper(
        str(y) if is_ultralytics_model(y)
        else 'yolov8n.pt',
    )

    results = yolo(
        source=source,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=args.agnostic_nms,
        stream=True,
        device=args.device,
        verbose=False,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
    )

    if not is_ultralytics_model(y):
        m = get_yolo_inferer(y)
        yolo_model = m(model=y, device=yolo.predictor.device,
                       args=yolo.predictor.args)
        yolo.predictor.model = yolo_model

        # If current model is YOLOX or RTDetr, change the preprocess and postprocess
        if is_yolox_model(y) or is_rtdetr_model(y):
            # add callback to save image paths for further processing
            yolo.add_callback("on_predict_batch_start",
                              lambda p: yolo_model.update_im_paths(p))
            yolo.predictor.preprocess = (
                lambda im: yolo_model.preprocess(im=im))
            yolo.predictor.postprocess = (
                lambda preds, im, im0s:
                yolo_model.postprocess(preds=preds, im=im, im0s=im0s))

    reids = []
    for r in args.reid_model:
        reid_model = ReidAutoBackend(weights=r,
                                     device=yolo.predictor.device,
                                     half=args.half).model
        reids.append(reid_model)
        embs_path = args.project / 'dets_n_embs' / y.stem / 'embs' / r.stem / (seq_name + '.txt')
        embs_path.parent.mkdir(parents=True, exist_ok=True)
        embs_path.touch(exist_ok=True)

        if os.path.getsize(embs_path) > 0:
            open(embs_path, 'w').close()

    yolo.predictor.custom_args = args

    dets_path = args.project / 'dets_n_embs' / y.stem / 'dets' / (seq_name + '.txt')
    dets_path.parent.mkdir(parents=True, exist_ok=True)
    dets_path.touch(exist_ok=True)

    if os.path.getsize(dets_path) > 0:
        open(dets_path, 'w').close()

    with open(str(dets_path), 'ab+') as f:
        np.savetxt(f, [], fmt='%f', header=str(source))

    frames_seen = 0
    det_rows_written = 0
    empty_det_frames = 0
    class_counts: dict[int, int] = {}
    emb_rows_written = {r.stem: 0 for r in args.reid_model}

    for frame_idx, r in enumerate(tqdm(results, desc="Frames")):
        frames_seen += 1
        nr_dets = len(r.boxes)
        frame_idx = torch.full((1, 1), frame_idx + 1).repeat(nr_dets, 1)
        img = r.orig_img

        dets = np.concatenate(
            [
                frame_idx,
                r.boxes.xyxy.to('cpu'),
                r.boxes.conf.unsqueeze(1).to('cpu'),
                r.boxes.cls.unsqueeze(1).to('cpu'),
            ], axis=1
        )

        # Filter dets with incorrect boxes: (x2 < x1 or y2 < y1)
        boxes = r.boxes.xyxy.to('cpu').numpy().round().astype(int)
        boxes_filter = ((np.maximum(0, boxes[:, 0]) < np.minimum(boxes[:, 2], img.shape[1])) &
                        (np.maximum(0, boxes[:, 1]) < np.minimum(boxes[:, 3], img.shape[0])))
        dets = dets[boxes_filter]
        if dets.shape[0] == 0:
            empty_det_frames += 1
        else:
            for cls_id, count in zip(*np.unique(dets[:, 6].astype(int), return_counts=True)):
                class_counts[int(cls_id)] = class_counts.get(int(cls_id), 0) + int(count)
        det_rows_written += int(dets.shape[0])

        with open(str(dets_path), 'ab+') as f:
            np.savetxt(f, dets, fmt='%f')

        for reid, reid_model_name in zip(reids, args.reid_model):
            embs = reid.get_features(dets[:, 1:5], img)
            embs_path = args.project / "dets_n_embs" / y.stem / 'embs' / reid_model_name.stem / (seq_name + '.txt')
            with open(str(embs_path), 'ab+') as f:
                np.savetxt(f, embs, fmt='%f')
            emb_rows_written[reid_model_name.stem] += int(embs.shape[0])

    det_state = _cache_file_state(dets_path)
    emb_states = {
        r.stem: _cache_file_state(args.project / "dets_n_embs" / y.stem / 'embs' / r.stem / (seq_name + '.txt'))
        for r in args.reid_model
    }
    LOGGER.info(
        "Generate done | seq={} | frames_seen={} | image_files={} | det_rows_written={} | "
        "empty_det_frames={} | class_counts={} | det_file_rows={} | det_file_bytes={} | emb_rows_written={} | emb_states={}",
        seq_name,
        frames_seen,
        len(source_frames),
        det_rows_written,
        empty_det_frames,
        dict(sorted(class_counts.items())),
        det_state["rows"],
        det_state["bytes"],
        emb_rows_written,
        emb_states,
    )


def parse_mot_results(results: str) -> dict:
    """
    从 MOTChallenge 评估输出中提取各类别及跨类别 COMBINED 指标。

    返回以类别名为键的字典；TrackEval 的跨类别键保持原名，便于调用方明确选择
    class-averaged 或 detection-averaged 口径。
    """
    metric_specs = {
        'HOTA': (
            'HOTA:',
            {
                'HOTA': 0,
                'DetA': 1,
                'AssA': 2,
                'DetRe': 3,
                'DetPr': 4,
                'AssRe': 5,
                'AssPr': 6,
                'LocA': 7,
            },
        ),
        'MOTA': (
            'CLEAR:',
            {
                'MOTA': 0,
                'MOTP': 1,
                'CLR_Re': 3,
                'CLR_Pr': 4,
                'FN': 10,
                'FP': 11,
                'IDSW': 12,
            },
        ),
        'IDF1': ('Identity:', {'IDF1': 0, 'IDR': 1, 'IDP': 2, 'IDTP': 3, 'IDFN': 4, 'IDFP': 5}),
        'IDs': ('Count:', {'Dets': 0, 'GT_Dets': 1, 'IDs': 2, 'GT_IDs': 3}),
    }

    int_fields = {'FN', 'FP', 'IDSW', 'IDTP', 'IDFN', 'IDFP', 'Dets', 'GT_Dets', 'IDs', 'GT_IDs'}
    parsed_results = {}
    
    lines = results.splitlines()
    current_class = None
    current_metric_type = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for header lines
        is_header = False
        for metric_name, (prefix, _) in metric_specs.items():
            if line.startswith(prefix):
                is_header = True
                current_metric_type = metric_name
                
                # Format: "HOTA: tracker-classHOTA ..."
                content = line[len(prefix):].strip()
                first_word = content.split()[0]
                if first_word.endswith(metric_name):
                    tracker_class = first_word[:-len(metric_name)]
                    if '-' in tracker_class:
                        current_class = tracker_class.split('-')[-1]
                    else:
                        current_class = 'default'
                    
                    if current_class not in parsed_results:
                        parsed_results[current_class] = {}
                break
        
        if is_header:
            continue
        
        # Check for COMBINED row
        if line.startswith('COMBINED') and current_class and current_metric_type:
            fields = line.split()
            if len(fields) > 1:
                values = fields[1:] # Skip 'COMBINED'
                _, field_map = metric_specs[current_metric_type]
                for key, idx in field_map.items():
                    if idx < len(values):
                        val = values[idx]
                        parsed_results[current_class][key] = int(val) if key in int_fields else float(val)

    return parsed_results


def build_mot_results_summary(
    parsed_results: dict[str, dict],
    class_names: list[str],
) -> dict[str, dict]:
    """筛选指定类别，并将 TrackEval 的 detection-averaged 跨类别结果命名为 overall。

    VisDrone 官方评估会先汇总所有类别与序列的基础计数，再重算最终指标；
    TrackEval 的 ``cls_comb_det_av`` 与这一口径一致。``cls_comb_cls_av`` 是宏平均，
    不应作为 VisDrone 论文主结果。
    """
    class_results = {
        class_name: parsed_results[class_name]
        for class_name in class_names
        if class_name in parsed_results
    }
    if len(class_names) > 1 and TRACKEVAL_DETECTION_AVERAGE_KEY in parsed_results:
        return {
            OVERALL_RESULT_KEY: parsed_results[TRACKEVAL_DETECTION_AVERAGE_KEY],
            **class_results,
        }
    return class_results


def _sequence_img_dir(seq_dir: Path) -> Path:
    img1 = seq_dir / "img1"
    return img1 if img1.exists() else seq_dir


def _collect_seq_info(source: Path) -> tuple[list[Path], dict[str, int]]:
    seq_paths = []
    seq_info: dict[str, int] = {}

    # 遍历源目录下的所有子目录（代表每个视频序列）
    for seq_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        # 获取实际包含图像的目录（兼容 MOT 格式的 'img1' 子文件夹）
        img_dir = _sequence_img_dir(seq_dir)

        # 收集所有图像文件（jpg/png）以统计序列长度
        frame_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))

        # 如果没有找到图像，跳过该序列
        if not frame_files:
            continue

        # 记录有效的图像目录路径和对应的序列帧数
        seq_paths.append(img_dir)
        seq_info[seq_dir.name] = len(frame_files)
    return seq_paths, seq_info


def build_dataset_eval_settings(
    args: argparse.Namespace,
    gt_folder: Path,
    seq_info: dict[str, int],
) -> dict:
    """Derive dataset-specific evaluation settings (classes, ids, distractors, gt path format).

    This centralizes logic for MOT-style datasets and non-MOT layouts such as VisDrone.
    """

    cfg = {}
    try:
        cfg = load_runtime_dataset_cfg(args)
    except FileNotFoundError:
        cfg = {}
    except Exception as e:  # noqa: BLE001
        LOGGER.warning(f"Error loading dataset config: {e}")
        cfg = {}

    bench_cfg = cfg.get("benchmark", {}) if isinstance(cfg, dict) else {}
    eval_classes_cfg = bench_cfg.get("eval_classes") if isinstance(bench_cfg, dict) else None
    distractor_cfg = bench_cfg.get("distractor_classes") if isinstance(bench_cfg, dict) else None
    ignore_region_cfg = bench_cfg.get("ignore_region_classes") if isinstance(bench_cfg, dict) else None

    # Classes and ids
    classes_to_eval = ["person"]
    class_ids = [1]
    if isinstance(eval_classes_cfg, dict) and len(eval_classes_cfg) > 0:
        ordered = sorted(((int(k), v) for k, v in eval_classes_cfg.items()), key=lambda kv: kv[0])
        class_ids = [k for k, _ in ordered]
        classes_to_eval = [v for _, v in ordered]
    elif hasattr(args, "classes") and args.classes is not None:
        class_indices = args.classes if isinstance(args.classes, list) else [args.classes]
        classes_to_eval = [COCO_CLASSES[int(i)] for i in class_indices]
        class_ids = [int(i) + 1 for i in class_indices]

    # Distractors
    distractor_ids: list[int] = []
    if isinstance(distractor_cfg, dict) and len(distractor_cfg) > 0:
        distractor_ids = [int(k) for k in distractor_cfg.keys()]

    ignore_region_ids: list[int] = []
    if isinstance(ignore_region_cfg, dict):
        ignore_region_ids = [int(k) for k in ignore_region_cfg.keys()]
    elif isinstance(ignore_region_cfg, list):
        ignore_region_ids = [int(class_id) for class_id in ignore_region_cfg]
    ignore_ioa_threshold = float(bench_cfg.get("ignore_ioa_threshold", 0.5))

    # Remove duplicates while preserving order
    seen = set()
    pairs = []
    for name, cid in zip(classes_to_eval, class_ids):
        if name in seen:
            continue
        seen.add(name)
        pairs.append((name, cid))
    classes_to_eval = [name for name, _ in pairs]
    class_ids = [cid for _, cid in pairs]

    # GT path format (VisDrone uses flat txt under annotations)
    gt_loc_format = bench_cfg.get("gt_loc_format", "{gt_folder}/{seq}/gt/gt_temp.txt")
    is_visdrone = "visdrone" in getattr(args, "benchmark", "").lower() or "visdrone" in str(gt_folder).lower()
    if is_visdrone and "gt_loc_format" not in bench_cfg:
        gt_loc_format = "{gt_folder}/{seq}/gt/gt.txt"

    benchmark_name = getattr(args, "benchmark", "")

    return {
        "classes_to_eval": classes_to_eval,
        "class_ids": class_ids,
        "distractor_ids": distractor_ids,
        "ignore_region_ids": ignore_region_ids,
        "ignore_ioa_threshold": ignore_ioa_threshold,
        "gt_loc_format": gt_loc_format,
        "benchmark_name": benchmark_name,
        "seq_info": seq_info,
    }


def resolve_tracker_class_id_offset(args: argparse.Namespace) -> int:
    """根据数据集评估类别编号决定 tracker 输出是否需要从 0 基转成 1 基。"""
    cfg = {}
    try:
        cfg = load_runtime_dataset_cfg(args)
    except FileNotFoundError:
        cfg = {}
    except Exception as e:  # noqa: BLE001
        LOGGER.warning(f"Error loading dataset config for class id offset: {e}")
        cfg = {}

    bench_cfg = cfg.get("benchmark", {}) if isinstance(cfg, dict) else {}
    explicit_offset = bench_cfg.get("tracker_class_id_offset") if isinstance(bench_cfg, dict) else None
    if explicit_offset is not None:
        return int(explicit_offset)

    eval_classes_cfg = bench_cfg.get("eval_classes") if isinstance(bench_cfg, dict) else None
    if isinstance(eval_classes_cfg, dict) and eval_classes_cfg:
        class_ids = [int(k) for k in eval_classes_cfg.keys()]
        return 0 if min(class_ids) == 0 else 1
    return 1


def validate_tracker_class_id_alignment(
    tracker_precheck: dict[str, dict],
    class_id_offset: int,
) -> None:
    """拒绝与当前编号协议不一致的 tracker 缓存，避免静默生成错误指标。"""
    if class_id_offset <= 0:
        return

    invalid_classes = {}
    for seq_name, state in tracker_precheck.items():
        class_counts = state.get("tracker_class_counts", {})
        invalid = {
            int(class_id): int(count)
            for class_id, count in class_counts.items()
            if int(class_id) < class_id_offset
        }
        if invalid:
            invalid_classes[seq_name] = invalid

    if invalid_classes:
        raise ValueError(
            "Tracker 结果的类别编号与当前评估配置不一致："
            f"tracker_class_id_offset={class_id_offset}，但发现更小的类别 ID "
            f"{invalid_classes}。这些文件通常是旧的 0-based 缓存；请重新生成 MOT "
            "结果后再评估，不能直接用于论文指标。"
        )


def format_gt_path(gt_folder: Path, gt_loc_format: str, seq_name: str) -> Path:
    """按 TrackEval 的路径模板解析单个序列 GT 文件。"""
    return Path(gt_loc_format.format(gt_folder=str(gt_folder), seq=seq_name))


def trackeval(
    args: argparse.Namespace,
    seq_paths: list,
    save_dir: Path,
    gt_folder: Path,
    metrics: list = ["HOTA", "CLEAR", "Identity"],
    seq_info: Optional[dict] = None,
) -> str:
    """
    Executes a Python script to evaluate MOT challenge tracking results using specified metrics.

    Args:
        seq_paths (list): List of sequence paths.
        save_dir (Path): Directory to save evaluation results.
        gt_folder (Path): Folder containing ground truth data.
        metrics (list, optional): List of metrics to use for evaluation. Defaults to ["HOTA", "CLEAR", "Identity"].

    Returns:
        str: Standard output from the evaluation script.
    """

    if not seq_info:
        seq_names = [seq_path.parent.name if seq_path.name == "img1" else seq_path.name for seq_path in seq_paths]
        seq_info = {name: None for name in seq_names}

    seq_info_args = []
    for name in sorted(seq_info.keys()):
        length = seq_info[name]
        seq_info_args.append(f"{name}:{length}" if length else name)

    seq_image_sizes = {}
    for seq_path in seq_paths:
        seq_name = seq_path.parent.name if seq_path.name == "img1" else seq_path.name
        frame_files = sorted(list(seq_path.glob("*.jpg")) + list(seq_path.glob("*.png")))
        if not frame_files:
            continue
        frame = cv2.imread(str(frame_files[0]))
        if frame is None:
            LOGGER.warning("无法读取序列 {} 的首帧尺寸，ignore region 将跳过图像边界裁剪", seq_name)
            continue
        seq_image_sizes[seq_name] = frame.shape[:2]

    # 获取数据集配置
    dataset_settings = build_dataset_eval_settings(args, gt_folder, seq_info)
    classes_to_eval = dataset_settings["classes_to_eval"]
    class_ids = dataset_settings["class_ids"]
    distractor_ids = dataset_settings["distractor_ids"]
    ignore_region_ids = dataset_settings["ignore_region_ids"]
    ignore_ioa_threshold = dataset_settings["ignore_ioa_threshold"]
    gt_loc_format = dataset_settings["gt_loc_format"]
    benchmark_name = dataset_settings["benchmark_name"]

    cmd_args = [
        sys.executable, ROOT / 'boxmot' / 'utils' / 'run_mot_challenge.py',
        "--GT_FOLDER", str(gt_folder),
        "--BENCHMARK", benchmark_name,
        "--TRACKERS_FOLDER", str(args.exp_dir.parent),
        "--TRACKERS_TO_EVAL", args.exp_dir.name,
        "--SPLIT_TO_EVAL", args.split,
        "--METRICS", *metrics,
        "--USE_PARALLEL", "True",
        "--TRACKER_SUB_FOLDER", "",
        "--NUM_PARALLEL_CORES", str(4),
        "--SKIP_SPLIT_FOL", "True",
        "--GT_LOC_FORMAT", gt_loc_format,
        "--CLASSES_TO_EVAL", *classes_to_eval,
        "--CLASS_IDS", *[str(i) for i in class_ids],
        "--DISTRACTOR_CLASS_IDS", *[str(i) for i in distractor_ids],
        "--IGNORE_CLASS_IDS", *[str(i) for i in ignore_region_ids],
        "--IGNORE_IOA_THRESHOLD", str(ignore_ioa_threshold),
        "--SEQ_INFO", *seq_info_args
    ]
    if seq_image_sizes:
        cmd_args.extend(
            [
                "--SEQ_IMAGE_SIZES",
                *[
                    f"{name}:{height}:{width}"
                    for name, (height, width) in sorted(seq_image_sizes.items())
                ],
            ]
        )
    LOGGER.info("TrackEval command | {}", " ".join(str(arg) for arg in cmd_args))

    p = subprocess.Popen(
        args=cmd_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout, stderr = p.communicate()

    if stderr:
        LOGGER.error("TrackEval stderr | {}", stderr)
    LOGGER.info(
        "TrackEval finished | returncode={} | stdout_chars={} | stderr_chars={}",
        p.returncode,
        len(stdout or ""),
        len(stderr or ""),
    )
    if p.returncode != 0:
        raise RuntimeError(f"TrackEval failed with return code {p.returncode}. See TrackEval stderr above.")
    return stdout


def process_single_det_emb(y: Path, source_path: Path, opt: argparse.Namespace, lock):
    try:
        new_opt = copy.deepcopy(opt)
        # Check if img1 exists, otherwise use source_path directly
        img_source = source_path / 'img1'
        if not img_source.exists():
            img_source = source_path
            
        # Use lock to ensure model loading/downloading is thread-safe
        with lock:
            if is_ultralytics_model(y):
                get_ultralytics_wrapper(y)(str(y))

        generate_dets_embs(new_opt, y, source=img_source)
    except Exception:
        traceback.print_exc()
        raise


def _count_non_comment_rows(path: Path) -> Optional[int]:
    """统计缓存文件中的有效数据行数，用于定位 det/emb 是否真实对齐。"""
    if not path.exists():
        return None

    rows = 0
    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                rows += 1
    return rows


def _cache_file_state(path: Path) -> dict:
    """汇总单个缓存文件状态，避免下次只能从 TrackEval 的缺文件错误倒推。"""
    exists = path.exists()
    return {
        "exists": exists,
        "bytes": path.stat().st_size if exists else None,
        "rows": _count_non_comment_rows(path) if exists else None,
    }


def _mot_class_counts(path: Path) -> dict[int, int]:
    """统计 MOT 文本第 8 列类别分布，用于确认 GT 与 tracker 类别编号是否对齐。"""
    counts: dict[int, int] = {}
    if not path.exists():
        return counts
    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = re.split(r"[,\s]+", stripped)
            if len(parts) < 8:
                continue
            cls_id = int(float(parts[7]))
            counts[cls_id] = counts.get(cls_id, 0) + 1
    return dict(sorted(counts.items()))


def run_generate_dets_embs(opt: argparse.Namespace) -> None:
    # 扫描数据源
    source_root = Path(opt.source)
    mot_folder_paths = sorted([item for item in source_root.iterdir() if item.is_dir()])
    
    # Create a manager to share the lock across processes
    # 后续会使用多进程，需要加锁保证同一时间只有一个进程加载模型
    manager = mp.Manager()
    lock = manager.Lock()

    for y in opt.yolo_model:
        dets_folder = Path(opt.project) / 'dets_n_embs' / y.stem / 'dets'
        embs_folder = Path(opt.project) / 'dets_n_embs' / y.stem / 'embs' / opt.reid_model[0].stem
        LOGGER.info(
            "Cache root check | source={} | project={} | dets_dir={} | embs_dir={} | sequences={}",
            source_root,
            opt.project,
            dets_folder,
            embs_folder,
            len(mot_folder_paths),
        )

        # Filter out already processed sequences
        tasks = []
        for i, mot_folder_path in enumerate(mot_folder_paths):
            dets_path = dets_folder / (mot_folder_path.name + '.txt')
            embs_path = embs_folder / (mot_folder_path.name + '.txt')
            det_state = _cache_file_state(dets_path)
            emb_state = _cache_file_state(embs_path)
            cache_rows_match = (
                det_state["rows"] is not None
                and emb_state["rows"] is not None
                and det_state["rows"] == emb_state["rows"]
            )
            action = "regenerate"
            if dets_path.exists() and embs_path.exists():
                if not prompt_overwrite('Detections and Embeddings', dets_path, opt.ci):
                    action = "reuse"
                    LOGGER.debug(f"Skipping generation for {mot_folder_path} as they already exist.")
            LOGGER.info(
                "Cache decision | seq={} | action={} | det_exists={} | det_bytes={} | det_rows={} | "
                "emb_exists={} | emb_bytes={} | emb_rows={} | rows_match={}",
                mot_folder_path.name,
                action,
                det_state["exists"],
                det_state["bytes"],
                det_state["rows"],
                emb_state["exists"],
                emb_state["bytes"],
                emb_state["rows"],
                cache_rows_match,
            )
            if action == "reuse":
                continue
            tasks.append((y, mot_folder_path))

        total_sequences = len(mot_folder_paths)
        if len(tasks) == 0:
            LOGGER.info(f"Detections and embeddings cached for all {total_sequences} sequences")
        else:
            LOGGER.info(f"Generating detections and embeddings for {len(tasks)}/{total_sequences} sequences with model {y.name}")

        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_THREADS, initializer=_worker_init) as executor:
            futures = [executor.submit(process_single_det_emb, y, source_path, opt, lock) for y, source_path in tasks]

            for fut in concurrent.futures.as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    LOGGER.error(f"An error occurred during detection/embedding generation: {e}")
                    raise e


def process_sequence(seq_name: str,
                     mot_root: str,
                     project_root: str,
                     model_name: str,
                     reid_name: str,
                     tracking_method: str,
                     exp_folder: str,
                     target_fps: Optional[int],
                     device: str,
                     class_id_offset: int,
                     per_class: bool,
                     cfg_dict: Optional[Dict] = None,
                     ):

    LOGGER.info(
        "Track sequence start | seq={} | mot_root={} | project_root={} | detector={} | reid={} | "
        "tracker={} | target_fps={} | class_id_offset={} | per_class={}",
        seq_name,
        mot_root,
        project_root,
        model_name,
        reid_name,
        tracking_method,
        target_fps,
        class_id_offset,
        per_class,
    )
    device = select_device(device)
    tracker = create_tracker(
        tracker_type=tracking_method,
        tracker_config=TRACKER_CONFIGS / (tracking_method + ".yaml"),
        reid_weights=Path(reid_name + '.pt'),
        device=device,
        half=False,
        per_class=per_class,
        evolve_param_dict=cfg_dict,
    )

    # load with the user’s FPS
    dataset = MOT17DetEmbDataset(
        mot_root=mot_root,
        det_emb_root=str(Path(project_root) / 'dets_n_embs'),
        model_name=model_name,
        reid_name=reid_name,
        target_fps=target_fps
    )
    sequence = dataset.get_sequence(seq_name)

    all_tracks = []
    kept_frame_ids = []
    frames_seen = 0
    frames_with_dets = 0
    frames_with_tracks = 0
    det_rows_seen = 0
    track_rows_written = 0
    for frame in sequence:
        frames_seen += 1
        fid  = int(frame['frame_id'])
        dets = frame['dets']
        embs = frame['embs']
        img  = frame['img']

        kept_frame_ids.append(fid)
        det_rows_seen += int(dets.shape[0]) if dets.ndim > 1 else int(dets.size > 0)

        if dets.size and embs.size:
            frames_with_dets += 1
            tracks = tracker.update(dets, img, embs)
            if tracks.size:
                mot_tracks = convert_to_mot_format(tracks, fid, class_id_offset=class_id_offset)
                all_tracks.append(mot_tracks)
                frames_with_tracks += 1
                track_rows_written += int(mot_tracks.shape[0])

    out_arr = np.vstack(all_tracks) if all_tracks else np.empty((0, 0))
    out_path = Path(exp_folder) / f"{seq_name}.txt"
    write_mot_results(out_path, out_arr)
    out_state = _cache_file_state(out_path)
    if out_arr.size:
        class_ids, class_counts = np.unique(out_arr[:, 7].astype(int), return_counts=True)
        output_class_counts = {
            int(cls_id): int(count)
            for cls_id, count in sorted(zip(class_ids, class_counts))
        }
    else:
        output_class_counts = {}
    LOGGER.info(
        "Track sequence done | seq={} | frames_seen={} | kept_frames={} | frames_with_dets={} | "
        "frames_with_tracks={} | det_rows_seen={} | track_rows_written={} | output={} | output_exists={} | "
        "output_rows={} | output_bytes={} | output_class_counts={}",
        seq_name,
        frames_seen,
        len(kept_frame_ids),
        frames_with_dets,
        frames_with_tracks,
        det_rows_seen,
        track_rows_written,
        out_path,
        out_state["exists"],
        out_state["rows"],
        out_state["bytes"],
        output_class_counts,
    )
    return seq_name, kept_frame_ids, str(out_path), track_rows_written


from boxmot.utils import configure_logging as _configure_logging

def _worker_init():
    # each spawned process needs its own sinks
    _configure_logging()

def run_generate_mot_results(opt: argparse.Namespace, evolve_config: dict = None) -> None:
    # Prepare experiment folder
    base = opt.project / 'mot' / f"{opt.yolo_model[0].stem}_{opt.reid_model[0].stem}_{opt.tracking_method}"
    exp_dir = increment_path(base, sep="_", exist_ok=False)
    exp_dir.mkdir(parents=True, exist_ok=True)
    opt.exp_dir = exp_dir
    class_id_offset = resolve_tracker_class_id_offset(opt)
    per_class = bool(getattr(opt, "per_class", False))

    # Just collect sequence names by scanning directory names
    sequence_names = []
    for d in Path(opt.source).iterdir():
        if not d.is_dir():
            continue
        img_dir = d / "img1" if (d / "img1").exists() else d
        if any(img_dir.glob("*.jpg")) or any(img_dir.glob("*.png")):
            sequence_names.append(d.name)
    sequence_names.sort()
    LOGGER.info(
        "Track generation start | sequences={} | exp_dir={} | source={} | project={} | "
        "class_id_offset={} | per_class={}",
        sequence_names,
        exp_dir,
        opt.source,
        opt.project,
        class_id_offset,
        per_class,
    )

    # Build task arguments
    task_args = [
        (
            seq,
            str(opt.source),
            str(opt.project),
            opt.yolo_model[0].stem,
            opt.reid_model[0].stem,
            opt.tracking_method,
            str(exp_dir),
            getattr(opt, 'fps', None),
            opt.device,
            class_id_offset,
            per_class,
            evolve_config,
        )
        for seq in sequence_names
    ]

    seq_frame_nums = {}
    failed_sequences = {}
    successful_outputs = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_THREADS, initializer=_worker_init) as executor:
        futures = {
            executor.submit(process_sequence, *args): args[0] for args in task_args
        }

        for fut in concurrent.futures.as_completed(futures):
            seq = futures[fut]
            try:
                seq_name, kept_ids, out_path, track_rows = fut.result()
                seq_frame_nums[seq_name] = kept_ids
                successful_outputs[seq_name] = {
                    "path": out_path,
                    "frames": len(kept_ids),
                    "track_rows": track_rows,
                }
            except Exception:
                failed_sequences[seq] = traceback.format_exc()
                LOGGER.exception(f"Error processing {seq}")

    expected_outputs = {seq: exp_dir / f"{seq}.txt" for seq in sequence_names}
    missing_outputs = [seq for seq, path in expected_outputs.items() if not path.exists()]
    LOGGER.info(
        "Track generation summary | expected={} | succeeded={} | failed={} | missing_outputs={} | outputs={}",
        len(sequence_names),
        sorted(successful_outputs.keys()),
        sorted(failed_sequences.keys()),
        missing_outputs,
        successful_outputs,
    )
    if failed_sequences or missing_outputs:
        failure_lines = []
        for seq in sorted(failed_sequences):
            failure_lines.append(f"{seq}: {failed_sequences[seq].splitlines()[-1]}")
        if missing_outputs:
            failure_lines.append(f"missing_outputs: {missing_outputs}")
        raise RuntimeError("MOT result generation failed; " + " | ".join(failure_lines))

    # Optional GSI postprocessing
    if getattr(opt, "postprocessing", "none") == "gsi":
        LOGGER.opt(colors=True).info("<cyan>[3b/4]</cyan> Applying GSI postprocessing...")
        from boxmot.postprocessing.gsi import gsi
        gsi(mot_results_folder=exp_dir)

    elif getattr(opt, "postprocessing", "none") == "gbrc":
        LOGGER.opt(colors=True).info("<cyan>[3b/4]</cyan> Applying GBRC postprocessing...")
        from boxmot.postprocessing.gbrc import gbrc
        gbrc(mot_results_folder=exp_dir)


def run_trackeval(opt: argparse.Namespace, verbose: bool = True) -> dict:
    """
    Runs the trackeval function to evaluate tracking results.

    Args:
        opt (Namespace): Parsed command line arguments.
        verbose (bool): Whether to print results summary. Default True.
    """
    seq_paths, seq_info = _collect_seq_info(opt.source)
    runtime_cfg = load_runtime_dataset_cfg(opt)
    bench_cfg = runtime_cfg.get("benchmark", {}) if isinstance(runtime_cfg, dict) else {}
    gt_folder_name = bench_cfg.get("gt_folder", "annotations") if isinstance(bench_cfg, dict) else "annotations"
    configured_gt_folder = opt.source.parent / gt_folder_name
    gt_folder = configured_gt_folder if configured_gt_folder.exists() else opt.source
    dataset_settings = build_dataset_eval_settings(opt, gt_folder, seq_info)
    gt_loc_format = dataset_settings["gt_loc_format"]

    if not seq_paths:
        raise ValueError(f"No sequences with images found under {opt.source}")

    if gt_folder.exists():
        for seq_name in list(seq_info.keys()):
            ann_file = format_gt_path(gt_folder, gt_loc_format, seq_name)
            if not ann_file.exists():
                continue
            try:
                with open(ann_file, "r") as f:
                    max_frame = 0
                    for line in f:
                        if not line.strip():
                            continue
                        frame_id = int(float(line.split(",", 1)[0]))
                        if frame_id > max_frame:
                            max_frame = frame_id
                    if max_frame:
                        # 如果图片数量和标注文件中的最大帧数不一致，以标注文件中的最大帧数为准
                        seq_info[seq_name] = max(seq_info.get(seq_name, 0) or 0, max_frame)
            except Exception:
                LOGGER.warning(f"Failed to read annotation file {ann_file} for sequence length inference")
    save_dir = Path(opt.project) / opt.name

    tracker_precheck = {}
    for seq_name in sorted(seq_info.keys()):
        tracker_path = Path(opt.exp_dir) / f"{seq_name}.txt"
        gt_path = format_gt_path(gt_folder, gt_loc_format, seq_name)
        tracker_precheck[seq_name] = {
            "tracker_exists": tracker_path.exists(),
            "tracker_rows": _count_non_comment_rows(tracker_path) if tracker_path.exists() else None,
            "tracker_bytes": tracker_path.stat().st_size if tracker_path.exists() else None,
            "tracker_class_counts": _mot_class_counts(tracker_path),
            "gt_exists": gt_path.exists(),
            "gt_rows": _count_non_comment_rows(gt_path) if gt_path.exists() else None,
            "gt_class_counts": _mot_class_counts(gt_path),
            "seq_len": seq_info[seq_name],
        }
    missing_tracker_files = [
        seq for seq, state in tracker_precheck.items()
        if not state["tracker_exists"]
    ]
    LOGGER.info(
        "TrackEval precheck | exp_dir={} | gt_folder={} | benchmark={} | split={} | missing_tracker_files={} | files={}",
        opt.exp_dir,
        gt_folder,
        getattr(opt, "benchmark", None),
        getattr(opt, "split", None),
        missing_tracker_files,
        tracker_precheck,
    )
    validate_tracker_class_id_alignment(
        tracker_precheck,
        resolve_tracker_class_id_offset(opt),
    )
    
    trackeval_results = trackeval(opt, seq_paths, save_dir, gt_folder, seq_info=seq_info)
    parsed_results = parse_mot_results(trackeval_results)

    # Load config to filter classes
    cfg_name = resolve_dataset_cfg_name(opt)
    try:
        cfg = load_runtime_dataset_cfg(opt) if cfg_name else {}
    except FileNotFoundError:
        cfg = {}
    if not cfg:
        missing_cfg_name = cfg_name or getattr(opt, 'benchmark', str(opt.source.parent.name))
        LOGGER.warning(f"Could not find dataset config for {missing_cfg_name}. Class filtering might be incorrect.")

    # Filter parsed_results based on user provided classes (opt.classes)
    single_class_mode = False
    selected_classes = None

    # Priority 1: Benchmark config classes (overrides user classes)
    # 第一优先级找 yaml 文件中的 benchmark.eval_classes字段
    if "benchmark" in cfg:
        bench_cfg = cfg["benchmark"]
        bench_classes = None

        if isinstance(bench_cfg, dict):
            if "eval_classes" in bench_cfg:
                bench_classes = [v for _, v in sorted(bench_cfg["eval_classes"].items(), key=lambda kv: int(kv[0]))]
            elif "classes" in bench_cfg:
                bench_classes = bench_cfg["classes"].split()

        if bench_classes:
            selected_classes = bench_classes
            parsed_results = build_mot_results_summary(parsed_results, bench_classes)
            if len(bench_classes) == 1:
                single_class_mode = True
    # Priority 2: User provided classes
    # 第二优先级使用命令行 --classes 指定
    elif hasattr(opt, 'classes') and opt.classes is not None:
        class_indices = opt.classes if isinstance(opt.classes, list) else [opt.classes]
        user_classes = [COCO_CLASSES[int(i)] for i in class_indices]
        selected_classes = user_classes
        parsed_results = build_mot_results_summary(parsed_results, user_classes)
        if len(user_classes) == 1:
            single_class_mode = True

    if selected_classes and len(selected_classes) > 1 and OVERALL_RESULT_KEY not in parsed_results:
        LOGGER.warning(
            "TrackEval 未返回 {}，无法生成跨类别 overall；请确认 dataset.should_classes_combine=True",
            TRACKEVAL_DETECTION_AVERAGE_KEY,
        )

    overall_metrics = parsed_results.get(OVERALL_RESULT_KEY)
    if overall_metrics is not None:
        overall_output = Path(opt.exp_dir) / "overall_summary.json"
        overall_output.write_text(
            json.dumps(
                {
                    "benchmark": dataset_settings["benchmark_name"],
                    "aggregation": TRACKEVAL_DETECTION_AVERAGE_KEY,
                    "classes": selected_classes,
                    "class_ids": dataset_settings["class_ids"],
                    "tracker_class_id_offset": resolve_tracker_class_id_offset(opt),
                    "ignore_region_class_ids": dataset_settings["ignore_region_ids"],
                    "ignore_ioa_threshold": dataset_settings["ignore_ioa_threshold"],
                    "ignore_removal_condition": "coverage >= threshold",
                    "ignore_filter_scope": "gt_and_tracker_before_class_matching",
                    "metrics": overall_metrics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        LOGGER.info("Overall summary saved to {}", overall_output)

    if single_class_mode and len(parsed_results) > 0:
        final_results = list(parsed_results.values())[0]
    else:
        final_results = parsed_results
    
    # Print results summary
    if verbose:
        LOGGER.info("")
        LOGGER.opt(colors=True).info("<blue>" + "="*90 + "</blue>")
        LOGGER.opt(colors=True).info("<bold><cyan>📊 Results Summary</cyan></bold>")
        LOGGER.opt(colors=True).info("<blue>" + "="*90 + "</blue>")
        
        headers = ["Class", "HOTA", "DetA", "AssA", "MOTA", "IDF1", "AssRe", "IDSW", "IDs"]
        header_str = "{:<15} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format(*headers)
        LOGGER.opt(colors=True).info(f"<bold>{header_str}</bold>")
        LOGGER.opt(colors=True).info("<blue>" + "-"*90 + "</blue>")
        
        for cls, metrics in parsed_results.items():
            row = [
                cls,
                f"{metrics.get('HOTA', 0):.2f}%",
                f"{metrics.get('DetA', 0):.2f}%",
                f"{metrics.get('AssA', 0):.2f}%",
                f"{metrics.get('MOTA', 0):.2f}%",
                f"{metrics.get('IDF1', 0):.2f}%",
                f"{metrics.get('AssRe', 0):.2f}%",
                f"{metrics.get('IDSW', 0)}",
                f"{metrics.get('IDs', 0)}"
            ]
            row_str = "{:<15} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format(*row)
            LOGGER.opt(colors=True).info(row_str)
            
        LOGGER.opt(colors=True).info("<blue>" + "="*90 + "</blue>")

    if opt.ci:
        with open(opt.tracking_method + "_output.json", "w") as outfile:
            outfile.write(json.dumps(final_results))
    
    return final_results


def main(args):
    # Print evaluation pipeline header (blue palette)
    LOGGER.info("")
    LOGGER.opt(colors=True).info("<blue>" + "="*60 + "</blue>")
    LOGGER.opt(colors=True).info("<bold><cyan>🚀 BoxMOT Evaluation Pipeline</cyan></bold>")
    LOGGER.opt(colors=True).info("<blue>" + "="*60 + "</blue>")
    LOGGER.opt(colors=True).info(f"<bold>Detector:</bold>  <cyan>{args.yolo_model[0]}</cyan>")
    LOGGER.opt(colors=True).info(f"<bold>ReID:</bold>      <cyan>{args.reid_model[0]}</cyan>")
    LOGGER.opt(colors=True).info(f"<bold>Tracker:</bold>   <cyan>{args.tracking_method}</cyan>")
    LOGGER.opt(colors=True).info(f"<bold>Benchmark:</bold> <cyan>{args.source}</cyan>")
    LOGGER.opt(colors=True).info("<blue>" + "="*60 + "</blue>")
    
    # Step 1: Download TrackEval
    LOGGER.opt(colors=True).info("<cyan>[1/4]</cyan> Setting up TrackEval...")
    eval_init(args)

    # Step 2: Generate detections and embeddings
    LOGGER.opt(colors=True).info("<cyan>[2/4]</cyan> Generating detections and embeddings...")
    run_generate_dets_embs(args)
    
    # Step 3: Generate MOT results
    LOGGER.opt(colors=True).info("<cyan>[3/4]</cyan> Running tracker...")
    run_generate_mot_results(args)
    
    # Step 4: Evaluate with TrackEval
    LOGGER.opt(colors=True).info("<cyan>[4/4]</cyan> Evaluating results...")
    results = run_trackeval(args)
    
    # Only plot if we have results for a single class or handle multi-class plotting differently
    # For now, let's just skip the radar chart if we have multiple classes or complex structure
    # Or pick the first class found?
    # The original code expected a flat dict of metrics. Now we have {class: {metric: val}}
    
    # Let's try to plot for each class or just skip for now to avoid breaking
    # If 'pedestrian' is in results, use that, otherwise use the first key
    
    # Check if results is flat (single class backward compatibility) or nested
    is_flat = False
    if results and isinstance(list(results.values())[0], (int, float)):
        is_flat = True
        metrics_data = results
        plot_class = 'single_class'
    else:
        plot_class = OVERALL_RESULT_KEY if OVERALL_RESULT_KEY in results else 'pedestrian'
        if plot_class not in results and len(results) > 0:
            plot_class = list(results.keys())[0]
        metrics_data = results.get(plot_class, {})

    if metrics_data:
        plotter = MetricsPlotter(args.exp_dir)
        
        # Filter only the metrics we want to plot
        plot_metrics = ['HOTA', 'MOTA', 'IDF1']
        plot_values = [metrics_data.get(m, 0) for m in plot_metrics]

        plotter.plot_radar_chart(
            {args.tracking_method: plot_values},
            plot_metrics,
            title=f"MOT metrics radar Chart ({plot_class})",
            ylim=(0, 100),
            yticks=[20, 40, 60, 80, 100],
            ytick_labels=['20', '40', '60', '80', '100']
        )
        

if __name__ == "__main__":
    main()
