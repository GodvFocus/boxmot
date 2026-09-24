"""将缓存的检测结果绘制到原始图像序列。"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from boxmot.utils import logger as LOGGER


VISDRONE_CLASS_NAMES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}

# 固定类别颜色，保证跨帧观察同一类别时颜色不会跳变。
CLASS_COLORS = {
    0: (0, 255, 255),
    1: (255, 255, 0),
    2: (255, 0, 255),
    3: (0, 255, 0),
    4: (255, 128, 0),
    5: (0, 128, 255),
    6: (128, 255, 0),
    7: (255, 0, 128),
    8: (0, 0, 255),
    9: (128, 0, 255),
}


def parse_args() -> argparse.Namespace:
    """解析检测可视化参数。"""
    parser = argparse.ArgumentParser(description="将 BoxMOT 检测缓存绘制到 VisDrone 原图")
    parser.add_argument("--dets-dir", type=Path, required=True, help="检测结果 txt 所在目录")
    parser.add_argument("--dataset-root", type=Path, required=True, help="VisDrone 数据集根目录")
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="序列名称；不指定时使用同时存在图片和检测结果的第一个序列",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="输出目录")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="最低检测置信度")
    parser.add_argument("--start-frame", type=int, default=None, help="起始帧号，默认从序列首帧开始")
    parser.add_argument("--max-frames", type=int, default=50, help="最多生成的帧数")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="输出 JPG 质量，范围为 1-100")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有可视化图片")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """尽早拒绝无效参数，避免处理到中途才失败。"""
    if not args.dets_dir.is_dir():
        raise FileNotFoundError(f"检测结果目录不存在：{args.dets_dir}")
    if not (args.dataset_root / "sequences").is_dir():
        raise FileNotFoundError(f"图片序列目录不存在：{args.dataset_root / 'sequences'}")
    if not 0.0 <= args.conf_thres <= 1.0:
        raise ValueError("--conf-thres 必须位于 [0, 1]")
    if args.start_frame is not None and args.start_frame < 1:
        raise ValueError("--start-frame 必须大于等于 1")
    if args.max_frames < 1:
        raise ValueError("--max-frames 必须大于等于 1")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality 必须位于 [1, 100]")


def resolve_sequence(args: argparse.Namespace) -> str:
    """确定要处理的序列，并检查图片与检测文件是否同时存在。"""
    sequences_root = args.dataset_root / "sequences"
    available = sorted(
        sequence_dir.name
        for sequence_dir in sequences_root.iterdir()
        if sequence_dir.is_dir() and (args.dets_dir / f"{sequence_dir.name}.txt").is_file()
    )
    if not available:
        raise FileNotFoundError("没有找到同时包含图片和检测结果的序列")

    sequence = args.sequence or available[0]
    if sequence not in available:
        raise FileNotFoundError(f"序列缺少图片或检测文件：{sequence}")
    return sequence


def load_detections(det_path: Path, conf_thres: float) -> dict[int, np.ndarray]:
    """读取七列检测缓存，并按帧号组织置信度达标的检测框。"""
    detections = np.loadtxt(det_path, comments="#", ndmin=2)
    if detections.size == 0:
        return {}
    if detections.shape[1] != 7:
        raise ValueError(f"检测文件应为 7 列，实际为 {detections.shape[1]} 列：{det_path}")

    detections = detections[detections[:, 5] >= conf_thres]
    by_frame: defaultdict[int, list[np.ndarray]] = defaultdict(list)
    for row in detections:
        by_frame[int(row[0])].append(row)
    return {frame_id: np.stack(rows) for frame_id, rows in by_frame.items()}


def select_images(
    sequence_dir: Path,
    start_frame: int | None,
    max_frames: int,
) -> list[Path]:
    """按文件名中的真实帧号选择图片，避免目录枚举顺序导致错位。"""
    image_paths = sorted(
        [*sequence_dir.glob("*.jpg"), *sequence_dir.glob("*.png")],
        key=lambda path: int(path.stem),
    )
    if start_frame is not None:
        image_paths = [path for path in image_paths if int(path.stem) >= start_frame]
    return image_paths[:max_frames]


def draw_label(
    image: np.ndarray,
    text: str,
    left: int,
    top: int,
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    """绘制带底色的标签，避免文字与复杂背景混在一起。"""
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )
    label_height = text_height + baseline + 4
    label_left = min(left, max(0, image.shape[1] - text_width - 4))
    label_right = min(image.shape[1] - 1, label_left + text_width + 4)
    if top >= label_height:
        label_top = top - label_height
        label_bottom = top
        text_y = top - baseline - 2
    else:
        # 框贴近图像上沿时把标签放到框内侧，避免文字被图像边界裁掉。
        label_top = top
        label_bottom = min(image.shape[0] - 1, top + label_height)
        text_y = min(image.shape[0] - baseline - 1, top + text_height + 2)

    cv2.rectangle(image, (label_left, label_top), (label_right, label_bottom), color, -1)
    cv2.putText(
        image,
        text,
        (label_left + 2, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def draw_detections(image: np.ndarray, detections: np.ndarray) -> np.ndarray:
    """在图像副本上绘制检测框、类别名称和置信度。"""
    rendered = image.copy()
    height, width = rendered.shape[:2]
    box_thickness = max(1, round(min(height, width) / 600))
    font_scale = max(0.4, min(height, width) / 1800)
    text_thickness = max(1, box_thickness - 1)

    # 先绘制低置信度框，高置信度框最后绘制时更容易被看清。
    for detection in detections[np.argsort(detections[:, 5])]:
        x1, y1, x2, y2 = detection[1:5]
        confidence = float(detection[5])
        class_id = int(detection[6])

        left = int(np.clip(round(x1), 0, width - 1))
        top = int(np.clip(round(y1), 0, height - 1))
        right = int(np.clip(round(x2), 0, width - 1))
        bottom = int(np.clip(round(y2), 0, height - 1))
        if right <= left or bottom <= top:
            continue

        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        class_name = VISDRONE_CLASS_NAMES.get(class_id, f"class-{class_id}")
        cv2.rectangle(rendered, (left, top), (right, bottom), color, box_thickness)
        draw_label(
            rendered,
            f"{class_name} {confidence:.2f}",
            left,
            top,
            color,
            font_scale,
            text_thickness,
        )
    return rendered


def visualize(args: argparse.Namespace) -> Path:
    """处理一个序列，并返回最终输出目录。"""
    validate_args(args)
    sequence = resolve_sequence(args)
    sequence_dir = args.dataset_root / "sequences" / sequence
    det_path = args.dets_dir / f"{sequence}.txt"
    detector_name = args.dets_dir.parent.name
    output_dir = args.output_dir or Path("runs") / "detection_vis" / detector_name / sequence

    existing_outputs = list(output_dir.glob("*.jpg")) if output_dir.exists() else []
    if existing_outputs and not args.overwrite:
        raise FileExistsError(f"输出目录已有 JPG；如需覆盖请添加 --overwrite：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    detections_by_frame = load_detections(det_path, args.conf_thres)
    image_paths = select_images(sequence_dir, args.start_frame, args.max_frames)
    if not image_paths:
        raise FileNotFoundError(f"指定范围内没有图片：{sequence_dir}")

    total_detections = 0
    for image_path in image_paths:
        frame_id = int(image_path.stem)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图片：{image_path}")

        frame_detections = detections_by_frame.get(frame_id, np.empty((0, 7)))
        rendered = draw_detections(image, frame_detections)
        output_path = output_dir / f"{image_path.stem}.jpg"
        success = cv2.imwrite(
            str(output_path),
            rendered,
            [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
        )
        if not success:
            raise OSError(f"图片写入失败：{output_path}")
        total_detections += len(frame_detections)

    LOGGER.info(
        "检测可视化完成 | sequence={} | frames={} | detections={} | conf_thres={} | output={}",
        sequence,
        len(image_paths),
        total_detections,
        args.conf_thres,
        output_dir.resolve(),
    )
    return output_dir.resolve()


def main() -> None:
    """命令行入口。"""
    visualize(parse_args())


if __name__ == "__main__":
    main()
