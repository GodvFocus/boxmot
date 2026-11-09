from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from trackeval.datasets.mot_challenge_2d_box import MotChallenge2DBox
import trackeval._timing as _timing

# Default COCO80 mapping; used when dataset configs do not supply explicit ids.
DEFAULT_CLASS_NAME_TO_ID = {
    'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5, 'bus': 6, 'train': 7, 'truck': 8, 'boat': 9, 'traffic light': 10,
    'fire hydrant': 11, 'stop sign': 12, 'parking meter': 13, 'bench': 14, 'bird': 15, 'cat': 16, 'dog': 17, 'horse': 18, 'sheep': 19, 'cow': 20,
    'elephant': 21, 'bear': 22, 'zebra': 23, 'giraffe': 24, 'backpack': 25, 'umbrella': 26, 'handbag': 27, 'tie': 28, 'suitcase': 29, 'frisbee': 30,
    'skis': 31, 'snowboard': 32, 'sports ball': 33, 'kite': 34, 'baseball bat': 35, 'baseball glove': 36, 'skateboard': 37, 'surfboard': 38, 'tennis racket': 39, 'bottle': 40,
    'wine glass': 41, 'cup': 42, 'fork': 43, 'knife': 44, 'spoon': 45, 'bowl': 46, 'banana': 47, 'apple': 48, 'sandwich': 49, 'orange': 50,
    'broccoli': 51, 'carrot': 52, 'hot dog': 53, 'pizza': 54, 'donut': 55, 'cake': 56, 'chair': 57, 'couch': 58, 'potted plant': 59, 'bed': 60,
    'dining table': 61, 'toilet': 62, 'tv': 63, 'laptop': 64, 'mouse': 65, 'remote': 66, 'keyboard': 67, 'cell phone': 68, 'microwave': 69, 'oven': 70,
    'toaster': 71, 'sink': 72, 'refrigerator': 73, 'book': 74, 'clock': 75, 'vase': 76, 'scissors': 77, 'teddy bear': 78, 'hair drier': 79, 'toothbrush': 80
}


def _rectangle_union_area(rectangles: np.ndarray) -> float:
    """计算轴对齐矩形并集面积，避免重叠 ignore region 被重复计数。"""
    if rectangles.size == 0:
        return 0.0

    x_edges = np.unique(rectangles[:, [0, 2]])
    area = 0.0
    for x_left, x_right in zip(x_edges[:-1], x_edges[1:]):
        if x_right <= x_left:
            continue

        active = rectangles[(rectangles[:, 0] < x_right) & (rectangles[:, 2] > x_left)]
        if active.size == 0:
            continue

        intervals = active[:, [1, 3]][np.argsort(active[:, 1])]
        covered_y = 0.0
        current_start, current_end = intervals[0]
        for start, end in intervals[1:]:
            if start > current_end:
                covered_y += current_end - current_start
                current_start, current_end = start, end
            else:
                current_end = max(current_end, end)
        covered_y += current_end - current_start
        area += (x_right - x_left) * covered_y

    return float(area)


def _official_ignore_coverages(
    detections: np.ndarray,
    ignore_regions: np.ndarray,
    image_size: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """按 VisDrone 官方 ignore mask 语义计算每个预测框的覆盖比例。

    官方 MATLAB 实现先对坐标取整，再用所有 ignore region 的并集面积除以
    当前框面积。这里直接计算矩形并集，避免创建整幅图像大小的临时 mask。
    """
    if detections.size == 0:
        return np.empty(0, dtype=float)
    if ignore_regions.size == 0:
        return np.zeros(len(detections), dtype=float)

    # VisDrone 标注坐标非负；floor(x + 0.5) 与 MATLAB 对非负数的 round 一致。
    rounded_detections = np.maximum(1.0, np.floor(np.asarray(detections, dtype=float) + 0.5))
    rounded_regions = np.maximum(1.0, np.floor(np.asarray(ignore_regions, dtype=float) + 0.5))
    # MATLAB 的 ignore mask 会包含 region 的两个端点；转换成半开区间时右端需 +1。
    region_x2 = rounded_regions[:, 0] + rounded_regions[:, 2]
    region_y2 = rounded_regions[:, 1] + rounded_regions[:, 3]
    if image_size is not None:
        image_height, image_width = image_size
        region_x2 = np.minimum(float(image_width), region_x2)
        region_y2 = np.minimum(float(image_height), region_y2)
    region_x2 += 1.0
    region_y2 += 1.0
    region_xyxy = np.column_stack(
        (rounded_regions[:, 0], rounded_regions[:, 1], region_x2, region_y2)
    )

    coverages = np.zeros(len(rounded_detections), dtype=float)
    for index, (x, y, width, height) in enumerate(rounded_detections):
        area = width * height
        if area <= 0:
            continue

        if image_size is not None:
            image_height, image_width = image_size
            x = min(float(image_width), x)
            y = min(float(image_height), y)
            x2 = min(float(image_width), x + width)
            y2 = min(float(image_height), y + height)
        else:
            x2 = x + width
            y2 = y + height

        # 官方积分图公式统计 (x, x+w] × (y, y+h]，因此查询左端需要 +1；
        # 右端转成半开区间同样 +1，宽高仍保持原始 w、h。
        x += 1.0
        y += 1.0
        x2 += 1.0
        y2 += 1.0
        intersections = np.column_stack(
            (
                np.maximum(x, region_xyxy[:, 0]),
                np.maximum(y, region_xyxy[:, 1]),
                np.minimum(x2, region_xyxy[:, 2]),
                np.minimum(y2, region_xyxy[:, 3]),
            )
        )
        intersections = intersections[
            (intersections[:, 2] > intersections[:, 0])
            & (intersections[:, 3] > intersections[:, 1])
        ]
        coverages[index] = min(1.0, _rectangle_union_area(intersections) / area)

    return coverages

class CustomMotChallenge2DBox(MotChallenge2DBox):
    """Custom Dataset class for MOT Challenge 2D bounding box tracking with multi-class support"""

    @staticmethod
    def get_default_dataset_config():
        """Default class config values"""
        default_config = MotChallenge2DBox.get_default_dataset_config()
        default_config['CLASSES_TO_EVAL'] = ['person']
        default_config['CLASS_IDS'] = [1]
        default_config['DISTRACTOR_CLASS_IDS'] = []
        default_config['IGNORE_CLASS_IDS'] = []
        default_config['IGNORE_IOA_THRESHOLD'] = 0.5
        default_config['SEQ_IMAGE_SIZES'] = None
        return default_config

    def __init__(self, config=None):
        """Initialise dataset, checking that all required files are present"""
        cfg = {} if config is None else dict(config)

        # Persist the actual evaluation classes in lowercase to keep consistency with TrackEval internals.
        real_classes = [cls.lower() for cls in cfg.get('CLASSES_TO_EVAL', ['person'])]
        class_ids = cfg.get('CLASS_IDS')
        distractor_ids = cfg.get('DISTRACTOR_CLASS_IDS') or []
        ignore_class_ids = cfg.get('IGNORE_CLASS_IDS') or []
        ignore_ioa_threshold = cfg.get('IGNORE_IOA_THRESHOLD', 0.5)
        seq_image_sizes = cfg.get('SEQ_IMAGE_SIZES') or {}

        # Create a temp config with 'pedestrian' to pass super().__init__ validation
        temp_config = cfg.copy()
        temp_config['CLASSES_TO_EVAL'] = ['pedestrian']
        
        # Initialize parent with temp config
        super().__init__(temp_config)
        
        # Restore real classes and ids in self.config
        self.config['CLASSES_TO_EVAL'] = real_classes
        self.config['CLASS_IDS'] = class_ids
        self.config['DISTRACTOR_CLASS_IDS'] = distractor_ids
        self.config['IGNORE_CLASS_IDS'] = ignore_class_ids
        self.config['IGNORE_IOA_THRESHOLD'] = ignore_ioa_threshold
        self.config['SEQ_IMAGE_SIZES'] = seq_image_sizes
        
        # Overwrite class validation and list with real classes
        self.valid_classes = real_classes
        self.class_list = real_classes
        # 只有多类别任务才需要跨类别汇总，单类别保持 TrackEval 原有输出结构。
        self.should_classes_combine = len(real_classes) > 1

        # Build the class-id map from config if provided, otherwise fallback to COCO-80.
        if class_ids is not None and len(class_ids) == len(real_classes):
            self.class_name_to_class_id = {cls: int(cid) for cls, cid in zip(real_classes, class_ids)}
        else:
            self.class_name_to_class_id = DEFAULT_CLASS_NAME_TO_ID

        self.distractor_class_ids = [int(i) for i in distractor_ids] if distractor_ids else None
        self.ignore_class_ids = [int(i) for i in ignore_class_ids]
        self.ignore_ioa_threshold = float(ignore_ioa_threshold)
        self.seq_image_sizes = {
            str(seq): (int(size[0]), int(size[1]))
            for seq, size in seq_image_sizes.items()
        }
        self.valid_class_numbers = list(self.class_name_to_class_id.values())

    def _load_raw_file(self, tracker, seq, is_gt):
        """加载 MOT 数据，并按官方规则清理 ignore region 内的 GT。"""
        raw_data = super()._load_raw_file(tracker, seq, is_gt)
        if not is_gt or not self.ignore_class_ids:
            return raw_data

        for timestep in range(raw_data['num_timesteps']):
            ignore_mask = np.isin(raw_data['gt_classes'][timestep], self.ignore_class_ids)
            ignore_regions = raw_data['gt_dets'][timestep][ignore_mask]
            raw_data['gt_crowd_ignore_regions'][timestep] = ignore_regions

            # 官方 dropObjects.m 会先移除 0/11，再移除 ignore 覆盖率达到 0.5 的有效 GT。
            keep_mask = ~ignore_mask
            candidate_indices = np.flatnonzero(keep_mask)
            if candidate_indices.size and ignore_regions.size:
                coverages = _official_ignore_coverages(
                    raw_data['gt_dets'][timestep][candidate_indices],
                    ignore_regions,
                    getattr(self, 'seq_image_sizes', {}).get(seq),
                )
                keep_mask[candidate_indices[coverages >= self.ignore_ioa_threshold]] = False

            raw_data['gt_ids'][timestep] = raw_data['gt_ids'][timestep][keep_mask]
            raw_data['gt_dets'][timestep] = raw_data['gt_dets'][timestep][keep_mask]
            raw_data['gt_classes'][timestep] = raw_data['gt_classes'][timestep][keep_mask]
            raw_data['gt_extras'][timestep]['zero_marked'] = (
                raw_data['gt_extras'][timestep]['zero_marked'][keep_mask]
            )

        return raw_data

    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
        self._check_unique_ids(raw_data)
        cls_id = self.class_name_to_class_id[cls]
        # MOT distractor set; prefer config override, otherwise fall back to legacy defaults
        #if self.distractor_class_ids is not None:
        distractor_classes = self.distractor_class_ids or []
        # else:
        #     distractor_classes = [12, 8, 6, 7, 2] if self.benchmark == 'MOT20' else [12, 8, 7, 2]

        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets',
                    'tracker_confidences', 'similarity_scores']
        data = {k: [None] * raw_data['num_timesteps'] for k in data_keys}

        unique_gt_ids, unique_tracker_ids = [], []
        num_gt_dets = num_tracker_dets = 0

        for t in range(raw_data['num_timesteps']):
            gt_ids            = raw_data['gt_ids'][t]
            gt_dets           = raw_data['gt_dets'][t]
            gt_classes        = raw_data['gt_classes'][t]
            gt_zero_marked    = raw_data['gt_extras'][t]['zero_marked']

            tracker_ids         = raw_data['tracker_ids'][t]
            tracker_dets        = raw_data['tracker_dets'][t]
            tracker_classes     = raw_data['tracker_classes'][t]
            tracker_confs       = raw_data['tracker_confidences'][t]
            similarity_scores   = raw_data['similarity_scores'][t]  # shape (num_gt, num_trk)

            # --- Keep only trackers of the eval class (columns) ---
            trk_keep_mask = (tracker_classes == cls_id)
            kept_tracker_ids   = tracker_ids[trk_keep_mask]
            kept_tracker_dets  = tracker_dets[trk_keep_mask, :] if tracker_dets.size else tracker_dets
            kept_tracker_confs = tracker_confs[trk_keep_mask]
            sim = similarity_scores
            sim = sim[:, trk_keep_mask]  # always slice columns; may yield (N, 0)

            # 官方 dropObjects.m 在类别匹配之前过滤所有落入 ignore 并集的预测框；
            # 因此这里不能仅处理未匹配预测，否则会得到更宽松但不可复现的分数。
            crowd_ignore_regions = raw_data['gt_crowd_ignore_regions'][t]
            if kept_tracker_ids.shape[0] > 0 and crowd_ignore_regions.shape[0] > 0:
                ignore_coverages = _official_ignore_coverages(
                    kept_tracker_dets,
                    crowd_ignore_regions,
                    getattr(self, 'seq_image_sizes', {}).get(raw_data['seq']),
                )
                tracker_keep_mask = ignore_coverages < self.ignore_ioa_threshold
                kept_tracker_ids = kept_tracker_ids[tracker_keep_mask]
                kept_tracker_dets = kept_tracker_dets[tracker_keep_mask]
                kept_tracker_confs = kept_tracker_confs[tracker_keep_mask]
                sim = sim[:, tracker_keep_mask]

            # --- Remove trackers that match distractor GT (match vs ALL GT, not filtered by class) ---
            if self.do_preproc and self.benchmark != 'MOT15' and gt_ids.shape[0] > 0 and kept_tracker_ids.shape[0] > 0 and sim.size > 0:
                # threshold & Hungarian (same as original)
                matching_scores = sim.copy()
                matching_scores[matching_scores < 0.5 - np.finfo('float').eps] = 0
                match_rows, match_cols = linear_sum_assignment(-matching_scores)
                actually_matched = matching_scores[match_rows, match_cols] > 0 + np.finfo('float').eps
                match_rows = match_rows[actually_matched]
                match_cols = match_cols[actually_matched]

                # mark tracker columns to remove if matched GT is a distractor class
                is_distr = np.isin(gt_classes[match_rows], distractor_classes)
                to_remove_cols = match_cols[is_distr]

                if to_remove_cols.size > 0:
                    kept_tracker_ids   = np.delete(kept_tracker_ids,   to_remove_cols, axis=0)
                    kept_tracker_dets  = np.delete(kept_tracker_dets,  to_remove_cols, axis=0) if kept_tracker_dets.size else kept_tracker_dets
                    kept_tracker_confs = np.delete(kept_tracker_confs, to_remove_cols, axis=0)
                    sim                = np.delete(sim, to_remove_cols, axis=1)

            # --- Now keep ONLY GT of the eval class and not zero-marked (rows) ---
            if self.do_preproc and self.benchmark != 'MOT15':
                gt_keep_mask = (gt_zero_marked != 0) & (gt_classes == cls_id)
            else:
                gt_keep_mask = (gt_zero_marked != 0)

            kept_gt_ids  = gt_ids[gt_keep_mask]
            kept_gt_dets = gt_dets[gt_keep_mask, :] if gt_dets.size else gt_dets
            sim = sim[gt_keep_mask, :]  # always slice rows; ensures sim.shape == (len(kept_gt_ids), len(kept_tracker_ids))

            # --- Write timestep outputs ---
            data['tracker_ids'][t]          = kept_tracker_ids
            data['tracker_dets'][t]         = kept_tracker_dets
            data['tracker_confidences'][t]  = kept_tracker_confs
            data['gt_ids'][t]               = kept_gt_ids
            data['gt_dets'][t]              = kept_gt_dets
            data['similarity_scores'][t]    = sim

            unique_gt_ids      += list(np.unique(kept_gt_ids))
            unique_tracker_ids += list(np.unique(kept_tracker_ids))
            num_tracker_dets   += len(kept_tracker_ids)
            num_gt_dets        += len(kept_gt_ids)

        # --- Relabel to contiguous ids (no gaps) ---
        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['gt_ids'][t]) > 0:
                    data['gt_ids'][t] = gt_map[data['gt_ids'][t]].astype(int)

        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            trk_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            trk_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['tracker_ids'][t]) > 0:
                    data['tracker_ids'][t] = trk_map[data['tracker_ids'][t]].astype(int)

        # --- Stats & checks ---
        data['num_tracker_dets'] = num_tracker_dets
        data['num_gt_dets']      = num_gt_dets
        data['num_tracker_ids']  = len(np.unique(unique_tracker_ids)) if len(unique_tracker_ids) > 0 else 0
        data['num_gt_ids']       = len(np.unique(unique_gt_ids)) if len(unique_gt_ids) > 0 else 0
        data['num_timesteps']    = raw_data['num_timesteps']
        data['seq']              = raw_data['seq']

        self._check_unique_ids(data, after_preproc=True)
        return data
