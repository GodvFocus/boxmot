import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "boxmot" / "engine" / "trackeval"))

from boxmot.engine.evaluator import (  # noqa: E402
    build_mot_results_summary,
    parse_mot_results,
    validate_tracker_class_id_alignment,
)
from boxmot.engine import tuner as tuner_module  # noqa: E402
from boxmot.utils.custom_mot_challenge_2d_box import (  # noqa: E402
    CustomMotChallenge2DBox,
    _official_ignore_coverages,
)
from trackeval.datasets.mot_challenge_2d_box import MotChallenge2DBox  # noqa: E402


def test_visdrone_config_uses_official_evaluation_semantics():
    """VisDrone 正式评估必须保留原始 GT，并只统计官方五类。"""
    config_path = ROOT / "boxmot" / "configs" / "datasets" / "visdrone-ablation.yaml"
    with config_path.open(encoding="utf-8") as file:
        benchmark = yaml.safe_load(file)["benchmark"]

    assert benchmark["gt_folder"] == "annotations_old"
    assert benchmark["gt_loc_format"] == "{gt_folder}/{seq}.txt"
    assert benchmark["eval_classes"] == {
        1: "pedestrian",
        4: "car",
        5: "van",
        6: "truck",
        9: "bus",
    }
    assert set(benchmark["ignore_region_classes"]) == {0, 11}
    assert benchmark["tracker_class_id_offset"] == 1


def test_custom_dataset_combines_only_multiclass_results(monkeypatch):
    """五类 VisDrone 开启跨类别汇总，单类别评估仍保持原有结构。"""

    def fake_parent_init(dataset, config):
        dataset.config = dict(config)

    monkeypatch.setattr(MotChallenge2DBox, "__init__", fake_parent_init)

    multiclass = CustomMotChallenge2DBox(
        {
            "CLASSES_TO_EVAL": ["pedestrian", "car"],
            "CLASS_IDS": [1, 4],
        }
    )
    single_class = CustomMotChallenge2DBox(
        {
            "CLASSES_TO_EVAL": ["pedestrian"],
            "CLASS_IDS": [1],
        }
    )

    assert multiclass.should_classes_combine is True
    assert single_class.should_classes_combine is False


def test_ignore_region_filters_tracker_before_valid_gt_matching():
    """官方规则在类别匹配前过滤预测，即使该预测原本可以匹配有效 GT。"""
    dataset = CustomMotChallenge2DBox.__new__(CustomMotChallenge2DBox)
    dataset.class_name_to_class_id = {"pedestrian": 1}
    dataset.distractor_class_ids = []
    dataset.ignore_ioa_threshold = 0.5
    dataset.do_preproc = True
    dataset.benchmark = "VisDrone2019-MOT-test-dev"

    raw_data = {
        "num_timesteps": 1,
        "seq": "fixture",
        "gt_ids": [np.array([10])],
        "gt_dets": [np.array([[0.0, 0.0, 10.0, 10.0]])],
        "gt_classes": [np.array([1])],
        "gt_extras": [{"zero_marked": np.array([1])}],
        "gt_crowd_ignore_regions": [np.array([[0.0, 0.0, 5.0, 10.0]])],
        "tracker_ids": [np.array([20])],
        "tracker_dets": [
            np.array(
                [
                    [0.0, 0.0, 10.0, 10.0],
                ]
            )
        ],
        "tracker_classes": [np.array([1])],
        "tracker_confidences": [np.array([0.9])],
        "similarity_scores": [np.array([[1.0]])],
    }

    data = dataset.get_preprocessed_seq_data(raw_data, "pedestrian")

    assert data["num_gt_dets"] == 1
    assert data["num_tracker_dets"] == 0
    assert data["tracker_ids"][0].tolist() == []


def test_ignore_region_uses_union_area_and_removes_exactly_half_covered_box():
    """多个 ignore region 的覆盖面积需要先取并集，且官方阈值包含 0.5。"""
    dataset = CustomMotChallenge2DBox.__new__(CustomMotChallenge2DBox)
    dataset.class_name_to_class_id = {"pedestrian": 1}
    dataset.distractor_class_ids = []
    dataset.ignore_ioa_threshold = 0.5
    dataset.do_preproc = True
    dataset.benchmark = "VisDrone2019-MOT-test-dev"

    raw_data = {
        "num_timesteps": 1,
        "seq": "fixture",
        "gt_ids": [np.empty(0, dtype=int)],
        "gt_dets": [np.empty((0, 4))],
        "gt_classes": [np.empty(0, dtype=int)],
        "gt_extras": [{"zero_marked": np.empty(0, dtype=int)}],
        "gt_crowd_ignore_regions": [
            np.array(
                [
                    [0.0, 0.0, 3.0, 10.0],
                    [3.0, 0.0, 3.0, 10.0],
                ]
            )
        ],
        "tracker_ids": [np.array([20])],
        "tracker_dets": [np.array([[0.0, 0.0, 10.0, 10.0]])],
        "tracker_classes": [np.array([1])],
        "tracker_confidences": [np.array([0.9])],
        "similarity_scores": [np.empty((0, 1))],
    }

    data = dataset.get_preprocessed_seq_data(raw_data, "pedestrian")

    assert data["num_tracker_dets"] == 0


def test_ignore_region_clips_overlap_to_image_boundary():
    """官方 mask 只存在于图像内，越界部分不能计入 ignore 覆盖面积。"""
    dataset = CustomMotChallenge2DBox.__new__(CustomMotChallenge2DBox)
    dataset.class_name_to_class_id = {"pedestrian": 1}
    dataset.distractor_class_ids = []
    dataset.ignore_ioa_threshold = 0.5
    dataset.seq_image_sizes = {"fixture": (10, 10)}
    dataset.do_preproc = True
    dataset.benchmark = "VisDrone2019-MOT-test-dev"

    raw_data = {
        "num_timesteps": 1,
        "seq": "fixture",
        "gt_ids": [np.empty(0, dtype=int)],
        "gt_dets": [np.empty((0, 4))],
        "gt_classes": [np.empty(0, dtype=int)],
        "gt_extras": [{"zero_marked": np.empty(0, dtype=int)}],
        "gt_crowd_ignore_regions": [np.array([[8.0, 1.0, 10.0, 9.0]])],
        "tracker_ids": [np.array([20])],
        "tracker_dets": [np.array([[8.0, 1.0, 10.0, 9.0]])],
        "tracker_classes": [np.array([1])],
        "tracker_confidences": [np.array([0.9])],
        "similarity_scores": [np.empty((0, 1))],
    }

    data = dataset.get_preprocessed_seq_data(raw_data, "pedestrian")

    assert data["num_tracker_dets"] == 1


def test_ignore_coverage_matches_official_discrete_mask_reference():
    """矩形并集实现应与 MATLAB mask 和积分图的离散像素语义一致。"""
    detections = np.array(
        [
            [1.0, 1.0, 5.0, 5.0],
            [8.0, 1.0, 10.0, 9.0],
            [3.0, 3.0, 4.0, 4.0],
        ]
    )
    ignore_regions = np.array(
        [
            [1.0, 1.0, 3.0, 4.0],
            [3.0, 2.0, 4.0, 5.0],
            [9.0, 1.0, 5.0, 9.0],
        ]
    )
    image_size = (10, 10)

    ignore_mask = np.zeros(image_size, dtype=float)
    rounded_regions = np.maximum(1, np.floor(ignore_regions + 0.5)).astype(int)
    for x, y, width, height in rounded_regions:
        ignore_mask[y - 1:min(image_size[0], y + height), x - 1:min(image_size[1], x + width)] = 1

    expected = []
    rounded_detections = np.maximum(1, np.floor(detections + 0.5)).astype(int)
    for x, y, width, height in rounded_detections:
        x = min(image_size[1], x)
        y = min(image_size[0], y)
        ignored_pixels = ignore_mask[
            y:min(image_size[0], y + height),
            x:min(image_size[1], x + width),
        ].sum()
        expected.append(ignored_pixels / (width * height))

    actual = _official_ignore_coverages(detections, ignore_regions, image_size)

    np.testing.assert_allclose(actual, expected)


def test_ignore_region_filters_ground_truth_with_official_rule(monkeypatch):
    """官方 dropObjects.m 对 GT 和 tracker 使用同一套 ignore 覆盖率规则。"""
    raw_data = {
        "num_timesteps": 1,
        "gt_ids": [np.array([10, 11])],
        "gt_dets": [
            np.array(
                [
                    [0.0, 0.0, 10.0, 10.0],
                    [0.0, 0.0, 5.0, 10.0],
                ]
            )
        ],
        "gt_classes": [np.array([1, 0])],
        "gt_extras": [{"zero_marked": np.array([1, 1])}],
        "gt_crowd_ignore_regions": [np.empty((0, 4))],
    }
    monkeypatch.setattr(
        MotChallenge2DBox,
        "_load_raw_file",
        lambda *_args, **_kwargs: raw_data,
    )
    dataset = CustomMotChallenge2DBox.__new__(CustomMotChallenge2DBox)
    dataset.ignore_class_ids = [0, 11]
    dataset.ignore_ioa_threshold = 0.5

    filtered = dataset._load_raw_file("tracker", "sequence", True)

    assert filtered["gt_ids"][0].tolist() == []
    assert filtered["gt_classes"][0].tolist() == []
    assert filtered["gt_crowd_ignore_regions"][0].tolist() == [[0.0, 0.0, 5.0, 10.0]]


def test_parse_mot_results_keeps_negative_mota():
    """负 MOTA 是有效诊断信息，不能在展示层静默改成零。"""
    output = """
CLEAR: tracker-pedestrianMOTA MOTP MODA CLR_Re CLR_Pr MTR PTR MLR sMOTA CLR_TP CLR_FN CLR_FP IDSW
COMBINED -2.5428 70.623 -1.2987 54.813 49.415 30.94 41.906 27.154 -18.645 58203 47981 59582 1321
"""

    results = parse_mot_results(output)

    assert results["pedestrian"]["MOTA"] == -2.5428
    assert results["pedestrian"]["IDSW"] == 1321


def test_visdrone_overall_uses_detection_averaged_result():
    """论文主结果必须选 pooled-count 口径，而不是五类指标的算术平均。"""
    parsed_results = {
        "pedestrian": {"HOTA": 40.0, "MOTA": 20.0},
        "car": {"HOTA": 60.0, "MOTA": 80.0},
        "cls_comb_cls_av": {"HOTA": 50.0, "MOTA": 50.0},
        "cls_comb_det_av": {"HOTA": 57.0, "MOTA": 71.0},
    }

    results = build_mot_results_summary(parsed_results, ["pedestrian", "car"])

    assert list(results) == ["overall", "pedestrian", "car"]
    assert results["overall"] == parsed_results["cls_comb_det_av"]
    assert results["overall"] != parsed_results["cls_comb_cls_av"]


def test_single_class_result_does_not_duplicate_overall():
    """单类别评估保持原有扁平化兼容行为，不额外制造重复 overall。"""
    parsed_results = {
        "pedestrian": {"HOTA": 40.0},
        "cls_comb_det_av": {"HOTA": 40.0},
    }

    results = build_mot_results_summary(parsed_results, ["pedestrian"])

    assert results == {"pedestrian": {"HOTA": 40.0}}


def test_tracker_precheck_rejects_stale_zero_based_cache():
    """启用 +1 偏移时，旧缓存中的类别 0 必须在评估前被明确拒绝。"""
    tracker_precheck = {
        "sequence-a": {"tracker_class_counts": {0: 12, 1: 30}},
        "sequence-b": {"tracker_class_counts": {1: 8}},
    }

    with np.testing.assert_raises_regex(ValueError, "旧的 0-based 缓存"):
        validate_tracker_class_id_alignment(tracker_precheck, class_id_offset=1)


def test_tracker_precheck_accepts_current_one_based_cache():
    """当前 VisDrone 输出为 1..10 时不应阻止评估。"""
    tracker_precheck = {
        "sequence-a": {"tracker_class_counts": {1: 12, 10: 30}},
    }

    validate_tracker_class_id_alignment(tracker_precheck, class_id_offset=1)


def test_tuner_prefers_overall_for_multiclass_results(monkeypatch):
    """调参目标必须与论文主结果使用相同的 pooled-count 口径。"""
    monkeypatch.setattr(tuner_module, "run_generate_mot_results", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        tuner_module,
        "run_trackeval",
        lambda _opt: {
            "overall": {"HOTA": 57.0},
            "pedestrian": {"HOTA": 40.0},
            "car": {"HOTA": 60.0},
        },
    )
    tracker = tuner_module.Tracker(SimpleNamespace(objectives=["HOTA"]))

    result = tracker.objective_function({})

    assert result == {"HOTA": 57.0}
