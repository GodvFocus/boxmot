# BoxMOT 项目使用指南

> 版本：16.0.9 | 许可：AGPL-3.0 | Python：>=3.9, <3.13
>
> 本指南基于当前代码仓库实际结构编写，如有与官方文档不一致之处，以本指南为准。

---

## 一、代码仓库介绍

### 1.1 顶层目录结构

```
boxmot/
├── boxmot/            # 核心 Python 包
├── assets/            # 测试资源（MOT17-mini 数据集样本）
├── tests/             # 单元测试与性能测试
├── examples/          # 使用示例 Notebook
├── docs/              # MkDocs 文档源码
├── videos/            # 示例视频
├── pyproject.toml     # 项目配置与依赖声明
├── Dockerfile         # Docker 容器配置
├── mkdocs.yml         # 文档站点配置
└── CLAUDE.md          # AI 辅助开发指引
```

### 1.2 核心包结构（boxmot/）

| 目录 | 用途 | 说明 |
|------|------|------|
| `trackers/` | 跟踪器实现 | 7 个跟踪算法，均继承 `BaseTracker` |
| `engine/` | 运行引擎 | CLI、跟踪编排、评估、调优、导出 |
| `reid/` | ReID 模型 | 骨干网络 + 推理后端 + 导出器 |
| `detectors/` | 检测器接口 | Ultralytics、YOLOX、RT-DETR |
| `motion/` | 运动模型 | 卡尔曼滤波器 + 相机运动补偿（CMC） |
| `configs/` | 配置文件 | 跟踪器参数 & 数据集定义 |
| `postprocessing/` | 后处理 | GSI（高斯平滑插值）、GBRC（梯度提升重连） |
| `utils/` | 工具集 | IoU 计算、匈牙利匹配、关联函数、可视化等 |

### 1.3 跟踪器（boxmot/trackers/）

| 跟踪器 | 类名 | 需要 ReID | 关键特点 |
|--------|------|-----------|----------|
| ByteTrack | `ByteTrack` | 否 | 双阶段关联（高/低置信度分离），KalmanFilterXYAH |
| OcSort | `OcSort` | 否 | 观测中心 SORT，速度方向一致性，KalmanFilterXYSR |
| BotSort | `BotSort` | 是 | ECC 相机补偿，外观 + 空间接近度融合 |
| StrongSort | `StrongSort` | 是 | EMA 特征更新，余弦度量，ECC 补偿 |
| DeepOcSort | `DeepOcSort` | 是 | OcSort + ReID，自适应权重，嵌入引导 |
| HybridSort | `HybridSort` | 是 | 4 点 TCM，长期 ReID 特征库 |
| BoostTrack | `BoostTrack` | 是 | 自定义关联 + 梯度提升细化分类 |

**分类**：
- **仅运动（无需 ReID）**：ByteTrack、OcSort — 速度快、内存低，适合实时场景
- **运动+外观（需 ReID）**：BotSort、StrongSort、DeepOcSort、HybridSort、BoostTrack — 精度更高

### 1.4 ReID 模型（boxmot/reid/）

**骨干网络（backbones/）**：

| 骨干网络 | 文件 | 说明 |
|----------|------|------|
| OSNet | `osnet.py`, `osnet_ain.py` | 全尺度网络（x0.25, x0.5, x0.75, x1.0） |
| LightMBN | `lmbn/` | 轻量多分支网络 |
| CLIP | `clip/` | 基于 CLIP 的 ReID |
| ResNet | `resnet.py` | ResNet-18/34/50/101 |
| MobileNetV2 | `mobilenetv2.py` | 轻量级骨干 |
| MLFN | `mlfn.py` | 多层级分解网络 |
| HACNN | `hacnn.py` | 和谐注意力 CNN |

**推理后端（backends/）**：

| 后端 | 文件扩展名 | 说明 |
|------|-----------|------|
| PyTorch | `.pt` | 原生 PyTorch 推理 |
| TorchScript | `.torchscript` | TorchScript 格式 |
| ONNX | `.onnx` | ONNX Runtime |
| TensorRT | `.engine` | NVIDIA GPU 加速 |
| OpenVINO | `.xml` / `.bin` | Intel 推理 |
| TFLite | `.tflite` | 移动端部署 |

`ReidAutoBackend` 根据文件扩展名自动选择后端，无需手动指定。

### 1.5 检测器（boxmot/detectors/）

| 检测器 | 支持的模型 | 依赖 |
|--------|-----------|------|
| Ultralytics | YOLOv8/v9/v10, YOLO11/12, SAM, RTDETR | `ultralytics` |
| YOLOX | yolox_n/s/m/l/x | `yolox` |
| RT-DETR | rtdetr_v2_r18vd/r50vd/r101vd | `transformers`, `timm` |

`get_yolo_inferer()` 工厂函数根据模型名称自动匹配检测策略。

### 1.6 运动模型（boxmot/motion/）

**卡尔曼滤波器（kalman_filters/）**：

| 滤波器 | 状态表示 | 使用者 |
|--------|---------|--------|
| KalmanFilterXYAH | 中心x/y, 宽高比, 高度 | ByteTrack |
| KalmanFilterXYSR | 中心x/y, 缩放, 比率 | OcSort, DeepOcSort |
| KalmanFilterXYWH | 中心x/y, 宽, 高 | BotSort |
| KalmanBoxTrackerOBB | 旋转边界框 | OBB 场景 |

**相机运动补偿 CMC（cmc/）**：

| 方法 | 说明 |
|------|------|
| ECC | 增强相关系数 |
| SOF | 简化光流 |
| ORB | ORB 特征匹配 |
| SIFT | SIFT 特征匹配 |

### 1.7 配置系统（boxmot/configs/）

**跟踪器配置**（`trackers/*.yaml`）— 定义每个跟踪器的超参数，包括默认值和调优搜索空间：

| 文件 | 跟踪器 |
|------|--------|
| `botsort.yaml` | BotSort |
| `bytetrack.yaml` | ByteTrack |
| `ocsort.yaml` | OcSort |
| `strongsort.yaml` | StrongSort |
| `deepocsort.yaml` | DeepOcSort |
| `hybridsort.yaml` | HybridSort |
| `boosttrack.yaml` | BoostTrack |
| `imprassoc.yaml` | ImprAssoc |

**数据集配置**（`datasets/*.yaml`）— 定义数据集下载地址、路径、评估类别：

| 文件 | 数据集 |
|------|--------|
| `MOT17-ablation.yaml` | MOT17 行人跟踪 |
| `MOT20-ablation.yaml` | MOT20 行人跟踪 |
| `dancetrack-ablation.yaml` | DanceTrack 舞蹈跟踪 |
| `SportsMOT.yaml` | 体育运动跟踪 |
| `visdrone-ablation.yaml` | 无人机视角跟踪 |
| `custom.yaml` | 自定义数据集模板 |

### 1.8 后处理（boxmot/postprocessing/）

| 方法 | 说明 |
|------|------|
| GSI | 高斯平滑插值 — 对跟踪间隙进行线性插值 + 高斯过程平滑 |
| GBRC | 梯度提升重连 — 线性插值 + 梯度提升回归平滑，支持多进程并行 |

### 1.9 关键设计模式

| 模式 | 位置 | 说明 |
|------|------|------|
| 工厂模式 | `tracker_zoo.py` 的 `create_tracker()` | 按名称实例化跟踪器 |
| 策略模式 | `detectors/__init__.py` 的 `get_yolo_inferer()` | 按模型名选择检测策略 |
| 策略模式 | `utils/iou.py` 的 `AssociationFunction` | 可选 IoU 变体和距离度量 |
| 装饰器模式 | `@setup_decorator`, `@per_class_decorator` | 跟踪器首帧初始化和按类跟踪 |
| 后端抽象 | `ReidAutoBackend` | 多推理后端统一接口 |

---

## 二、训练与调优

### 重要说明

**BoxMOT 本身不包含检测器和 ReID 模型的训练代码。** 它的设计理念是"即插即用"——使用预训练的检测模型和 ReID 模型，专注于跟踪算法本身。

如需训练检测器或 ReID 模型，请使用对应的上游项目：
- **检测器训练**：使用 [Ultralytics](https://github.com/ultralytics/ultralytics) 训练 YOLO 系列，或使用 [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) 训练 YOLOX
- **ReID 模型训练**：使用 [torchreid](https://github.com/KaiyangZhou/deep-person-reid) 或 [fast-reid](https://github.com/facebookresearch/fast-reid) 训练行人重识别模型

BoxMOT 提供的是**跟踪器超参数调优**功能，即通过 Ray Tune + Optuna 自动搜索最优跟踪参数。

### 2.1 跟踪器超参数调优

#### 基本命令

```bash
D:\anaconda3\envs\detr\python.exe -m boxmot.engine.cli tune \
    yolov8n osnet_x0_25_msmt17 deepocsort \
    --source MOT17-ablation
```

#### 位置参数

| 参数 | 示例 | 说明 |
|------|------|------|
| DETECTOR | `yolov8n`, `yolox_x_MOT17_ablation` | 检测模型名称 |
| REID | `osnet_x0_25_msmt17`, `lmbn_n_duke` | ReID 模型名称 |
| TRACKER | `deepocsort`, `botsort`, `strongsort` | 要调优的跟踪算法 |

#### 关键选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--source` | `0` | 数据集路径或数据集名称 |
| `--n-trials` | `4` | 调优试验次数 |
| `--objectives` | `["HOTA", "MOTA", "IDF1"]` | 优化目标指标 |
| `--tracking-method` | `deepocsort` | 跟踪方法（与位置参数等效） |
| `--yolo-model` | `yolov8n.pt` | YOLO 模型权重 |
| `--reid-model` | `osnet_x0_25_msmt17.pt` | ReID 模型权重 |
| `--conf` | `0.01` | 检测置信度阈值 |
| `--device` | `""` | 设备（如 `0` 或 `cpu`） |
| `--half` | `False` | FP16 半精度推理 |
| `--per-class` | `False` | 按类别分别跟踪 |
| `--postprocessing` | `none` | 后处理方法：`none` / `gsi` / `gbrc` |

#### 调优流程

1. **安装调优依赖**：`pip install "ray[tune]==2.49.2" optuna==3.5.0 bayesian-optimization==2.0.4 plotly==5.19.0 pydantic==2.7.2`
2. **加载 YAML 配置**：将跟踪器配置中的参数搜索空间转换为 Optuna 搜索空间
3. **准备评估环境**：下载 TrackEval 工具和数据集（首次自动执行）
4. **生成检测和嵌入**：运行检测器和 ReID 模型，缓存结果（只执行一次）
5. **执行超参数搜索**：Ray Tune 驱动 Optuna 进行多目标优化

#### 搜索空间类型

| 类型 | 说明 | YAML 示例 |
|------|------|-----------|
| `uniform` | 连续均匀分布 | `track_high_thresh: {type: uniform, default: 0.6, range: [0.3, 0.7]}` |
| `randint` | 随机整数 | `min_hits: {type: randint, default: 3, range: [1, 6]}` |
| `qrandint` | 量化随机整数（带步长） | `track_buffer: {type: qrandint, default: 30, range: [10, 61, 10]}` |
| `choice` | 离散选择 | `asso_func: {type: choice, default: iou, options: [iou, giou, diou, ciou, hmiou]}` |
| `grid_search` | 网格搜索 | `max_age: {type: grid_search, default: 30, range: [10, 20, 30, 40, 50, 60]}` |
| `loguniform` | 对数均匀分布 | `Q_xy_scaling: {type: loguniform, default: 0.01, range: [0.01, 1]}` |

#### 调优结果存储

结果保存在 `{project}/ray/{tracker}_tune` 目录下，支持恢复之前的调优运行。

### 2.2 自定义跟踪器配置

如需手动调整跟踪器参数，直接编辑 `boxmot/configs/trackers/` 下对应的 YAML 文件：

```yaml
# 以 deepocsort.yaml 为例
det_thresh:
  type: uniform
  default: 0.5
  range: [0.3, 0.6]

max_age:
  type: qrandint
  default: 30
  range: [10, 61, 10]

asso_func:
  type: choice
  default: iou
  options: [iou, giou, diou, ciou, hmiou]
```

`default` 值即为实际运行时使用的参数。调优时 `range` / `options` 定义搜索空间。

### 2.3 训练检测器（外部工具）

如果需要在自己的数据集上训练检测器：

1. **YOLO 系列**（推荐）：
   ```bash
   # 使用 Ultralytics 训练
   pip install ultralytics
   yolo train model=yolov8n.pt data=your_data.yaml epochs=100
   ```

2. **YOLOX**：
   ```bash
   # 参考 https://github.com/Megvii-BaseDetection/YOLOX
   python tools/train.py -f exps/default/yolox_s.py -d 8 -b 64
   ```

3. **RT-DETR**：
   ```bash
   # 参考 https://github.com/lyuwenyu/RT-DETR
   python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
   ```

训练完成后，将权重文件（`.pt`）直接用于 BoxMOT 即可。

### 2.4 训练 ReID 模型（外部工具）

如果需要在特定域上训练 ReID 模型：

1. **torchreid**（推荐）：
   ```bash
   # 参考 https://github.com/KaiyangZhou/deep-person-reid
   python train.py --config-file configs/person/osnet_x0_25_MSMT17.yml
   ```

2. **fast-reid**：
   ```bash
   # 参考 https://github.com/facebookresearch/fast-reid
   python tools/train_net.py --config-file configs/Market1500/bagtricks_S50.yml
   ```

训练后的权重文件通过 `--reid-model` 参数指定即可使用。

### 2.5 自定义数据集

使用自定义数据集进行评估/调优时，需准备 MOT 格式的数据：

```
your_dataset/
├── train/
│   └── SEQ_NAME/
│       ├── img1/          # 图像帧（000001.jpg, 000002.jpg, ...）
│       │   └── ...
│       ├── det/           # 检测结果（det.txt，可选）
│       │   └── det.txt
│       └── gt/            # 真值标注（gt.txt）
│           └── gt.txt
```

然后在 `boxmot/configs/datasets/custom.yaml` 中配置：

```yaml
download:
  runs_url: ""       # 预计算检测/嵌入的下载地址（可选）
  dataset_url: ""    # 数据集下载地址（可选）

benchmark:
  source: "assets/MOT17-mini"  # 数据集本地路径
  split: "train"               # 使用哪个子集
  eval_classes:                 # 要评估的类别
    1: pedestrian
  distractor_classes:           # 干扰类别（评估时忽略）
    2: person_on_vehicle
    7: static_person
    8: distractor
    12: reflection
```

### 2.6 生成检测和嵌入

在评估或调优前，可以预先缓存检测结果和 ReID 嵌入，避免重复计算：

```bash
D:\anaconda3\envs\detr\python.exe -m boxmot.engine.cli generate \
    yolov8n osnet_x0_25_msmt17 \
    --source MOT17-ablation
```

此步骤会遍历数据集每个序列，运行检测和 ReID，结果缓存到 `boxmot/engine/trackeval/` 目录下。

---

## 三、推理与评估

### 3.1 跟踪推理（track）

#### 基本用法

```bash
# 使用默认参数（YOLOv8n + OSNet + DeepOcSort）
D:\anaconda3\envs\detr\python.exe -m boxmot.engine.cli track --source video.mp4

# 指定检测器、ReID 模型和跟踪器
D:\anaconda3\envs\detr\python.exe -m boxmot.engine.cli track \
    yolov8n osnet_x0_25_msmt17 botsort \
    --source video.mp4

# 使用摄像头
D:\anaconda3\envs\detr\python.exe -m boxmot.engine.cli track --source 0

# 使用 YOLOX 检测器
D:\anaconda3\envs\detr\python.exe -m boxmot.engine.cli track \
    yolox_s osnet_x0_25_msmt17 deepocsort \
    --source video.mp4

# 使用 RT-DETR 检测器
D:\anaconda3\envs\detr\python.exe -m boxmot.engine.cli track \
    rtdetr_v2_r50vd osnet_x0_25_msmt17 botsort \
    --source video.mp4
```

#### 仅运动跟踪（无需 ReID，速度更快）

```bash
# ByteTrack — 无需 ReID 模型
D:\anaconda3\envs\detr\python.exe -m boxmot.engine.cli track \
    yolov8n bytetrack \
    --source video.mp4

# OcSort — 无需 ReID 模型
D:\anaconda3\envs\detr\python.exe -m boxmot.engine.cli track \
    yolov8n ocsort \
    --source video.mp4
```

#### 常用推理选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--source` | `0` | 输入源：视频文件/目录/URL/glob/0(摄像头) |
| `--yolo-model` | `yolov8n.pt` | 检测模型权重 |
| `--reid-model` | `osnet_x0_25_msmt17.pt` | ReID 模型权重 |
| `--tracking-method` | `deepocsort` | 跟踪算法 |
| `--imgsz` | `640` | 输入图像尺寸 |
| `--conf` | `0.01` | 检测置信度阈值 |
| `--iou` | `0.7` | NMS IoU 阈值 |
| `--device` | `""` | 推理设备（`0`/`0,1,2,3`/`cpu`） |
| `--half` | `False` | FP16 半精度推理 |
| `--vid-stride` | `1` | 视频帧间隔 |
| `--per-class` | `False` | 按类别分别跟踪 |
| `--classes` | `None` | 过滤特定类别（如 `0` 或 `0,1`） |
| `--agnostic-nms` | `False` | 类别无关 NMS |
| `--postprocessing` | `none` | 后处理：`none`/`gsi`/`gbrc` |
| `--fps` | `30` | 输出视频帧率 |

#### 输出与可视化选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--save` | `False` | 保存标注后的视频 |
| `--save-txt` | `False` | 保存 MOT 格式的 txt 结果 |
| `--save-crop` | `False` | 保存裁剪的检测区域 |
| `--show` | `False` | 实时显示跟踪画面 |
| `--show-labels` | `True` | 显示检测标签 |
| `--show-conf` | `True` | 显示置信度 |
| `--show-trajectories` | `False` | 叠加轨迹线 |
| `--show-lost` | `False` | 显示丢失的跟踪 |
| `--target-id` | `None` | 高亮指定 ID（绿色） |
| `--line-width` | auto | 边框线宽 |
| `--project` | `ROOT/runs` | 输出目录 |
| `--name` | `""` | 输出子目录名 |
| `--exist-ok` | `True` | 允许覆盖已有输出 |

#### 输出格式

当使用 `--save-txt` 时，每行格式为 MOT 标准：

```
<frame>, <id>, <x>, <y>, <w>, <h>, <conf>, <class>, -1
```

### 3.2 Python API 推理

除了命令行，还可以通过 Python API 直接调用：

```python
from boxmot import create_tracker, get_tracker_config

# 创建跟踪器
tracker_config = get_tracker_config("botsort")
tracker = create_tracker(
    tracker_type="botsort",
    tracker_config=tracker_config,
    reid_weights="osnet_x0_25_msmt17.pt",
    device="cpu",
    half=False,
    per_class=False,
)

# 逐帧更新
import cv2
cap = cv2.VideoCapture("video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 假设已有检测结果 dets: numpy array [N, 5] (x1, y1, x2, y2, conf)
    # 假设已有类别信息 info: numpy array [N] 或 [N, 1]
    dets = ...   # 从检测器获取
    info = ...   # 类别信息

    tracks = tracker.update(dets, info, frame)
    # tracks: numpy array [M, 8] (x1, y1, x2, y2, id, conf, class, det_ind)

cap.release()
```

也可以使用便捷的 `track()` 函数一步完成检测+跟踪：

```python
from boxmot import track

results = track(
    source="video.mp4",
    yolo_model="yolov8n.pt",
    reid_model="osnet_x0_25_msmt17.pt",
    tracking_method="botsort",
    save=True,
)
```

### 3.3 评估（eval）

#### 基本用法

```bash
# 在 MOT17-ablation 上评估
D:\anaconda3\envs\detr\python.exe -m boxmot.engine.cli eval \
    yolov8n osnet_x0_25_msmt17 deepocsort \
    --source MOT17-ablation

# 在自定义数据集上评估
D:\anaconda3\envs\detr\python.exe -m boxmot.engine.cli eval \
    yolov8n osnet_x0_25_msmt17 botsort \
    --source custom
```

#### 评估流程（4 步）

1. **初始化**（`eval_init`）：下载 TrackEval 工具和数据集（首次自动执行）
2. **生成检测和嵌入**（`generate_dets_embs`）：运行检测器和 ReID，结果缓存
3. **运行跟踪器**（`run_generate_mot_results`）：用缓存结果运行跟踪，生成 MOT 格式输出
4. **计算指标**（`run_trackeval`）：使用 TrackEval 计算 MOT 指标

#### 评估指标

| 指标 | 全称 | 说明 |
|------|------|------|
| HOTA | Higher Order Tracking Accuracy | 综合检测和关联精度 |
| MOTA | Multiple Object Tracking Accuracy | 检测精度为主 |
| IDF1 | ID F1 Score | 关联精度为主 |
| AssA | Association Accuracy | 关联准确率 |
| AssRe | Association Recall | 关联召回率 |
| IDSW | ID Switches | 身份切换次数 |
| IDs | Number of IDs | 唯一身份数量 |

#### 后处理增强

评估时可启用后处理以提升指标：

```bash
# 使用高斯平滑插值
D:\anaconda3\envs\detr\python.exe -m boxmot.engine.cli eval \
    yolov8n osnet_x0_25_msmt17 deepocsort \
    --source MOT17-ablation \
    --postprocessing gsi

# 使用梯度提升重连
D:\anaconda3\envs\detr\python.exe -m boxmot.engine.cli eval \
    yolov8n osnet_x0_25_msmt17 deepocsort \
    --source MOT17-ablation \
    --postprocessing gbrc
```

### 3.4 ReID 模型导出（export）

将 PyTorch ReID 模型导出为部署优化格式：

```bash
# 导出为 ONNX
D:\anaconda3\envs\detr\python.exe -m boxmot.engine.cli export \
    --weights osnet_x0_25_msmt17.pt \
    --include onnx

# 同时导出多种格式
D:\anaconda3\envs\detr\python.exe -m boxmot.engine.cli export \
    --weights osnet_x0_25_msmt17.pt \
    --include onnx openvino engine

# FP16 半精度导出（GPU 加速）
D:\anaconda3\envs\detr\python.exe -m boxmot.engine.cli export \
    --weights osnet_x0_25_msmt17.pt \
    --include engine \
    --half \
    --device 0
```

#### 导出选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--weights` | `osnet_x0_25_msmt17.pt` | 模型权重路径 |
| `--include` | `torchscript` | 导出格式：`torchscript`/`onnx`/`openvino`/`engine`/`tflite` |
| `--batch-size` | `1` | 推理批量大小 |
| `--imgsz` | `640` | 输入尺寸 (H,W) |
| `--device` | `cpu` | 设备 |
| `--half` | `False` | FP16 半精度 |
| `--optimize` | `False` | 优化 TorchScript（移动端） |
| `--dynamic` | `False` | 动态轴（ONNX/TensorRT） |
| `--simplify` | `False` | 简化 ONNX 模型 |
| `--opset` | `18` | ONNX opset 版本 |
| `--workspace` | `4` | TensorRT 工作空间大小（GB） |

#### 导出后的模型输入尺寸

| 模型类型 | 输入尺寸 |
|---------|---------|
| vehicleid / veri 系列 | (256, 256) |
| lmbn 系列 | (384, 128) |
| hacnn 系列 | (160, 64) |
| 其他（默认） | (256, 128) |

#### 导出后使用

导出后的模型可直接替换原始 `.pt` 文件使用：

```bash
# 使用 ONNX ReID 模型
D:\anaconda3\envs\detr\python.exe -m boxmot.engine.cli track \
    yolov8n osnet_x0_25_msmt17.onnx deepocsort \
    --source video.mp4

# 使用 TensorRT ReID 模型
D:\anaconda3\envs\detr\python.exe -m boxmot.engine.cli track \
    yolov8n osnet_x0_25_msmt17.engine deepocsort \
    --source video.mp4
```

`ReidAutoBackend` 会根据文件扩展名自动选择对应后端。

---

## 四、支持的模型速查

### 4.1 检测模型

| 模型族 | 模型名称 | 默认图像尺寸 |
|--------|---------|-------------|
| YOLOv8 | `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x` | [640, 640] |
| YOLOv9 | `yolov9c`, `yolov9e` | [640, 640] |
| YOLOv10 | `yolov10n`, `yolov10s`, `yolov10m`, `yolov10l`, `yolov10x` | [640, 640] |
| YOLO11 | `yolo11n`, `yolo11s`, `yolo11m`, `yolo11l`, `yolo11x` | [640, 640] |
| YOLO12 | `yolo12n`, `yolo12s`, `yolo12m`, `yolo12l`, `yolo12x` | [640, 640] |
| YOLOX | `yolox_n`, `yolox_s`, `yolox_m`, `yolox_l`, `yolox_x` | [800, 1440] |
| RT-DETR | `rtdetr_v2_r18vd`, `rtdetr_v2_r50vd`, `rtdetr_v2_r101vd` | [640, 640] |

### 4.2 ReID 模型

预训练 ReID 权重命名规则：`{架构}_{变体}_{训练数据集}.{格式}`

| 模型 | 说明 |
|------|------|
| `osnet_x0_25_msmt17` | OSNet 25% 宽度，MSMT17 训练（推荐，轻量快速） |
| `osnet_x0_5_msmt17` | OSNet 50% 宽度 |
| `osnet_x0_75_msmt17` | OSNet 75% 宽度 |
| `osnet_x1_0_msmt17` | OSNet 100% 宽度（最高精度） |
| `osnet_ain_x0_25_msmt17` | OSNet-AIN 25% 宽度 |
| `lmbn_n_duke` | LightMBN，DukeMTMC 训练 |
| `lmbn_n_msmt17` | LightMBN，MSMT17 训练 |
| `clip_*` | CLIP-based ReID |

模型文件会自动从 GitHub Releases 下载到 `boxmot/engine/weights/` 目录。

---

## 五、常见场景操作速查

### 5.1 快速上手：对视频进行跟踪

```bash
# 最简单的用法
D:\anaconda3\envs\detr\python.exe -m boxmot.engine.cli track --source video.mp4 --save

# 高精度场景
D:\anaconda3\envs\detr\python.exe -m boxmot.engine.cli track \
    yolov8x osnet_x1_0_msmt17 botsort \
    --source video.mp4 --save --show-trajectories

# 高速度场景（无 ReID）
D:\anaconda3\envs\detr\python.exe -m boxmot.engine.cli track \
    yolov8n bytetrack \
    --source video.mp4 --save
```

### 5.2 在基准数据集上评估

```bash
# MOT17
D:\anaconda3\envs\detr\python.exe -m boxmot.engine.cli eval \
    yolov8n osnet_x0_25_msmt17 deepocsort \
    --source MOT17-ablation

# DanceTrack
D:\anaconda3\envs\detr\python.exe -m boxmot.engine.cli eval \
    yolov8n osnet_x0_25_msmt17 botsort \
    --source dancetrack-ablation
```

### 5.3 调优跟踪器参数

```bash
D:\anaconda3\envs\detr\python.exe -m boxmot.engine.cli tune \
    yolov8n osnet_x0_25_msmt17 deepocsort \
    --source MOT17-ablation \
    --n-trials 50
```

### 5.4 只跟踪特定类别

```bash
# 只跟踪行人（class 0）
D:\anaconda3\envs\detr\python.exe -m boxmot.engine.cli track \
    --source video.mp4 --classes 0 --save

# 跟踪行人和车辆（class 0 和 2）
D:\anaconda3\envs\detr\python.exe -m boxmot.engine.cli track \
    --source video.mp4 --classes 0 2 --save
```

### 5.5 多 GPU 推理

```bash
D:\anaconda3\envs\detr\python.exe -m boxmot.engine.cli track \
    --source video.mp4 --device 0,1,2,3 --save
```

### 5.6 使用自定义训练的模型

```bash
# 使用自己训练的 YOLO 模型
D:\anaconda3\envs\detr\python.exe -m boxmot.engine.cli track \
    --yolo-model /path/to/best.pt \
    --source video.mp4 --save

# 使用自己训练的 ReID 模型
D:\anaconda3\envs\detr\python.exe -m boxmot.engine.cli track \
    yolov8n /path/to/my_reid.pt deepocsort \
    --source video.mp4 --save
```

---

## 六、项目依赖与安装

### 6.1 核心依赖

```bash
# 基础安装
pip install -e .

# 含 YOLO 支持
pip install -e ".[yolo]"

# 含调优功能
pip install -e ".[evolve]"

# 含 ONNX 导出
pip install -e ".[onnx]"

# 含 OpenVINO
pip install -e ".[openvino]"

# 含 TFLite
pip install -e ".[tflite]"
```

### 6.2 关键依赖版本

| 包 | 版本要求 |
|----|---------|
| Python | >=3.9, <3.13 |
| PyTorch | >=2.2.1, <3.0.0 |
| torchvision | >=0.17.1, <1.0.0 |
| ultralytics | >=3.8.200 |
| filterpy | >=1.4.5, <2.0.0 |
| lapx | >=0.5.5, <1.0.0 |
| opencv-python | >=4.7.0, <5.0.0 |
| scikit-learn | >=1.3.0, <2.0.0 |
| numpy | * |
| click | >=8.1.8 |

---

## 七、测试与开发

### 7.1 运行测试

```bash
# 全部测试
D:\anaconda3\envs\detr\python.exe -m pytest

# 单个测试文件
D:\anaconda3\envs\detr\python.exe -m pytest tests/test_cli.py

# 指定单元测试
D:\anaconda3\envs\detr\python.exe -m pytest tests/unit/test_trackers.py
```

### 7.2 代码检查

```bash
# Lint 检查
D:\anaconda3\envs\detr\Scripts\ruff.exe check .

# 格式化
D:\anaconda3\envs\detr\Scripts\ruff.exe format .
```

### 7.3 示例 Notebook

| 文件 | 说明 |
|------|------|
| `examples/det/yolox_boxmot.ipynb` | YOLOX 检测器 + BoxMOT |
| `examples/det/efficientdet_boxmot.ipynb` | EfficientDet + BoxMOT |
| `examples/det/torchvision_boxmot.ipynb` | Torchvision 检测器 + BoxMOT |
| `examples/det/rfdetr_boxmot.ipynb` | RF-DETR + BoxMOT |
| `examples/det/obb.ipynb` | 旋转边界框跟踪 |
| `examples/pose/torchvision_boxmot.ipynb` | 姿态估计 + 跟踪 |
| `examples/seg/torchvision_boxmot.ipynb` | 分割 + 跟踪 |
