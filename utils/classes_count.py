import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# VisDrone2019-MOT 类别映射
CLASS_MAPPING = {
    0: "ignored regions",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others"
}


def count_classes_in_file(label_path):
    """统计单个标签文件中的各类别数量"""
    class_counts = {cls: 0 for cls in CLASS_MAPPING.keys()}

    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 8:  # 确保标签格式正确
                    class_id = int(parts[7])  # 类别ID在第8列
                    if class_id in class_counts:
                        class_counts[class_id] += 1
                    else:
                        print(f"Warning: Unknown class ID {class_id} in {label_path}")
    except Exception as e:
        print(f"Error reading {label_path}: {e}")

    return class_counts


def find_annotation_folders(dataset_path):
    """查找dataset下所有子文件夹中的annotations目录"""
    annotation_folders = []
    subfolders = ["VisDrone2019-MOT-train", "VisDrone2019-MOT-val", "VisDrone2019-MOT-test-dev"]  # 支持的子数据集文件夹

    for subfolder in subfolders:
        subpath = os.path.join(dataset_path, subfolder)
        if not os.path.exists(subpath):
            continue

        annotation_path = os.path.join(subpath, "annotations")
        if os.path.exists(annotation_path):
            annotation_folders.append(annotation_path)
            print(f"找到标注文件夹: {annotation_path}")

    if not annotation_folders:
        print(f"警告: 在 {dataset_path} 下未找到任何标注文件夹 (train/val/test/annotations)")

    return annotation_folders


def analyze_visdrone_dataset(dataset_path, output_dir, plot):
    """分析VisDrone2019-MOT数据集的标签"""
    # 初始化总统计
    total_counts = {cls: 0 for cls in CLASS_MAPPING.keys()}
    total_objects = 0
    processed_files = 0

    # 查找所有标注文件夹
    annotation_folders = find_annotation_folders(dataset_path)
    if not annotation_folders:
        print("错误: 未找到任何标注文件夹，程序退出")
        return

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 按子数据集分别统计
    subset_stats = {}

    for annotation_folder in annotation_folders:
        subset_name = os.path.basename(os.path.dirname(annotation_folder))
        subset_stats[subset_name] = {cls: 0 for cls in CLASS_MAPPING.keys()}

        label_files = sorted([f for f in os.listdir(annotation_folder) if f.endswith('.txt')])
        print(f"\n=== 处理 {subset_name} 子集 ===")
        print(f"找到 {len(label_files)} 个标签文件")

        for label_file in tqdm(label_files, desc="处理标签"):
            label_path = os.path.join(annotation_folder, label_file)
            file_counts = count_classes_in_file(label_path)

            # 累加子集统计
            for cls in subset_stats[subset_name]:
                subset_stats[subset_name][cls] += file_counts[cls]

            # 累加总统计
            for cls in total_counts:
                total_counts[cls] += file_counts[cls]

            processed_files += 1

    # 输出统计结果
    print("\n=== VisDrone2019-MOT 全局类别统计 ===")
    total_objects = sum(total_counts.values())

    # 按数量排序
    sorted_classes = sorted(total_counts.items(), key=lambda x: x[1], reverse=True)

    for cls_id, count in sorted_classes:
        cls_name = CLASS_MAPPING.get(cls_id, f"Unknown({cls_id})")
        percentage = (count / total_objects) * 100 if total_objects > 0 else 0
        print(f"{cls_name} ({cls_id}): {count} ({percentage:.2f}%)")

    print(f"\n总目标数量: {total_objects}")
    print(f"处理的标签文件数量: {processed_files}")
    print(f"有效类别数量: {len([c for c in total_counts.values() if c > 0])}")

    # 输出各子集统计
    print("\n=== 各子集类别统计 ===")
    for subset_name, subset_count in subset_stats.items():
        subset_total = sum(subset_count.values())
        print(f"\n--- {subset_name} 子集 ---")
        print(f"目标数量: {subset_total}")

        subset_sorted = sorted(subset_count.items(), key=lambda x: x[1], reverse=True)
        for cls_id, count in subset_sorted:
            if count == 0:
                continue

            cls_name = CLASS_MAPPING.get(cls_id, f"Unknown({cls_id})")
            percentage = (count / subset_total) * 100 if subset_total > 0 else 0
            print(f"{cls_name} ({cls_id}): {count} ({percentage:.2f}%)")

    # 保存统计结果
    stats_file = os.path.join(output_dir, "class_stats.txt")
    with open(stats_file, 'w') as f:
        f.write("=== VisDrone2019-MOT 全局类别统计 ===\n")
        for cls_id, count in sorted_classes:
            cls_name = CLASS_MAPPING.get(cls_id, f"Unknown({cls_id})")
            percentage = (count / total_objects) * 100 if total_objects > 0 else 0
            f.write(f"{cls_name} ({cls_id}): {count} ({percentage:.2f}%)\n")
        f.write(f"\n总目标数量: {total_objects}\n")
        f.write(f"处理的标签文件数量: {processed_files}\n\n")

        f.write("=== 各子集类别统计 ===\n")
        for subset_name, subset_count in subset_stats.items():
            subset_total = sum(subset_count.values())
            f.write(f"\n--- {subset_name} 子集 ---\n")
            f.write(f"目标数量: {subset_total}\n")

            subset_sorted = sorted(subset_count.items(), key=lambda x: x[1], reverse=True)
            for cls_id, count in subset_sorted:
                if count == 0:
                    continue

                cls_name = CLASS_MAPPING.get(cls_id, f"Unknown({cls_id})")
                percentage = (count / subset_total) * 100 if subset_total > 0 else 0
                f.write(f"{cls_name} ({cls_id}): {count} ({percentage:.2f}%)\n")

    print(f"\n统计结果已保存至: {stats_file}")

    # 生成可视化图表
    if plot:
        plot_classes(total_counts, output_dir, "全局")
        for subset_name, subset_count in subset_stats.items():
            if sum(subset_count.values()) > 0:
                plot_classes(subset_count, output_dir, subset_name)


def plot_classes(class_counts, output_dir, subset_name=""):
    """生成类别分布图表"""
    # 过滤掉数量为0的类别
    filtered_classes = {CLASS_MAPPING[k]: v for k, v in class_counts.items() if v > 0}

    if not filtered_classes:
        print(f"警告: {subset_name} 子集没有可绘制的类别")
        return

    # 准备数据
    classes = list(filtered_classes.keys())
    counts = list(filtered_classes.values())

    # 创建图表
    plt.figure(figsize=(12, 8))
    bars = plt.barh(classes, counts, color='skyblue')
    plt.xlabel('数量')
    plt.ylabel('类别')
    plt.title(f'VisDrone2019-MOT {subset_name} 类别分布')
    plt.tight_layout()

    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        plt.text(width + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{int(width)}', ha='left', va='center')

    # 保存图表
    plot_file = os.path.join(output_dir, f"class_distribution_{subset_name}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"类别分布图表已保存至: {plot_file}")

    plt.close()  # 关闭图表以释放资源


def main():
    parser = argparse.ArgumentParser(description='统计VisDrone2019-MOT数据集的类别分布')
    parser.add_argument('--dataset_path', default='dataset', help='数据集根目录（默认: dataset）')
    parser.add_argument('--output_dir', default='output', help='输出结果保存目录（默认: output）')
    parser.add_argument('--plot', action='store_true', help='生成类别分布图表')
    args = parser.parse_args()

    # 检查数据集路径是否存在
    if not os.path.exists(args.dataset_path):
        print(f"错误: 数据集路径 {args.dataset_path} 不存在!")
        print("请使用 --dataset_path 指定正确的路径")
        return

    analyze_visdrone_dataset(args.dataset_path, args.output_dir, args.plot)


if __name__ == "__main__":
    main()
