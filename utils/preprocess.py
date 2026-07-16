import os
import argparse
from tqdm import tqdm
import shutil


def process_label_file(input_path, output_path):
    """处理单个标签文件，剔除第0类和第11类，并重新映射类别"""
    with open(input_path, 'r') as f:
        lines = f.readlines()

    processed_lines = []
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) < 8:
            continue  # 跳过格式不正确的行

        class_id = int(parts[7])

        # 剔除第0类和第11类
        if class_id == 0 or class_id == 11:
            continue

        # 重新映射类别：1~10 → 0~9
        new_class_id = class_id - 1
        parts[7] = str(new_class_id)

        processed_lines.append(','.join(parts) + '\n')

    # 保存处理后的标签文件
    with open(output_path, 'w') as f:
        f.writelines(processed_lines)


def process_dataset(input_dir, output_dir, overwrite=False):
    """处理整个数据集，包括训练集、验证集和测试集"""
    # 检查输出目录
    # if os.path.exists(output_dir):
    #     if overwrite:
    #         print(f"警告：输出目录 {output_dir} 已存在，将被覆盖")
    #         shutil.rmtree(output_dir)
    #     else:
    #         print(f"错误：输出目录 {output_dir} 已存在，请使用 --overwrite 参数覆盖")
    #         return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 处理的子集（训练集、验证集、测试集）
    subsets = ['VisDrone2019-MOT-train', 'VisDrone2019-MOT-val', 'VisDrone2019-MOT-test-dev']

    for subset in subsets:
        subset_input_dir = os.path.join(input_dir, subset)
        subset_output_dir = os.path.join(output_dir, subset)

        if not os.path.exists(subset_input_dir):
            print(f"跳过不存在的子集：{subset_input_dir}")
            continue

        # 创建子输出目录
        os.makedirs(subset_output_dir, exist_ok=True)

        # 处理annotations目录，输出到同级的 new_annotations
        annotations_input_dir = os.path.join(subset_input_dir, 'annotations')
        annotations_output_dir = os.path.join(subset_output_dir, 'new_annotations')

        if not os.path.exists(annotations_input_dir):
            print(f"警告：子集 {subset} 中找不到 annotations 目录")
            continue

        os.makedirs(annotations_output_dir, exist_ok=True)

        # 处理所有标签文件
        label_files = [f for f in os.listdir(annotations_input_dir) if f.endswith('.txt')]
        print(f"处理 {subset} 子集：{len(label_files)} 个标签文件")

        for label_file in tqdm(label_files, desc=f"处理 {subset} 标签"):
            input_path = os.path.join(annotations_input_dir, label_file)
            output_path = os.path.join(annotations_output_dir, label_file)
            process_label_file(input_path, output_path)


def main():
    parser = argparse.ArgumentParser(description='处理VisDrone2019-MOT数据集标签，剔除第0类和第11类，并重新映射类别')
    parser.add_argument('--input_dir', default='dataset', help='输入数据集目录')
    parser.add_argument('--output_dir', default = 'output', help='输出数据集目录')
    parser.add_argument('--overwrite', action='store_true', help='覆盖已存在的输出目录')
    args = parser.parse_args()

    # 检查输入目录是否存在
    if not os.path.exists(args.input_dir):
        print(f"错误：输入目录 {args.input_dir} 不存在")
        return

    # 处理数据集
    process_dataset(args.input_dir, args.output_dir, args.overwrite)

    print("\n处理完成！")
    print(f"已处理的数据集保存在：{args.output_dir}")
    print("类别映射说明：")
    print("原类别0和11已被剔除")
    print("原类别1~10已重新映射为0~9")


if __name__ == "__main__":
    main()
