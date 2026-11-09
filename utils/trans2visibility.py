import csv
import os
from glob import glob

def estimate_visibility_discrete(trunc: int, occ: int) -> float:
    if trunc == 0 and occ == 0:
        return 1.0
    elif trunc == 0 and occ == 1:
        return 0.9
    elif trunc == 1 and occ == 0:
        return 0.8
    elif trunc == 0 and occ == 2:
        return 0.7
    elif trunc == 1 and occ == 1:
        return 0.5
    elif trunc == 1 and occ == 2:
        return 0.3
    else:
        return 0.0  # fallback

def convert_visdrone_to_mot_all_classes(input_path: str, output_gt_path: str):
    os.makedirs(os.path.dirname(output_gt_path), exist_ok=True)

    with open(input_path, 'r') as fin, open(output_gt_path, 'w', newline='') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)

        for row in reader:
            if len(row) != 10:
                continue

            frame = int(row[0])
            target_id = int(row[1])
            x = float(row[2])
            y = float(row[3])
            w = float(row[4])
            h = float(row[5])
            score = float(row[6])
            obj_class = int(row[7])
            truncation = int(row[8])
            occlusion = int(row[9])

            visibility = estimate_visibility_discrete(truncation, occlusion)

            mot_row = [frame, target_id, x, y, w, h, score, obj_class, visibility]
            writer.writerow(mot_row)

    print(f"✅ 转换完成：{input_path} → {output_gt_path}")

def batch_convert_visdrone_folder(input_dir: str, output_root: str):
    input_files = glob(os.path.join(input_dir, "*.txt"))

    for input_path in input_files:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_gt_path = os.path.join(output_root, base_name, "gt", "gt.txt")
        convert_visdrone_to_mot_all_classes(input_path, output_gt_path)

    print(f"\n✅ 所有文件已处理完成，共 {len(input_files)} 个视频。")

if __name__ == "__main__":
    input_annotation_dir = r"E:\Experiment\datasets\VisDrone-MOT\VisDrone2019-MOT-val\annotations"
    output_dir = r"E:\Experiment\datasets\VisDrone-MOT\VisDrone2019-MOT-val\annotations_mot"

    batch_convert_visdrone_folder(input_annotation_dir, output_dir)

