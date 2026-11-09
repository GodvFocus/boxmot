import os
import glob

def analyze_occlusion_values(label_folder):
    """
    分析指定文件夹中所有VisDrone2019-MOT标签文件的occlusion字段值
    
    参数:
    label_folder (str): 包含标签文件的文件夹路径
    
    返回:
    set: 包含所有occlusion字段值的集合
    """
    # 存储所有occlusion值的集合
    occlusion_values = set()
    
    # 获取文件夹中所有txt文件
    label_files = glob.glob(os.path.join(label_folder, "*.txt"))
    
    # 检查是否找到标签文件
    if not label_files:
        print(f"在文件夹 {label_folder} 中未找到标签文件！")
        return occlusion_values
    
    print(f"找到 {len(label_files)} 个标签文件，开始分析...")
    
    # 遍历每个标签文件
    for label_file in label_files:
        try:
            # 读取标签文件内容
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            # 遍历每行标签
            for line in lines:
                # 分割标签行，提取occlusion字段（索引为8）
                fields = line.strip().split(',')
                if len(fields) >= 9:  # 确保标签行包含足够的字段
                    occlusion = fields[9]
                    occlusion_values.add(occlusion)
        
        except Exception as e:
            print(f"处理文件 {label_file} 时出错: {e}")
    
    return occlusion_values

def main():
    # 设置标签文件夹路径
    label_folder = r'E:\Experiment\datasets\VisDrone-MOT\VisDrone2019-MOT-test-dev\annotations'
    
    # 检查文件夹是否存在
    if not os.path.isdir(label_folder):
        print(f"错误: 文件夹 {label_folder} 不存在！")
        return
    
    # 分析occlusion值
    occlusion_values = analyze_occlusion_values(label_folder)
    
    # 打印结果
    print("\n所有可能的occlusion值:")
    for value in sorted(occlusion_values, key=lambda x: (x.isdigit(), int(x) if x.isdigit() else x)):
        print(f"- {value}")
    
    print(f"\n总共找到 {len(occlusion_values)} 种不同的occlusion值")

if __name__ == "__main__":
    main()
