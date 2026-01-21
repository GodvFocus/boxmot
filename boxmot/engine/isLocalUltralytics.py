import sys
import ultralytics

# 打印sys.path的查找顺序（核心依据）
print("=== sys.path 查找顺序（优先级从上到下）===")
for idx, path in enumerate(sys.path):
    print(f"{idx+1}. {path}")

# 打印实际加载的ultralytics包的根路径（最关键）
print("\n=== 实际加载的ultralytics路径 ===")
print(ultralytics.__file__)

# 验证nn.modules的路径
from ultralytics.nn import modules
print("\n=== nn.modules 实际路径 ===")
print(modules.__file__)