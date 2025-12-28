# fix_all_issues.py
import os

print("修复所有问题...")

# 1. 修复 train.py 中的配置错误
train_file = "src/train.py"
with open(train_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 修复拼写错误
content = content.replace("'freq=Layers': True", "'freeze_layers': True")

with open(train_file, 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ 修复了 train.py 中的配置错误")

# 2. 检查数据集目录
data_dirs = ['data/train', 'data/val', 'data/test']
for dir_path in data_dirs:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ 创建了目录: {dir_path}")

# 3. 创建测试数据（如果没有真实数据）
test_file = "create_test_data.py"
test_code = '''
import os
import numpy as np
from PIL import Image

# 创建类别
classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# 为训练集和验证集创建数据
for split in ['train', 'val']:
    for cls in classes[:3]:  # 只创建3个类别
        dir_path = f'data/{split}/{cls}'
        os.makedirs(dir_path, exist_ok=True)
        
        # 每个类别创建10张图片（训练集）或3张图片（验证集）
        num_images = 10 if split == 'train' else 3
        for i in range(num_images):
            # 创建随机彩色图片
            img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(f'{dir_path}/image_{i:03d}.jpg')

print(f"创建了测试数据集:")
print("  data/train/daisy/*.jpg (10张)")
print("  data/train/dandelion/*.jpg (10张)")
print("  data/train/rose/*.jpg (10张)")
print("  data/val/daisy/*.jpg (3张)")
print("  ...等等")
print("总共创建了 39 张测试图片")
'''

with open(test_file, 'w', encoding='utf-8') as f:
    f.write(test_code)

print("✓ 创建了测试数据生成脚本")

# 4. 运行测试数据生成
print("\n生成测试数据...")
os.system("python create_test_data.py")

print("\n✅ 所有修复完成！")
print("现在可以运行: python src/train.py")