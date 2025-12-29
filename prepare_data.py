# prepare_data.py
import os
import shutil
from sklearn.model_selection import train_test_split

def prepare_dataset():
    # 原始数据路径
    source_dir = "data/flowers"
    
    # 目标路径
    splits = ['train', 'val', 'test']
    classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    
    # 创建目录
    for split in splits:
        for cls in classes:
            os.makedirs(f"data/{split}/{cls}", exist_ok=True)
    
    # 划分数据集 (70%训练, 15%验证, 15%测试)
    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        if not os.path.exists(cls_path):
            print(f"警告: {cls_path} 不存在，跳过")
            continue
            
        images = [f for f in os.listdir(cls_path) 
                 if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"{cls}: 找到 {len(images)} 张图片")
        
        # 划分
        train, temp = train_test_split(images, test_size=0.3, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        
        # 复制文件
        for img in train:
            shutil.copy(os.path.join(cls_path, img), 
                       f"data/train/{cls}/{img}")
        for img in val:
            shutil.copy(os.path.join(cls_path, img), 
                       f"data/val/{cls}/{img}")
        for img in test:
            shutil.copy(os.path.join(cls_path, img), 
                       f"data/test/{cls}/{img}")
        
        print(f"  -> 训练集: {len(train)}, 验证集: {len(val)}, 测试集: {len(test)}")
    
    print("\n数据集准备完成！")

if __name__ == "__main__":
    prepare_dataset()