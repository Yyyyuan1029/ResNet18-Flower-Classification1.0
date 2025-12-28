# src/complete_eval.py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import os
import sys

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.model import ResNet18FlowerClassifier
from src.dataset import get_data_loaders

def comprehensive_evaluation(model_path='checkpoints/best_model_acc_85.98.pth'):
    """
    综合评估模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("ResNet18 花卉分类模型综合评估")
    print("=" * 60)
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    model = ResNet18FlowerClassifier(
        num_classes=config['num_classes'],
        pretrained=True,
        freeze_layers=False  # 加载时不冻结
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载数据
    _, _, test_loader = get_data_loaders(
        data_dir=config['data_dir'],
        batch_size=16
    )
    
    # 收集所有预测和标签
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("进行预测...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 1. 总体准确率
    accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"\n 总体准确率: {accuracy*100:.2f}%")
    
    # 2. 各类别准确率
    classes = test_loader.dataset.classes
    class_accuracy = {}
    
    print(f"\n 各类别准确率:")
    for i, class_name in enumerate(classes):
        class_mask = np.array(all_labels) == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(np.array(all_preds)[class_mask] == np.array(all_labels)[class_mask])
            class_accuracy[class_name] = class_acc
            print(f"  {class_name}: {class_acc*100:.2f}% ({np.sum(class_mask)}张)")
    
    # 3. 分类报告
    print(f"\n 详细分类报告:")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))
    
    # 4. 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.title('混淆矩阵 - ResNet18 花卉分类', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('docs/images/confusion_matrix_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. 各类别精度和召回率可视化
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 精度条形图
    bars1 = ax1.bar(range(len(classes)), precision, color='skyblue')
    ax1.set_xlabel('类别', fontsize=12)
    ax1.set_ylabel('精度 (Precision)', fontsize=12)
    ax1.set_title('各类别精度', fontsize=14)
    ax1.set_xticks(range(len(classes)))
    ax1.set_xticklabels(classes, rotation=45, ha='right')
    ax1.set_ylim([0, 1])
    
    # 在条形上添加数值
    for bar, val in zip(bars1, precision):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom')
    
    # 召回率条形图
    bars2 = ax2.bar(range(len(classes)), recall, color='lightcoral')
    ax2.set_xlabel('类别', fontsize=12)
    ax2.set_ylabel('召回率 (Recall)', fontsize=12)
    ax2.set_title('各类别召回率', fontsize=14)
    ax2.set_xticks(range(len(classes)))
    ax2.set_xticklabels(classes, rotation=45, ha='right')
    ax2.set_ylim([0, 1])
    
    # 在条形上添加数值
    for bar, val in zip(bars2, recall):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('docs/images/precision_recall_by_class.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. 保存结果到CSV
    results_df = pd.DataFrame({
        '类别': classes,
        '样本数量': [np.sum(np.array(all_labels) == i) for i in range(len(classes))],
        '准确率': [class_accuracy.get(cls, 0) for cls in classes],
        '精度': precision,
        '召回率': recall,
        'F1分数': f1
    })
    
    os.makedirs('docs/results', exist_ok=True)
    results_df.to_csv('docs/results/classification_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n 详细结果已保存到: docs/results/classification_results.csv")
    
    # 7. 生成总结报告
    print(f"\n{'='*60}")
    print(" 模型性能总结")
    print(f"{'='*60}")
    print(f" 总体准确率: {accuracy*100:.2f}%")
    print(f" 最佳类别: {classes[np.argmax(precision)]} (精度: {np.max(precision)*100:.2f}%)")
    print(f" 最差类别: {classes[np.argmin(precision)]} (精度: {np.min(precision)*100:.2f}%)")
    print(f" 平均精度: {np.mean(precision)*100:.2f}%")
    print(f" 平均召回率: {np.mean(recall)*100:.2f}%")
    print(f" 平均F1分数: {np.mean(f1)*100:.2f}%")
    
    return accuracy, class_accuracy

if __name__ == "__main__":
    comprehensive_evaluation()