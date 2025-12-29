"""
ResNet18花卉分类交互式演示系统
支持：上传图片预测、实时摄像头识别、结果可视化、模型解释
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
import base64
from io import BytesIO

# 导入必要的库
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Flask/Gradio集成
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("Gradio未安装，使用: pip install gradio")

try:
    from flask import Flask, render_template, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# 设置中文字体（可选）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 花卉类别和颜色
FLOWER_CLASSES = [
    "Daisy",           # 雏菊
    "Dandelion",       # 蒲公英
    "Rose",            # 玫瑰
    "Sunflower",       # 向日葵
    "Tulip"            # 郁金香
]

CLASS_COLORS = {
    "Daisy": "#FFD700",      # 金色
    "Dandelion": "#32CD32",  # 绿色
    "Rose": "#FF69B4",       # 粉色
    "Sunflower": "#FF8C00",  # 橙色
    "Tulip": "#9370DB"       # 紫色
}

class ResNet18FlowerClassifier(nn.Module):
    """ResNet18花卉分类模型（与train.py保持一致）"""
    
    def __init__(self, num_classes=5, pretrained=True, freeze_layers=True):
        super(ResNet18FlowerClassifier, self).__init__()
        
        # 加载预训练ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # 冻结前几层
        if freeze_layers:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # 修改最后一层（与train.py相同）
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

class FlowerClassifierDemo:
    """花卉分类演示系统"""
    
    def __init__(self, model_path=None):
        """
        初始化演示系统
        
        Args:
            model_path: 模型文件路径，如果为None则使用预训练模型
        """
        self.device = device
        self.classes = FLOWER_CLASSES
        self.class_colors = CLASS_COLORS
        
        # 加载模型
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print(f"模型加载完成！可分类: {', '.join(self.classes)}")
    
    def load_model(self, model_path=None):
        """加载模型"""
        model = ResNet18FlowerClassifier(num_classes=len(self.classes))
        
        if model_path and os.path.exists(model_path):
            print(f"加载训练好的模型: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"模型准确率: {checkpoint.get('val_acc', 'N/A')}%")
        else:
            print("使用预训练ResNet18（未微调）")
            # 使用预训练权重，但需要确保最后一层正确
            model = ResNet18FlowerClassifier(
                num_classes=len(self.classes),
                pretrained=True,
                freeze_layers=False
            )
        
        return model.to(self.device)
    
    def predict(self, image):
        """
        预测单张图片
        
        Args:
            image: PIL Image对象
            
        Returns:
            dict: 包含预测结果和可视化信息
        """
        # 预处理
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
        # 获取预测结果
        probs = probabilities[0].cpu().numpy()
        predicted_idx = np.argmax(probs)
        predicted_class = self.classes[predicted_idx]
        confidence = probs[predicted_idx]
        
        # 创建可视化
        result = {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'all_probs': probs.tolist(),
            'visualization': self.create_visualization(image, predicted_class, confidence, probs),
            'gradcam': self.create_gradcam_visualization(image, input_tensor, predicted_idx) if confidence > 0.3 else None
        }
        
        return result
    
    def create_visualization(self, original_img, pred_class, confidence, probs):
        """创建结果可视化图表"""
        fig = plt.figure(figsize=(14, 6))
        
        # 1. 左侧：原始图片 + 预测结果
        ax1 = plt.subplot(1, 3, 1)
        ax1.imshow(original_img)
        ax1.axis('off')
        
        # 添加预测标签
        title_color = self.class_colors.get(pred_class, 'black')
        ax1.set_title(f'预测: {pred_class}\n置信度: {confidence*100:.1f}%', 
                     fontsize=14, color=title_color, fontweight='bold')
        
        # 2. 中间：概率条形图
        ax2 = plt.subplot(1, 3, 2)
        colors = [self.class_colors.get(cls, '#3498db') for cls in self.classes]
        bars = ax2.barh(self.classes, probs, color=colors, edgecolor='black')
        
        # 添加数值标签
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax2.text(prob + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{prob*100:.1f}%', va='center', fontsize=10)
        
        ax2.set_xlabel('概率', fontsize=12)
        ax2.set_xlim([0, 1.1])
        ax2.set_title('各类别概率分布', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        
        # 3. 右侧：置信度仪表盘
        ax3 = plt.subplot(1, 3, 3, polar=True)
        
        # 创建仪表盘
        angles = np.linspace(0, 2 * np.pi, len(self.classes), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        probs_circular = list(probs) + [probs[0]]
        ax3.plot(angles, probs_circular, 'o-', linewidth=2, color='#3498db')
        ax3.fill(angles, probs_circular, alpha=0.25, color='#3498db')
        
        # 设置极坐标
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(self.classes, fontsize=10)
        ax3.set_ylim([0, 1])
        ax3.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax3.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8)
        ax3.set_title('置信度雷达图', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # 保存到内存
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        # 转换为base64
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    
    def create_gradcam_visualization(self, original_img, input_tensor, target_idx):
        """创建Grad-CAM热力图（简化版）"""
        try:
            # 获取最后一个卷积层的特征
            features = None
            gradients = None
            
            def save_features(module, input, output):
                nonlocal features
                features = output
            
            def save_gradients(module, grad_in, grad_out):
                nonlocal gradients
                gradients = grad_out[0]
            
            # 注册钩子
            target_layer = self.model.resnet.layer4[-1].conv2
            handle_forward = target_layer.register_forward_hook(save_features)
            handle_backward = target_layer.register_full_backward_hook(save_gradients)
            
            # 前向传播
            outputs = self.model(input_tensor)
            target = outputs[0, target_idx]
            
            # 反向传播
            self.model.zero_grad()
            target.backward()
            
            # 计算权重
            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
            
            # 加权特征图
            for i in range(features.shape[1]):
                features[:, i, :, :] *= pooled_gradients[i]
            
            heatmap = torch.mean(features, dim=1).squeeze()
            heatmap = torch.nn.functional.relu(heatmap)  # ReLU
            heatmap /= torch.max(heatmap)  # 归一化
            
            # 转换为numpy
            heatmap = heatmap.cpu().detach().numpy()
            
            # 移除钩子
            handle_forward.remove()
            handle_backward.remove()
            
            # 创建热力图叠加
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            # 原始图片
            axes[0].imshow(original_img)
            axes[0].set_title('原始图片', fontsize=12)
            axes[0].axis('off')
            
            # 热力图
            im = axes[1].imshow(heatmap, cmap='jet')
            axes[1].set_title('Grad-CAM热力图', fontsize=12)
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            
            # 叠加图
            original_img_resized = original_img.resize((heatmap.shape[1], heatmap.shape[0]))
            img_array = np.array(original_img_resized) / 255.0
            
            # 创建叠加
            heatmap_resized = np.uint8(255 * heatmap)
            heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
            
            alpha = 0.5
            superimposed = heatmap_colored * alpha + img_array * (1 - alpha)
            
            axes[2].imshow(superimposed)
            axes[2].set_title('热力图叠加', fontsize=12)
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # 保存到内存
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            
            # 转换为base64
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            print(f"Grad-CAM生成失败: {e}")
            return None
    
    def batch_predict(self, image_folder):
        """批量预测文件夹中的图片"""
        results = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        
        for img_file in Path(image_folder).iterdir():
            if img_file.suffix.lower() in image_extensions:
                try:
                    img = Image.open(img_file).convert('RGB')
                    result = self.predict(img)
                    result['filename'] = img_file.name
                    results.append(result)
                except Exception as e:
                    print(f"处理 {img_file} 时出错: {e}")
        
        return results
    
    def create_summary_report(self, batch_results):
        """创建批量预测总结报告"""
        if not batch_results:
            return None
        
        # 统计信息
        total = len(batch_results)
        confidences = [r['confidence'] for r in batch_results]
        classes = [r['predicted_class'] for r in batch_results]
        
        # 创建总结图表
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. 置信度分布
        axes[0].hist(confidences, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(confidences), color='red', linestyle='--', label=f'平均: {np.mean(confidences):.3f}')
        axes[0].set_xlabel('置信度', fontsize=12)
        axes[0].set_ylabel('图片数量', fontsize=12)
        axes[0].set_title('置信度分布', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # 2. 类别分布
        from collections import Counter
        class_counts = Counter(classes)
        
        colors = [self.class_colors.get(cls, '#3498db') for cls in class_counts.keys()]
        bars = axes[1].bar(class_counts.keys(), class_counts.values(), color=colors, edgecolor='black')
        
        # 添加数值标签
        for bar, count in zip(bars, class_counts.values()):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom', fontsize=10)
        
        axes[1].set_xlabel('花卉类别', fontsize=12)
        axes[1].set_ylabel('图片数量', fontsize=12)
        axes[1].set_title('预测类别分布', fontsize=14, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存到内存
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        # 转换为base64
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        # 创建总结文本
        summary = {
            'total_images': total,
            'avg_confidence': float(np.mean(confidences)),
            'class_distribution': dict(class_counts),
            'top_class': max(class_counts, key=class_counts.get),
            'report_chart': f"data:image/png;base64,{img_base64}"
        }
        
        return summary

# ==================== Gradio界面 ====================
if GRADIO_AVAILABLE:
    # 初始化分类器
    classifier = FlowerClassifierDemo(model_path='checkpoints/best_model.pth')
    
    # 示例图片
    example_images = [
        ["sample_images/daisy.jpg", "雏菊示例"],
        ["sample_images/rose.jpg", "玫瑰示例"],
        ["sample_images/sunflower.jpg", "向日葵示例"],
        ["sample_images/tulip.jpg", "郁金香示例"],
        ["sample_images/dandelion.jpg", "蒲公英示例"]
    ]
    
    # 自定义CSS
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .output-image img {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .result-box {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        margin: 10px 0;
    }
    """
    
    def predict_interface(image):
        """Gradio预测接口"""
        if image is None:
            return None, None, "请上传图片"
        
        try:
            # 转换为PIL Image
            if isinstance(image, str):
                img = Image.open(image)
            else:
                img = Image.fromarray(image)
            
            # 预测
            result = classifier.predict(img)
            
            # 创建HTML结果
            html_result = f"""
            <div class="success-box">
                <h3>🌺 预测结果</h3>
                <p><strong>花卉种类:</strong> {result['predicted_class']}</p>
                <p><strong>置信度:</strong> {result['confidence']*100:.2f}%</p>
                <p><strong>模型:</strong> ResNet18 (84.13%测试准确率)</p>
            </div>
            
            <div class="result-box">
                <h4>📊 详细概率:</h4>
                <table style="width:100%">
            """
            
            for cls, prob in zip(classifier.classes, result['all_probs']):
                color = classifier.class_colors.get(cls, '#3498db')
                bar_width = prob * 100
                html_result += f"""
                <tr>
                    <td style="width:30%"><strong>{cls}</strong></td>
                    <td style="width:60%">
                        <div style="background:#e0e0e0; height:20px; border-radius:10px;">
                            <div style="background:{color}; width:{bar_width}%; height:20px; border-radius:10px;"></div>
                        </div>
                    </td>
                    <td style="width:10%; text-align:right">{prob*100:.1f}%</td>
                </tr>
                """
            
            html_result += """
                </table>
            </div>
            
            <div class="result-box">
                <h4>ℹ️ 模型信息:</h4>
                <ul>
                    <li><strong>架构:</strong> ResNet18 with Transfer Learning</li>
                    <li><strong>训练数据:</strong> Kaggle Flowers Recognition (5类)</li>
                    <li><strong>测试准确率:</strong> 84.13%</li>
                    <li><strong>训练时间:</strong> 31.2分钟</li>
                </ul>
            </div>
            """
            
            # 返回结果
            return result['visualization'], html_result
            
        except Exception as e:
            return None, None, f"预测出错: {str(e)}"
    
    def create_gradio_app():
        """创建Gradio应用"""
        with gr.Blocks(title="ResNet18花卉分类演示", css=custom_css) as demo:
            gr.Markdown("# 🌸 ResNet18花卉分类演示系统")
            gr.Markdown("""
            ### 上传花朵图片，体验深度学习分类模型
            - **模型**: ResNet18 with Transfer Learning
            - **准确率**: 84.13% on test set
            - **训练数据**: Kaggle Flowers Recognition (5 classes)
            - **支持**: 单图预测、批量处理、模型解释
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    # 输入组件
                    image_input = gr.Image(
                        type="pil", 
                        label="上传花朵图片",
                        height=300,
                        sources=["upload", "clipboard", "webcam"]
                    )
                    
                    gr.Examples(
                        examples=example_images,
                        inputs=image_input,
                        label="示例图片",
                        examples_per_page=3
                    )
                    
                    submit_btn = gr.Button("🚀 开始分类", variant="primary", size="lg")
                    
                    # 批量处理
                    with gr.Accordion("📁 批量处理（高级功能）", open=False):
                        folder_input = gr.File(
                            label="选择多个图片文件",
                            file_count="multiple",
                            file_types=["image"]
                        )
                        batch_btn = gr.Button("批量处理", variant="secondary")
                
                with gr.Column(scale=1):
                    # 输出组件
                    output_image = gr.Image(
                        label="预测结果可视化",
                        height=400,
                        interactive=False
                    )
                    
                    output_html = gr.HTML(
                        label="详细结果",
                        value="<div style='padding:20px;text-align:center;color:#666;'>等待图片上传...</div>"
                    )
            
            # 模型解释部分
            with gr.Accordion("🔍 模型解释与可视化", open=False):
                gr.Markdown("""
                ### Grad-CAM 可视化
                Grad-CAM (Gradient-weighted Class Activation Mapping) 显示模型关注的图像区域。
                热力图显示模型在做出决策时关注的图像部分。
                """)
                
                cam_output = gr.Image(
                    label="Grad-CAM热力图",
                    interactive=False
                )
            
            # 项目信息
            with gr.Accordion("📚 项目信息", open=False):
                gr.Markdown("""
                ### 项目详情
                - **GitHub仓库**: [ResNet18-Flower-Classification](https://github.com/Yyyyuan1029/ResNet18-Flower-Classification1.0)
                - **完整报告**: [Final Report PDF](Final_report_template.pdf)
                - **团队成员**: Siyuan Luo, Yuran Li
                - **课程**: Macau University of Science and Technology, CS460/EIE460/SE460
                
                ### 技术栈
                - **深度学习框架**: PyTorch 1.13
                - **Web框架**: Gradio 4.0+
                - **可视化**: Matplotlib, Seaborn
                - **数据处理**: NumPy, PIL
                
                ### 模型性能
                | 指标 | 值 |
                |------|-----|
                | 测试准确率 | 84.13% |
                | 最佳验证准确率 | 85.98% |
                | 训练时间 | 31.2分钟 |
                | 模型大小 | 44.7 MB |
                """)
            
            # 事件绑定
            submit_btn.click(
                fn=predict_interface,
                inputs=[image_input],
                outputs=[output_image, output_html]
            )
            
            # 批量处理功能
            def batch_process(files):
                if not files:
                    return None, "请选择文件"
                
                results = []
                for file in files:
                    try:
                        img = Image.open(file.name).convert('RGB')
                        result = classifier.predict(img)
                        results.append({
                            'filename': os.path.basename(file.name),
                            'class': result['predicted_class'],
                            'confidence': f"{result['confidence']*100:.1f}%"
                        })
                    except Exception as e:
                        results.append({
                            'filename': os.path.basename(file.name),
                            'class': '错误',
                            'confidence': str(e)
                        })
                
                # 创建结果表格
                html_table = """
                <div style="background:#f8f9fa;padding:20px;border-radius:10px;">
                    <h3>批量处理结果</h3>
                    <table style="width:100%;border-collapse:collapse;">
                        <tr style="background:#3498db;color:white;">
                            <th style="padding:10px;text-align:left;">文件名</th>
                            <th style="padding:10px;text-align:left;">预测类别</th>
                            <th style="padding:10px;text-align:left;">置信度</th>
                        </tr>
                """
                
                for i, result in enumerate(results):
                    bg_color = "#ffffff" if i % 2 == 0 else "#f2f2f2"
                    color = classifier.class_colors.get(result['class'], '#666666')
                    
                    html_table += f"""
                    <tr style="background:{bg_color};">
                        <td style="padding:10px;border-bottom:1px solid #ddd;">{result['filename']}</td>
                        <td style="padding:10px;border-bottom:1px solid #ddd;">
                            <span style="color:{color};font-weight:bold;">{result['class']}</span>
                        </td>
                        <td style="padding:10px;border-bottom:1px solid #ddd;">{result['confidence']}</td>
                    </tr>
                    """
                
                html_table += "</table></div>"
                return None, html_table
            
            batch_btn.click(
                fn=batch_process,
                inputs=[folder_input],
                outputs=[output_image, output_html]
            )
        
        return demo
    
    # 运行Gradio应用
    def run_gradio():
        demo = create_gradio_app()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,  # 创建公开链接
            debug=False,
            show_error=True
        )
    
    # 主函数
    if __name__ == "__main__":
        print("=" * 60)
        print("🌺 ResNet18花卉分类演示系统")
        print("=" * 60)
        print(f"模型设备: {device}")
        print(f"可分类: {', '.join(FLOWER_CLASSES)}")
        print("\n访问地址:")
        print("本地: http://localhost:7860")
        print("公开链接将在启动后显示")
        print("=" * 60)
        
        run_gradio()

else:
    print("请先安装Gradio: pip install gradio")
    print("或使用Flask版本，运行: python demo/app_flask.py")