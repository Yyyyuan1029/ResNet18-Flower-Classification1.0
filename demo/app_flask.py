import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename
import json
import base64
from io import BytesIO

# 导入ResNet18模型
from torchvision.models import resnet18

# 花卉类别信息
FLOWER_CLASSES = [
    {
        "id": 0,
        "name": "Daisy",
        "chinese": "雏菊",
        "color": "#FFD700",
        "icon": "🌼",
        "description": "雏菊是菊科植物的一种，常见于欧洲，花语是天真、和平、希望。"
    },
    {
        "id": 1,
        "name": "Dandelion",
        "chinese": "蒲公英",
        "color": "#FFA500",
        "icon": "🌼",
        "description": "蒲公英是菊科蒲公英属植物，具有药用价值，花语是勇敢、自信、自由。"
    },
    {
        "id": 2,
        "name": "Rose",
        "chinese": "玫瑰",
        "color": "#FF1493",
        "icon": "🌹",
        "description": "玫瑰是蔷薇科植物，象征爱情与美丽，花语是爱情、浪漫、热情。"
    },
    {
        "id": 3,
        "name": "Sunflower",
        "chinese": "向日葵",
        "color": "#FFD700",
        "icon": "🌻",
        "description": "向日葵是菊科植物，面向太阳生长，花语是忠诚、阳光、积极。"
    },
    {
        "id": 4,
        "name": "Tulip",
        "chinese": "郁金香",
        "color": "#800080",
        "icon": "🌷",
        "description": "郁金香是百合科植物，原产中亚，花语是永恒的爱、高贵、优雅。"
    }
]

# 花卉分类器类
class FlowerClassifier:
    def __init__(self, model_path=None, device='cpu'):
        """
        初始化花卉分类器
        
        Args:
            model_path: 模型文件路径
            device: 运行设备 ('cpu' 或 'cuda')
        """
        self.device = torch.device(device)
        self.classes = [cls["name"] for cls in FLOWER_CLASSES]
        self.model = self._load_model(model_path)
        self.transform = self._get_transforms()
        
    def _load_model(self, model_path):
        """
        加载预训练模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            ResNet18模型
        """
        print("使用预训练ResNet18权重")
        model = resnet18(pretrained=True)
        
        # 修改最后的全连接层，适应5个类别
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(self.classes))
        
        # 如果有保存的模型权重，加载它们
        if model_path and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"加载模型权重: {model_path}")
        
        model = model.to(self.device)
        model.eval()  # 设置为评估模式
        return model
    
    def _get_transforms(self):
        """
        获取图像预处理变换
        
        Returns:
            图像变换组合
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image):
        """
        预测单张图片
        
        Args:
            image: PIL Image对象
            
        Returns:
            dict: 预测结果
        """
        try:
            # 预处理
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 预测
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # 获取结果
            probs = probabilities[0].cpu().numpy()
            predicted_idx = np.argmax(probs)
            predicted_class = self.classes[predicted_idx]
            confidence = float(probs[predicted_idx])
            
            # 获取详细概率
            class_probs = []
            for i, (cls_name, prob) in enumerate(zip(self.classes, probs)):
                flower_info = FLOWER_CLASSES[i]
                class_probs.append({
                    "id": flower_info["id"],
                    "name": cls_name,
                    "chinese": flower_info["chinese"],
                    "probability": float(prob),
                    "color": flower_info["color"],
                    "icon": flower_info["icon"]
                })
            
            # 按概率排序
            class_probs.sort(key=lambda x: x["probability"], reverse=True)
            
            # 创建结果
            result = {
                "success": True,
                "predicted_class": predicted_class,
                "predicted_chinese": FLOWER_CLASSES[predicted_idx]["chinese"],
                "confidence": confidence,
                "class_probabilities": class_probs,
                "top_3": class_probs[:3],
                "visualization": self._create_visualization(image, predicted_class, confidence, probs),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"预测失败: {str(e)}"
            }
    
    def _create_visualization(self, image, predicted_class, confidence, probs):
        """
        创建可视化结果
        
        Args:
            image: PIL Image对象
            predicted_class: 预测类别
            confidence: 置信度
            probs: 各类别概率
            
        Returns:
            dict: 可视化数据
        """
        # 查找预测类别的颜色
        predicted_color = "#4facfe"  # 默认颜色
        for flower in FLOWER_CLASSES:
            if flower["name"] == predicted_class:
                predicted_color = flower["color"]
                break
        
        # 创建概率条形图数据
        bar_chart_data = []
        for i, cls in enumerate(self.classes):
            flower_info = FLOWER_CLASSES[i]
            bar_chart_data.append({
                "class": cls,
                "chinese": flower_info["chinese"],
                "probability": float(probs[i]) * 100,
                "color": flower_info["color"]
            })
        
        # 对条形图数据按概率排序
        bar_chart_data.sort(key=lambda x: x["probability"], reverse=True)
        
        # 将图像转换为base64字符串
        buffered = BytesIO()
        image_resized = image.resize((300, 300))
        image_resized.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "bar_chart": bar_chart_data,
            "image_base64": img_str,
            "predicted_color": predicted_color
        }

# 创建Flask应用
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = 'flower-classification-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB最大文件大小

# 创建必要的文件夹
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# 初始化花卉分类器
print("初始化花卉分类器...")
flower_classifier = FlowerClassifier(device='cpu')
print("花卉分类器初始化完成")
print(f"设备: {flower_classifier.device}")
print(f"类别: {', '.join(flower_classifier.classes)}")

# 主页路由
@app.route('/')
def index():
    """
    渲染主页
    """
    return render_template('index.html', 
                         flower_classes=FLOWER_CLASSES,
                         model_info={
                             "architecture": "ResNet18",
                             "num_classes": len(FLOWER_CLASSES),
                             "parameters": "11.2M",
                             "training_time": "31.2分钟"
                         })

# 预测路由
@app.route('/predict', methods=['POST'])
def predict():
    """
    处理图片上传并返回预测结果
    """
    try:
        # 1. 检查是否有文件上传
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': '没有选择文件'
            }), 400
        
        file = request.files['file']
        
        # 2. 检查文件名
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': '未选择文件'
            }), 400
        
        # 3. 检查文件类型
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({
                'success': False,
                'error': '不支持的文件类型，请上传图片文件'
            }), 400
        
        # 4. 保存文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 5. 打开图片并预测
        image = Image.open(filepath).convert('RGB')
        
        # 调用预测方法
        result = flower_classifier.predict(image)
        
        # 6. 添加文件名到结果
        result['filename'] = filename
        result['filepath'] = filepath
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"预测错误: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': f'预测失败: {str(e)}'
        }), 500

# 获取上传的文件
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    返回上传的文件
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# 示例图片路由
@app.route('/sample_images')
def sample_images():
    """
    返回示例图片列表
    """
    samples = []
    sample_dir = os.path.join('static', 'samples')
    
    if os.path.exists(sample_dir):
        for i, cls in enumerate(flower_classifier.classes):
            img_path = os.path.join(sample_dir, f"{cls.lower()}.jpg")
            if os.path.exists(img_path):
                with open(img_path, "rb") as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode()
                
                samples.append({
                    "class": cls,
                    "chinese": FLOWER_CLASSES[i]["chinese"],
                    "image_base64": img_base64
                })
    
    return jsonify({"samples": samples})

# 模型信息路由
@app.route('/model_info')
def model_info():
    """
    返回模型信息
    """
    return jsonify({
        "architecture": "ResNet18",
        "num_classes": len(FLOWER_CLASSES),
        "parameters": "11.2M",
        "training_time": "31.2分钟",
        "device": str(flower_classifier.device),
        "classes": flower_classifier.classes
    })

# Favicon路由
@app.route('/favicon.ico')
def favicon():
    """
    返回favicon
    """
    favicon_path = os.path.join(app.static_folder, 'favicon.ico')
    if os.path.exists(favicon_path):
        return send_file(favicon_path)
    else:
        # 返回一个简单的图标
        from flask import Response
        # 一个1x1像素的透明ICO
        favicon_base64 = "AAABAAEAEBAAAAEAIABoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAQAABILAAASCwAAAAAAAAAAAACZmZn/mZmZ/5mZmf+ZmZn/mZmZ/5mZmf+ZmZn/mZmZ/5mZmf+ZmZn/mZmZ/5mZmf+ZmZn/mZmZ/5mZmf+ZmZn/mZmZ/5mZmf+ZmZn/mZmZ/5mZmf+ZmZn/mZmZ/5mZmf+ZmZn/mZmZ/5mZmf+ZmZn/mZmZ/5mZmf+ZmZn/mZmZ/5mZmf+ZmZn/mZmZ/5mZmf+ZmZn/mZmZ/5mZmf+ZmZn/mZmZ/5mZmf+ZmZn/mZmZ/5mZmf8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        return Response(base64.b64decode(favicon_base64), mimetype='image/x-icon')

# 健康检查路由
@app.route('/health')
def health():
    """
    健康检查
    """
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_loaded": True,
        "device": str(flower_classifier.device)
    })

# 错误处理
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "请求的资源不存在"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "服务器内部错误"}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "文件太大，最大支持16MB"}), 413

if __name__ == '__main__':
    print("\n" + "="*50)
    print("ResNet18花卉分类演示系统 - Flask版本")
    print("="*50)
    print(f"上传文件夹: {app.config['UPLOAD_FOLDER']}")
    print(f"访问地址: http://localhost:5000")
    print(f"模型类别: {len(FLOWER_CLASSES)}类")
    print(f"设备: {flower_classifier.device}")
    print("="*50 + "\n")
    
    # 创建示例图片文件夹
    sample_dir = os.path.join('static', 'samples')
    os.makedirs(sample_dir, exist_ok=True)
    print("示例图片准备完成")
    
    # 运行Flask应用
    app.run(debug=True, host='0.0.0.0', port=5000)