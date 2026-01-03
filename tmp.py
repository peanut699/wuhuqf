from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
import base64
import os
import yaml
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from utils import get_obj_from_string

app = Flask(__name__)
CORS(app)

# 全局变量
model = None
device = None
config = None
temp_dir = './temp_uploads'

def initialize_model(config_path, checkpoint='best', cuda_id=0):
    """初始化模型"""
    global model, device, config
    
    # 设置设备
    device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 创建临时目录
    os.makedirs(temp_dir, exist_ok=True)
    
    # 构建模型路径
    config['model_name'] = config['model_file'].split('.')[1]
    log_path = f"./exp_results/{config['name']}_{config['model_name']}_lr{config['lr']}_wd{config['weight_decay']}_epoch{config['num_epochs']}_revweight{config['rev_weight']}_Aweight{config['a_weight']}_log/"
    snapshot_path = os.path.join(log_path, 'weights/')
    
    # 加载模型
    in_channels = 3
    num_class = 3
    model = get_obj_from_string(config['model_file'])(
        in_channels, num_class, img_size=config['img_size']
    ).to(device)
    
    # 加载权重
    ckpt_map = {
        'best': 'best_val.pth',
        'best2': 'best_val2.pth',
        'musiq': 'best_musiq.pth',
        'last': 'latest.pth',
        'p': 'best_psnr_val.pth',
        's': 'best_ssim_val.pth'
    }
    ckpt_file = ckpt_map.get(checkpoint, 'best_val.pth')
    ckpt_path = os.path.join(snapshot_path, ckpt_file)
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['net'])
    model.eval()
    
    print(f'Model initialized successfully with checkpoint: {ckpt_file}')

def perform_enhancement(image, resize=0, return_maps=False):
    """执行图像增强推理"""
    global model, device
    
    # 转换为tensor
    image_tensor = TF.to_tensor(image).unsqueeze(0)  # Add batch dimension
    
    # 存储原始大小
    _, _, h_orig, w_orig = image_tensor.shape
    
    # 调整大小（如果需要）
    if resize > 0:
        image_tensor = F.interpolate(
            image_tensor, size=(resize, resize), mode='bilinear'
        )
    
    image_tensor = image_tensor.to(device)
    
    # 推理
    with torch.no_grad():
        pred_ = model(image_tensor)
        
        # 处理不同的输出格式
        if isinstance(pred_, tuple):
            pred, D_logits, B_logits, _, _ = pred_
        else:
            pred = pred_
            D_logits = None
            B_logits = None
        
        # 调整回原始大小
        if resize > 0:
            pred = F.interpolate(pred, size=(h_orig, w_orig), mode='bilinear')
            if D_logits is not None:
                D_logits = F.interpolate(D_logits, size=(h_orig, w_orig), mode='bilinear')
            if B_logits is not None:
                B_logits = F.interpolate(B_logits, size=(h_orig, w_orig), mode='bilinear')
    
    # 准备返回结果
    result = {
        'enhanced_image': pred,
        'D_map': D_logits,
        'B_map': B_logits
    }
    
    return result

def tensor_to_base64(tensor):
    """将tensor转换为base64编码的图像"""
    # 转换为PIL图像
    img_np = tensor.squeeze(0).cpu().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    
    # 转换为base64
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    img_data = base64.b64encode(img_io.read()).decode('utf-8')
    
    return img_data

def map_to_base64(tensor):
    """将传输图转换为base64编码的可视化图像"""
    if tensor is None:
        return None
    
    # 提取红色通道
    map_np = tensor.detach().cpu().numpy()[0][0]
    
    # 创建可视化
    plt.figure(figsize=(10, 8))
    plt.imshow(map_np, cmap='jet')
    plt.colorbar()
    plt.axis('off')
    
    # 保存到BytesIO
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    
    img_io.seek(0)
    img_data = base64.b64encode(img_io.read()).decode('utf-8')
    
    return img_data

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'device': str(device) if device else 'not initialized'
    })

@app.route('/enhance', methods=['POST'])
def enhance_image():
    """图像增强接口"""
    try:
        # 检查模型是否已加载
        if model is None:
            return jsonify({'error': 'Model not initialized'}), 500
        
        # 检查是否有图像
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        # 获取参数
        image_file = request.files['image']
        resize = int(request.form.get('resize', 0))
        return_maps = request.form.get('return_maps', 'false').lower() == 'true'
        
        # 读取图像
        image = Image.open(image_file).convert('RGB')
        
        print(f'Processing image: {image_file.filename}, size: {image.size}')
        
        # 执行推理
        result = perform_enhancement(image, resize=resize, return_maps=return_maps)
        
        # 准备响应
        response_data = {
            'status': 'success',
            'enhanced_image': tensor_to_base64(result['enhanced_image']),
            'original_size': list(image.size)
        }
        
        # 如果需要返回传输图
        if return_maps:
            if result['D_map'] is not None:
                response_data['D_map'] = map_to_base64(result['D_map'])
            if result['B_map'] is not None:
                response_data['B_map'] = map_to_base64(result['B_map'])
        
        print(f'Enhancement completed successfully')
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f'Error during enhancement: {str(e)}')
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/batch-enhance', methods=['POST'])
def batch_enhance():
    """批量图像增强接口"""
    try:
        if model is None:
            return jsonify({'error': 'Model not initialized'}), 500
        
        # 检查是否有图像
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400
        
        images = request.files.getlist('images')
        resize = int(request.form.get('resize', 0))
        
        results = []
        
        for idx, image_file in enumerate(images):
            try:
                image = Image.open(image_file).convert('RGB')
                result = perform_enhancement(image, resize=resize)
                
                results.append({
                    'filename': image_file.filename,
                    'status': 'success',
                    'enhanced_image': tensor_to_base64(result['enhanced_image'])
                })
                
                print(f'Processed {idx + 1}/{len(images)}: {image_file.filename}')
                
            except Exception as e:
                results.append({
                    'filename': image_file.filename,
                    'status': 'error',
                    'error': str(e)
                })
        
        return jsonify({
            'status': 'completed',
            'total': len(images),
            'results': results
        })
    
    except Exception as e:
        print(f'Error during batch enhancement: {str(e)}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Underwater Image Enhancement Server')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--ckpt', type=str, default='best', help='Checkpoint to use')
    parser.add_argument('--cuda_id', type=int, default=0, help='CUDA device ID')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=5001, help='Server port')
    
    args = parser.parse_args()
    
    # 初始化模型
    print('Initializing model...')
    initialize_model(args.config, checkpoint=args.ckpt, cuda_id=args.cuda_id)
    print('Model initialization completed')
    
    # 启动服务器
    print(f'Starting server on {args.host}:{args.port}')
    app.run(host=args.host, port=args.port, debug=False)
