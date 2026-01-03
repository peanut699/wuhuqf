# wuhuqf

https://peanut699.github.io/wuhuqf/syh_20251119.pdf

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import argparse
import os
import yaml
import matplotlib.pyplot as plt
import scipy.io as sio
from utils import get_obj_from_string

parser = argparse.ArgumentParser(description='Single Image Inference for Underwater Image Enhancement')
parser.add_argument('--cuda_id', type=int, default=0, help="CUDA device ID, default: 0")
parser.add_argument('--exp', type=str, required=True, help="Path to experiment config file")
parser.add_argument('--ckpt', type=str, default="best", help="Checkpoint to use: best/best2/musiq/last/p/s")
parser.add_argument('--input', type=str, required=True, help="Path to input image")
parser.add_argument('--output', type=str, default="./output", help="Output directory")
parser.add_argument('--resize', type=int, default=0, help="Resize input image (0 = no resize)")
parser.add_argument('--save_maps', action='store_true', help="Save transmission maps (D and B)")
args = parser.parse_args()

# Load config
with open(args.exp, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Set device
device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create output directories
output_dir = args.output
os.makedirs(output_dir, exist_ok=True)
if args.save_maps:
    maps_dir = os.path.join(output_dir, 'maps')
    os.makedirs(maps_dir, exist_ok=True)

# Get model name and checkpoint path
config['model_name'] = config['model_file'].split('.')[1]
log_path = f"./exp_results/{config['name']}_{config['model_name']}_lr{config['lr']}_wd{config['weight_decay']}_epoch{config['num_epochs']}_revweight{config['rev_weight']}_Aweight{config['a_weight']}_log/"
snapshot_path = os.path.join(log_path, 'weights/')

# Load model
in_channels = 3
num_class = 3
net = get_obj_from_string(config['model_file'])(
    in_channels, num_class, img_size=config['img_size']
).to(device)

# Load checkpoint
print(f'Loading checkpoint...')
ckpt_map = {
    'best': 'best_val.pth',
    'best2': 'best_val2.pth',
    'musiq': 'best_musiq.pth',
    'last': 'latest.pth',
    'p': 'best_psnr_val.pth',
    's': 'best_ssim_val.pth'
}
ckpt_file = ckpt_map.get(args.ckpt, 'best_val.pth')
ckpt_path = os.path.join(snapshot_path, ckpt_file)

if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

ckpt = torch.load(ckpt_path, map_location=device)
net.load_state_dict(ckpt['net'])
print(f'Loaded checkpoint: {ckpt_file}')

# Load and preprocess image
if not os.path.exists(args.input):
    raise FileNotFoundError(f"Input image not found: {args.input}")

print(f'Loading image: {args.input}')
img = Image.open(args.input).convert('RGB')
image_tensor = TF.to_tensor(img).unsqueeze(0)  # Add batch dimension

# Store original size
_, _, h_orig, w_orig = image_tensor.shape

# Resize if needed
if args.resize > 0:
    image_tensor = F.interpolate(
        image_tensor, size=(args.resize, args.resize), mode='bilinear'
    )
    print(f'Resized image from ({h_orig}, {w_orig}) to ({args.resize}, {args.resize})')

image_tensor = image_tensor.to(device)

# Inference
print('Running inference...')
net.eval()
with torch.no_grad():
    pred_ = net(image_tensor)
    
    # Handle different output formats
    if isinstance(pred_, tuple):
        pred, D_logits, B_logits, _, _ = pred_
    else:
        pred = pred_
        D_logits = None
        B_logits = None
    
    # Resize back to original size if needed
    if args.resize > 0:
        pred = F.interpolate(pred, size=(h_orig, w_orig), mode='bilinear')
        if D_logits is not None:
            D_logits = F.interpolate(D_logits, size=(h_orig, w_orig), mode='bilinear')
        if B_logits is not None:
            B_logits = F.interpolate(B_logits, size=(h_orig, w_orig), mode='bilinear')

# Get output filename
input_name = os.path.splitext(os.path.basename(args.input))[0]
output_path = os.path.join(output_dir, f'{input_name}_enhanced.png')

# Save enhanced image
torchvision.utils.save_image(pred, output_path)
print(f'Saved enhanced image to: {output_path}')

# Save transmission maps if requested
if args.save_maps:
    if D_logits is not None:
        D_logits_np = D_logits.detach().cpu().numpy()
        D_logits_img_R = D_logits_np[0][0]  # Red channel
        d_map_path = os.path.join(maps_dir, f'{input_name}_D_map.mat')
        sio.savemat(d_map_path, {'trans': D_logits_img_R})
        print(f'Saved D transmission map to: {d_map_path}')
        
        # Save visual representation
        plt.figure(figsize=(10, 8))
        plt.imshow(D_logits_img_R, cmap='jet')
        plt.colorbar()
        plt.title('D Transmission Map')
        plt.axis('off')
        d_vis_path = os.path.join(maps_dir, f'{input_name}_D_map.png')
        plt.savefig(d_vis_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f'Saved D map visualization to: {d_vis_path}')
    
    if B_logits is not None:
        B_logits_np = B_logits.detach().cpu().numpy()
        B_logits_img_R = B_logits_np[0][0]  # Red channel
        b_map_path = os.path.join(maps_dir, f'{input_name}_B_map.mat')
        sio.savemat(b_map_path, {'trans': B_logits_img_R})
        print(f'Saved B transmission map to: {b_map_path}')
        
        # Save visual representation
        plt.figure(figsize=(10, 8))
        plt.imshow(B_logits_img_R, cmap='jet')
        plt.colorbar()
        plt.title('B Transmission Map')
        plt.axis('off')
        b_vis_path = os.path.join(maps_dir, f'{input_name}_B_map.png')
        plt.savefig(b_vis_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f'Saved B map visualization to: {b_vis_path}')

print('\nInference completed successfully!')
print(f'Output saved to: {output_dir}')
