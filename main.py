from __future__ import absolute_import, division, print_function
import datetime
import cv2
import math
import natsort
import os
import sys
import glob
import argparse
import time  # 已经导入
import io
import multiprocessing

import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import pynng
from pynng import nng

import torch
from torchvision import transforms, datasets



from improved_seathru import *
from matplotlib import pyplot as plt


import warnings
warnings.filterwarnings(action='ignore')

import os
import cv2
import numpy as np
from PIL import Image
import argparse
import natsort
from skimage.restoration import denoise_tv_chambolle, estimate_sigma

import numpy as np
import random
import copy

from UIQM import calculate_uiqm_from_pil

import os

import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

# from depth_anything.dpt import DepthAnything
from depth_anything_v2.dpt import DepthAnythingV2
# from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet



from datetime import datetime

import numpy as np
from PIL import Image
import torch
import os
import logging
from arch import deep_wb_model, deep_wb_single_task
import utilities.utils as utls
from utilities.deepWB import deep_wb
import arch.splitNetworks as splitter
from torchvision import transforms
from torchvision.transforms import ToPILImage

from skimage import exposure, color, filters
from skimage.restoration import denoise_tv_chambolle, estimate_sigma
from skimage.filters import gaussian
from scipy import ndimage
import cv2
import warnings
warnings.filterwarnings('ignore')



def process_image_from_array(img_array, model_dir='./models', output_dir='../result_images',
                             task='all', target_color_temp=None, max_size=656, save=True, show=True,
                             device='cuda', image_name='processed_image'):
    if device.lower() == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if target_color_temp:
        assert 2850 <= target_color_temp <= 7500, 'Color temperature should be in the range [2850 - 7500].'
        if task.lower() != 'editing':
            raise Exception('The task should be "editing" when a target color temperature is specified.')

    logging.info(f'Using device {device}')

    if save and not os.path.exists(output_dir):
        os.mkdir(output_dir)

    img = Image.fromarray((img_array).astype('uint8'), 'RGB')
    


    if task.lower() == 'all':
        net_awb, net_t, net_s = load_models(task, model_dir, device)
    elif task.lower() == 'awb':
        net_awb = load_single_task_model('net_awb.pth', model_dir, device)
    elif task.lower() == 'editing':
        net_t, net_s = load_models(task, model_dir, device)
    else:
        raise Exception("Wrong task! Task should be: 'AWB', 'editing', or 'all'")

    logging.info("Processing image ...")
    outputs = perform_deep_wb(img, task, net_awb, net_t, net_s, device, max_size, target_color_temp)
        
    return outputs

def load_models(task, model_dir, device):
    # Load models based on the task
    if os.path.exists(os.path.join(model_dir, 'net.pth')):
        net = deep_wb_model.deepWBNet()
        logging.info("Loading model {}".format(os.path.join(model_dir, 'net.pth')))
        net.load_state_dict(torch.load(os.path.join(model_dir, 'net.pth'), map_location=device))
        net = net.to(device)  # 将模型移动到指定设备
        return splitter.splitNetworks(net)
    elif task.lower() == 'all' or task.lower() == 'editing':
        return load_individual_models(model_dir, device)
    else:
        raise Exception('Model not found!')

def load_individual_models(model_dir, device):
    net_awb = load_single_task_model('net_awb.pth', model_dir, device)
    net_t = load_single_task_model('net_t.pth', model_dir, device) 
    net_s = load_single_task_model('net_s.pth', model_dir, device)
    return net_awb, net_t, net_s

def load_single_task_model(filename, model_dir, device):
    # Load a single model for the AWB task
    net = deep_wb_single_task.deepWBnet()
    model_path = os.path.join(model_dir, filename)
    logging.info("Loading model {}".format(model_path))
    net.load_state_dict(torch.load(model_path, map_location=device))
    net = net.to(device)  # 将模型移动到指定设备
    net.eval()
    return net

def perform_deep_wb(img, task, net_awb, net_t, net_s, device, S, target_color_temp):
    # Perform the white balance corrections
    
    # 将img从Image转换为Tensor
    # img = transforms.ToTensor()(img).unsqueeze(0).to(device)
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    
    if task.lower() == 'all':
        return deep_wb(img, task=task.lower(), net_awb=net_awb, net_s=net_s, net_t=net_t, device=device, s=S)
    elif task.lower() == 'awb':
        return deep_wb(img, task=task.lower(), net_awb=net_awb, device=device, s=S)
    elif task.lower() == 'editing':
        return deep_wb(img, task=task.lower(), net_s=net_s, net_t=net_t, device=device, s=S)

def scale(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def post_process_enhanced(img):
    """
    优化的后处理方案，针对性提升各项指标
    
    Args:
        img: 输入图像 (范围[0,1]的float32类型)
    Returns:
        处理后的图像
    """
    # 保存原始格式
    original_type = img.dtype
    
    def adaptive_tv_denoise(image, weight_factor=0.1):
        """自适应TV去噪"""
        sigma = estimate_sigma(image, channel_axis=-1, average_sigmas=True)
        weight = sigma * weight_factor
        return denoise_tv_chambolle(image, weight=weight, channel_axis=-1)
    
    def enhance_local_contrast(img, clip_limit=0.01, kernel_size=8):
        """增强局部对比度"""
        # 转换到LAB空间
        lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        l_channel = lab[:,:,0].astype(np.float32)
        
        # 应用CLAHE到L通道
        clahe = cv2.createCLAHE(clipLimit=clip_limit*100, tileGridSize=(kernel_size,kernel_size))
        l_channel = clahe.apply(l_channel.astype(np.uint8))
        
        # 更新L通道
        lab[:,:,0] = l_channel
        
        # 转换回RGB
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return result.astype(np.float32) / 255.0
    
    def color_balance(img, saturation_factor=1.2):
        """色彩平衡增强"""
        # 转换到HSV空间
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(np.float32)
        
        # 适度增加饱和度
        hsv[:,:,1] = np.clip(hsv[:,:,1] * saturation_factor, 0, 255)
        
        # 转换回RGB
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return result.astype(np.float32) / 255.0
    
    def sharpen_details(img, sigma=0.5, alpha=0.5):
        """细节锐化"""
        # 使用高斯差分进行锐化
        blurred = gaussian(img, sigma=sigma, channel_axis=-1)
        detail = img - blurred
        return np.clip(img + alpha * detail, 0, 1)
    
    try:
        # 1. 自适应去噪（提升NIQE）
        img_denoised = adaptive_tv_denoise(img, weight_factor=0.08)
        
        # 2. 局部对比度增强（提升UIQM和UCIQE）
        img_contrast = enhance_local_contrast(
            img_denoised,
            clip_limit=0.01,
            kernel_size=8
        )
        
        # 3. 色彩平衡（提升UICM）
        img_color = color_balance(img_contrast, saturation_factor=1.15)
        
        # 4. 细节增强（提升UISM）
        img_sharp = sharpen_details(img_color, sigma=0.5, alpha=0.4)
        
        # 5. 最终的范围裁剪
        result = np.clip(img_sharp, 0, 1)
        
        # 计算质量得分，根据得分决定是否采用处理结果
        if calculate_quality_score(result) > calculate_quality_score(img):
            return result.astype(original_type)
        return img
    
    except Exception as e:
        print(f"Post-processing failed: {str(e)}")
        return img

def calculate_quality_score(img):
    """
    计算图像质量得分
    结合多个指标的加权平均
    """
    try:
        # 确保图像在0-255范围内
        img_uint8 = (img * 255).astype(np.uint8)
        
        # 转换到LAB空间计算色彩统计特征
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        l, a, b = lab[:,:,0], lab[:,:,1], lab[:,:,2]
        
        # 计算对比度（UCIQE相关）
        contrast = np.std(l)
        
        # 计算饱和度（UICM相关）
        saturation = np.sqrt(np.mean(a**2 + b**2))
        
        # 计算清晰度（UISM相关）
        sharpness = np.mean(np.abs(ndimage.laplace(l)))
        
        # 计算自然度（NIQE相关）
        naturalness = -np.mean(np.abs(filters.sobel(l)))
        
        # 加权平均
        weights = {
            'contrast': 0.3,
            'saturation': 0.2,
            'sharpness': 0.3,
            'naturalness': 0.2
        }
        
        # 归一化各个指标
        contrast_norm = normalize_single(contrast)
        saturation_norm = normalize_single(saturation)
        sharpness_norm = normalize_single(sharpness)
        naturalness_norm = normalize_single(naturalness)
        
        score = (
            weights['contrast'] * contrast_norm +
            weights['saturation'] * saturation_norm +
            weights['sharpness'] * sharpness_norm +
            weights['naturalness'] * naturalness_norm
        )
        
        return score
    
    except Exception as e:
        print(f"Quality score calculation failed: {str(e)}")
        return 0

def normalize_single(value):
    """
    归一化单个值到[0,1]范围
    使用sigmoid函数进行软归一化
    """
    return 1 / (1 + np.exp(-value/100))

# 使用示例
def process_image(img):
    """
    完整的图像处理流程
    """
    # 确保输入图像在[0,1]范围内
    if img.max() > 1.0:
        img = img.astype(np.float32) / 255.0
    
    # 应用后处理
    processed = post_process_enhanced(img)
    
    return processed

def run(image_path, out, dep, removD, args, log_file):

    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    raw_image = Image.open(image_path).convert('RGB')

    raw_image = np.array(raw_image) #/ 255.0
    raw_image, _, _ = process_image_from_array(raw_image)
    raw_image = scale(raw_image)

    
    img = raw_image
    img_depth = (img * 255).astype(np.uint8)
    with torch.no_grad():
        depth = depth_anything.infer_image(img_depth, 518)


    depth = torch.from_numpy(depth)    

    depth = (depth - depth.min()) / (depth.max() - depth.min()) #* 255.0
    depth = 0.05 + 0.95*depth
    depth = depth.cpu().numpy().astype(np.float32)

    grayscale_depth = (depth * 255).astype(np.uint8)


    print("Processed image", flush=True)



    # 使用最优参数进行图像恢复
    recovered = run_pipeline(img, grayscale_depth, args)  # 需要定义 run_pipeline 函数
    recovered = process_image(recovered)

    
    output_image = Image.fromarray((np.round(recovered * 255.0)).astype(np.uint8))

    output_image.save(out, format='png')
    depth_image = Image.fromarray(grayscale_depth)
    depth_image.save(dep, format='png')



def process_file(file, input_path, output_path, depth_mipr_path, removal_path, log_path, args):
    input_file = os.path.join(input_path, file)
    output_file = os.path.join(output_path, file)
    depth_mipr_file = os.path.join(depth_mipr_path, os.path.splitext(file)[0] + '.png')
    removal_file = os.path.join(removal_path, file)
    log_file = os.path.join(log_path, os.path.splitext(file)[0] + '_log.csv')

    if os.path.isfile(input_file):
        print(f'Processing file: {file}')
        
        # 记录每张图片处理的开始时间
        img_start_time = time.time()
        
        run(input_file, output_file, depth_mipr_file, removal_file, args, log_file)
        
        # 记录每张图片处理的结束时间
        img_end_time = time.time()
        img_elapsed_time = img_end_time - img_start_time
        
        # 将每张图片的处理时间写入日志文件
        with open(log_file, 'a') as f:
            f.write(f"# Image processing time: {img_elapsed_time:.4f} seconds\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output.png', help='输出文件名')
    parser.add_argument('--f', type=float, default=3.0, help='控制亮度的 f 值')
    parser.add_argument('--l', type=float, default=0.5, help='控制衰减常数平衡的 l 值')
    parser.add_argument('--p', type=float, default=0.1, help='控制光照图局部性的 p 值')
    parser.add_argument('--min-depth', type=float, default=0.0, help='用于估计的最小深度值（范围 0-1）')
    parser.add_argument('--max-depth', type=float, default=1.0, help='无效深度的替换深度百分位值（范围 0-1）')
    parser.add_argument('--spread-data-fraction', type=float, default=0.05, help='在衰减估计中要求数据与深度范围的此分数相差')
    parser.add_argument('--size', type=int, default=1280, help='输出尺寸')
    parser.add_argument('--monodepth-add-depth', type=float, default=2.0, help='单深度图的附加值')
    parser.add_argument('--monodepth-multiply-depth', type=float, default=10.0, help='单深度图的乘法值')
    parser.add_argument('--model-name', type=str, default="mono_1024x320", help='monodepth 模型名称')
    parser.add_argument('--raw', action='store_true', help='RAW 图像')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl'])
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    folder = "./processed_images_OceanDark_depthanythingv2_improved_seathru/"
    input_path = "/public/home/shaojx8/UWdataset/OceanDark"
    output_path = os.path.join(folder, timestamp, "OutputImages")
    depth_mipr_path = os.path.join(folder, timestamp, "depthimage_mipr")
    removal_path = os.path.join(folder, timestamp, "removal-D")
    log_path = os.path.join(folder, timestamp, "logs")

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(depth_mipr_path, exist_ok=True)
    os.makedirs(removal_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    files = os.listdir(input_path)
    files = natsort.natsorted(files)

    # 记录总处理的开始时间
    total_start_time = time.time()

    for file in tqdm(files, desc="Processing images"):
        process_file(file, input_path, output_path, depth_mipr_path, removal_path, log_path, args)

    # 记录总处理的结束时间
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time

    print(f'Total processing time for {len(files)} images: {total_elapsed_time:.2f} seconds.')
