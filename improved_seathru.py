import collections
import sys
import cv2
import argparse
import numpy as np
import sklearn as sk
import scipy as sp
import scipy.optimize
import scipy.stats
import math
from PIL import Image
import rawpy
from cv2 import medianBlur
from skimage import exposure, restoration
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle, estimate_sigma
from skimage.morphology import closing, opening, erosion, dilation, disk, diamond, square
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from UCIQE import getUCIQE
import numpy as np
import cv2
from skimage import restoration

import numpy as np
import scipy as sp
import cv2
from scipy.optimize import differential_evolution, curve_fit
from skopt import gp_minimize
from skopt.space import Real
#from pyswarm import pso


def improved_backscatter_model(depth, B_inf, beta_B, gamma):
    return B_inf * (1 - np.exp(-beta_B * depth**gamma))

def find_backscatter_estimation_points(img, depths, num_bins=10, fraction=0.01, max_vals=20, min_depth_percent=0.0):
    z_max, z_min = np.max(depths), np.min(depths)
    min_depth = z_min + (min_depth_percent * (z_max - z_min))
    z_ranges = np.linspace(z_min, z_max, num_bins + 1)
    img_norms = np.mean(img, axis=2) if img.ndim == 3 else img
    points = []
    for i in range(len(z_ranges) - 1):
        a, b = z_ranges[i], z_ranges[i+1]
        locs = np.where(np.logical_and(depths > min_depth, np.logical_and(depths >= a, depths <= b)))
        norms_in_range, px_in_range, depths_in_range = img_norms[locs], img[locs], depths[locs]
        arr = sorted(zip(norms_in_range, px_in_range, depths_in_range), key=lambda x: x[0])
        points.extend([(z, p) for n, p, z in arr[:min(math.ceil(fraction * len(arr)), max_vals)]])
    return np.array(points)

def find_backscatter_values_improved(B_pts, depths, wavelength, restarts=10):
    B_vals, B_depths = B_pts[:, 1], B_pts[:, 0]
    
    def loss(B_inf, beta_B, gamma):
        return np.mean(np.abs(B_vals - improved_backscatter_model(B_depths, B_inf, beta_B, gamma)))
    
    bounds_lower = [0, 0, 0.5]
    bounds_upper = [1, 5, 2]
    
    best_loss = np.inf
    best_params = None
    
    for _ in range(restarts):
        try:
            # 修改初始参数生成方式
            p0 = bounds_lower + np.random.random(3) * (np.array(bounds_upper) - np.array(bounds_lower))
            params, _ = sp.optimize.curve_fit(
                improved_backscatter_model,
                B_depths,
                B_vals,
                p0=p0,
                bounds=(bounds_lower, bounds_upper)
            )
            current_loss = loss(*params)
            if current_loss < best_loss:
                best_loss = current_loss
                best_params = params
        except RuntimeError as re:
            print(f"Optimization failed: {re}", file=sys.stderr)
        except ValueError as ve:
            print(f"Value error during optimization: {ve}", file=sys.stderr)
    
    if best_params is None:
        print("Warning: Could not find accurate backscatter model. Using linear approximation.", flush=True)
        slope, intercept, _, _, _ = sp.stats.linregress(B_depths, B_vals)
        return lambda d: slope * d + intercept, np.array([slope, intercept, 1])  # 添加一个默认的 gamma 值
    
    return lambda d: improved_backscatter_model(d, *best_params), best_params


import numpy as np
import cv2

def henvey_greenstein_phase_function(theta, g):
    return (1 - g**2) / (1 + g**2 - 2 * g * np.cos(theta))**1.5


def monte_carlo_scattering(D, depths, scattering_coeff):
    """
    蒙特卡洛散射模拟函数
    
    参数:
        D: 直接信号（去除背散射后的图像）
        depths: 深度图
        scattering_coeff: 散射系数
    
    返回:
        scattered_light: 散射光估计
    """
    # 确保depths是浮点类型，避免uint8溢出问题
    depths_float = depths.astype(np.float32)
    
    # 定义亨尼-格林斯坦相函数参数
    g = 0.8  # 不对称因子，水下环境通常取 0.8
    energy_factor = 0.01  # 控制散射核的总能量
    
    # 调整吸收系数，避免过大的值导致数值问题
    absorption_coeff = 0.5  # 从1.0降低到0.5，更合理的水体吸收系数

    # 核的尺寸
    kernel_size = int(scattering_coeff * 10)
    kernel_size = max(kernel_size, 3)  # 确保最小尺寸
    if kernel_size % 2 == 0:
        kernel_size += 1  # 确保为奇数

    # 生成二维坐标网格
    center = kernel_size // 2
    x = np.arange(kernel_size) - center
    y = np.arange(kernel_size) - center
    xv, yv = np.meshgrid(x, y)
    r = np.sqrt(xv**2 + yv**2) + 1e-6  # 防止除零

    # 计算对应的散射角（小角度近似）
    # 使用安全的深度均值计算
    depth_mean = np.mean(depths_float)
    if depth_mean < 1e-6:
        depth_mean = 1.0  # 防止除零
    
    theta = np.arctan(r / depth_mean)

    # 计算散射核
    phase_function = henvey_greenstein_phase_function(theta, g)
    kernel = phase_function / (np.sum(phase_function) + 1e-10)  # 归一化，添加小值防止除零
    kernel *= energy_factor  # 调整能量

    # 对直接信号应用散射核
    scattered_light = cv2.filter2D(D, -1, kernel)

    # 加入深度依赖的衰减因子
    # 使用安全的指数计算，避免数值溢出
    exponent = -absorption_coeff * depths_float
    # 限制指数范围，避免极端值
    exponent = np.clip(exponent, -10, 0)
    depth_attenuation = np.exp(exponent)
    
    scattered_light *= depth_attenuation

    return scattered_light

'''
def monte_carlo_scattering(D, depths, scattering_coeff):
    # 根据散射系数创建散射核
    depth_weights = np.exp(-depths / np.mean(depths))
    kernel_size = int(scattering_coeff * 10)
    kernel_size = max(kernel_size, 3)  # 确保最小尺寸
    kernel = cv2.getGaussianKernel(kernel_size, scattering_coeff)
    kernel = kernel * kernel.T  # 创建二维高斯核
    
    # 对直接信号应用散射核
    # scattered_light = cv2.filter2D(D, -1, kernel)
    scattered_light = cv2.filter2D(D * depth_weights, -1, kernel)
    return scattered_light
'''

def estimate_illumination_improved(img, B, depths, neighborhood_map, num_neighborhoods, scattering_coeff, p=0.5, f=2.0, max_iters=100, tol=1E-5):
    epsilon = 1e-2
    D = np.maximum(img - B, epsilon)

    # D = np.maximum(img - B, 0)
    avg_cs = D.copy()
    avg_cs_prime = np.copy(avg_cs)
    sizes = np.zeros(num_neighborhoods)
    locs_list = [None] * num_neighborhoods
    
    for label in range(1, num_neighborhoods + 1):
        locs_list[label - 1] = np.where(neighborhood_map == label)
        sizes[label - 1] = np.size(locs_list[label - 1][0])
    
    for _ in range(max_iters):
        for label in range(1, num_neighborhoods + 1):
            locs = locs_list[label - 1]
            size = sizes[label - 1] - 1
            avg_cs_prime[locs] = (np.sum(avg_cs[locs]) - avg_cs[locs]) / size
        
        # 加入多次散射估计
        multi_scatter = monte_carlo_scattering(D, depths, scattering_coeff)
        new_avg_cs = np.maximum((D * p) + (avg_cs_prime * (1 - p)) + multi_scatter, 0)
        
        if np.max(np.abs((avg_cs - new_avg_cs) / (avg_cs + 1e-6))) < tol:
            break
        avg_cs = new_avg_cs
    
    return f * denoise_bilateral(avg_cs)


def radiative_transfer_model(depth, a, b, c):
    return np.exp(-(a + b) * depth) + c * (1 - np.exp(-b * depth))




def beta_model(depths, a1, b1, a2, b2):
    """
    Double-exponential model for attenuation coefficient.

    Parameters:
    - depths (numpy.ndarray): Depth values.
    - a1 (float): Parameter a1.
    - b1 (float): Parameter b1.
    - a2 (float): Parameter a2.
    - b2 (float): Parameter b2.

    Returns:
    - numpy.ndarray: Computed attenuation coefficients.
    """
    return a1 * np.exp(-b1 * depths) + a2 * np.exp(-b2 * depths)

def load_water_coefficients(water_type, wavelength):
    """
    Load absorption and scattering coefficients based on water type and wavelength.

    Parameters:
    - water_type (str): Type of water ('I', 'II', 'III').
    - wavelength (int): Wavelength in nm.

    Returns:
    - tuple: (a, b) absorption and scattering coefficients.
    """
    water_types = {
        'I': {'a': {400: 0.01, 500: 0.02, 600: 0.05}, 'b': {400: 0.03, 500: 0.02, 600: 0.01}},
        'II': {'a': {400: 0.02, 500: 0.04, 600: 0.07}, 'b': {400: 0.05, 500: 0.04, 600: 0.03}},
        'III': {'a': {400: 0.04, 500: 0.06, 600: 0.1}, 'b': {400: 0.08, 500: 0.07, 600: 0.05}},
    }
    known_wavelengths = [400, 500, 600]
    closest_wavelength = min(known_wavelengths, key=lambda x: abs(x - wavelength))
    water = water_types.get(water_type, water_types['II'])
    a = water['a'][closest_wavelength]
    b = water['b'][closest_wavelength]
    return a, b

def estimate_wideband_attenuation_improved(depths, illum, wavelength, water_type='I', method='differential_evolution', **kwargs):
    """
    Estimate wideband attenuation using a global optimization algorithm.

    Parameters:
    - depths (numpy.ndarray): Depth map.
    - illum (numpy.ndarray): Illumination map.
    - wavelength (int): Wavelength in nm.
    - water_type (str): Type of water ('I', 'II', 'III').
    - method (str): Optimization method ('differential_evolution', 'bayesian', 'pso').
    - kwargs: Additional keyword arguments for the optimization method.

    Returns:
    - beta_smoothed (numpy.ndarray): Smoothed attenuation coefficient map.
    """
    # Load water coefficients
    a, b = load_water_coefficients(water_type, wavelength)
    beta_theoretical = a + b  # Theoretical attenuation coefficient

    # Data preprocessing
    depth_min = 0.1
    depths_safe = np.clip(depths, depth_min, np.max(depths))
    illum_safe = np.clip(illum, 0.01, 1.0)

    # Initial empirical estimation
    with np.errstate(divide='ignore'):
        beta_empirical = -np.log(illum_safe) / depths_safe
    beta_empirical = np.clip(beta_empirical, 0, 5.0)

    # Data filtering
    valid_mask = (illum > 0.01) & (depths > depth_min)
    depths_valid = depths_safe[valid_mask]
    beta_valid = beta_empirical[valid_mask]

    if len(depths_valid) < 3:
        print('Warning: Too few points for curve fitting. Using theoretical beta.', flush=True)
        beta_final = beta_theoretical * np.ones_like(depths_safe)
        return beta_final

    # Define the loss function
    def loss_func(params):
        beta_estimated = beta_model(depths_valid, *params)
        return np.mean((beta_valid - beta_estimated) ** 2)

    # Optimization based on selected method
    if method == 'differential_evolution':
        bounds = [(0, 10), (0, 1), (0, 10), (0, 1)]
        result = differential_evolution(loss_func, bounds=bounds, strategy='best1bin', **kwargs)
        if result.success:
            best_params = result.x
            print(f"Optimization succeeded with loss {result.fun}", flush=True)
        else:
            print('Optimization failed. Using theoretical beta.', flush=True)
            best_params = [0, 0, 0, 0]  # Default parameters
    elif method == 'bayesian':
        space = [Real(0, 10, name='a1'), Real(0, 1, name='b1'), Real(0, 10, name='a2'), Real(0, 1, name='b2')]
        result = gp_minimize(loss_func, space, **kwargs)
        if result.fun is not None:
            best_params = result.x
            print(f"Optimization succeeded with loss {result.fun}", flush=True)
        else:
            print('Optimization failed. Using theoretical beta.', flush=True)
            best_params = [0, 0, 0, 0]
    #elif method == 'pso':
    #    lb = [0, 0, 0, 0]
    #    ub = [10, 1, 10, 1]
    #    best_params, best_loss = pso(loss_func, lb, ub, **kwargs)
    #    print(f"Optimization {'succeeded' if best_params is not None else 'failed'} with loss {best_loss}", flush=True)
    else:
        raise ValueError("Unsupported optimization method.")

    # Apply the model with best parameters
    if method in ['differential_evolution', 'bayesian', 'pso']:
        beta_final = beta_model(depths_safe, *best_params)
    else:
        beta_final = beta_theoretical * np.ones_like(depths_safe)
    
    beta_final = np.clip(beta_final, 0, None)

    # Filtering and smoothing
    beta_smoothed = cv2.medianBlur(beta_final.astype(np.float32), 5)
    beta_smoothed = cv2.bilateralFilter(beta_smoothed, 9, 75, 75)
    beta_smoothed = cv2.edgePreservingFilter(beta_smoothed.astype(np.float32), flags=1, sigma_s=60, sigma_r=0.4)

    return beta_smoothed



def post_process(img):
    # 去噪
    denoised = restoration.denoise_wavelet(img, channel_axis=-1, convert2ycbcr=True, method='BayesShrink')
    
    # 自适应对比度增强
    p2, p98 = np.percentile(denoised, (2, 98))
    enhanced = exposure.rescale_intensity(denoised, in_range=(p2, p98))
    
    # 温和的色彩平衡
    #balanced = color_balance(enhanced, strength=0.5)
    
    return enhanced

import cv2
import numpy as np
from skimage import exposure
from skimage.color import rgb2lab, lab2rgb
import warnings
warnings.filterwarnings('ignore')

class EnhancementProcessor:
    """
    Underwater Image Enhancement Processor
    实现自适应白平衡和图像增强的集成处理
    """
    
    def __init__(self, 
                 clahe_clip_limit=1.5,
                 clahe_grid_size=(8, 8),
                 gamma=1.15,
                 alpha=0.7,
                 beta=0.6,
                 white_balance_method='adaptive'):
        """
        初始化增强处理器
        
        Args:
            clahe_clip_limit: CLAHE对比度限制
            clahe_grid_size: CLAHE网格大小
            gamma: gamma校正值
            alpha: CLAHE结果权重
            beta: gamma校正结果权重
            white_balance_method: 白平衡方法 ['gray_world', 'retinex', 'shades_of_gray', 'adaptive']
        """
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_grid_size
        )
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.white_balance_method = white_balance_method
    
    def _normalize_image(self, img):
        """
        规范化图像格式，确保数据类型一致性
        """
        # 处理数据类型
        if img.dtype == np.float64:
            img = img.astype(np.float32)
        
        # 处理值范围
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        
        return img
    
    def _restore_format(self, img, original):
        """
        还原到原始图像格式，确保数据类型一致性
        """
        if original.dtype in [np.float32, np.float64] and original.max() <= 1.0:
            img = img.astype(np.float32) / 255.0
            if original.dtype == np.float64:
                img = img.astype(np.float64)
        return img
    
    def apply_gray_world(self, img):
        """
        应用灰度世界假设的白平衡
        确保所有通道数据类型一致
        """
        img = self._normalize_image(img)  # 确保输入是uint8类型
        
        # 转换到LAB空间
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(img_lab)
        
        # 确保所有通道都是相同的数据类型
        l = l.astype(np.float32)
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        
        # 计算均值
        l_mean = np.mean(l)
        a_mean = np.mean(a)
        b_mean = np.mean(b)
        
        # 调整a和b通道
        a = a - ((a_mean - 128) * (l_mean / 128.0))
        b = b - ((b_mean - 128) * (l_mean / 128.0))
        
        # 确保值在有效范围内
        a = np.clip(a, 0, 255).astype(np.uint8)
        b = np.clip(b, 0, 255).astype(np.uint8)
        l = np.clip(l, 0, 255).astype(np.uint8)
        
        # 合并通道并转换回RGB
        balanced_lab = cv2.merge([l, a, b])
        balanced_rgb = cv2.cvtColor(balanced_lab, cv2.COLOR_LAB2RGB)
        
        return balanced_rgb
    
    def apply_retinex(self, img):
        """
        应用Retinex理论的自适应白平衡
        确保数据类型一致性
        """
        img = self._normalize_image(img)
        
        # 转换为浮点数进行计算
        img_float = img.astype(np.float32)
        r, g, b = cv2.split(img_float)
        
        # 计算对数域
        r_log = np.log1p(r)
        g_log = np.log1p(g)
        b_log = np.log1p(b)
        
        # 计算均值和增益
        r_mean = np.mean(r_log)
        g_mean = np.mean(g_log)
        b_mean = np.mean(b_log)
        avg_mean = (r_mean + g_mean + b_mean) / 3
        
        # 计算增益系数
        r_gain = avg_mean / (r_mean + np.finfo(float).eps)
        g_gain = avg_mean / (g_mean + np.finfo(float).eps)
        b_gain = avg_mean / (b_mean + np.finfo(float).eps)
        
        # 应用增益
        r_balanced = np.expm1(r_log * r_gain)
        g_balanced = np.expm1(g_log * g_gain)
        b_balanced = np.expm1(b_log * b_gain)
        
        # 合并通道前确保数据类型一致
        r_balanced = np.clip(r_balanced, 0, 255).astype(np.uint8)
        g_balanced = np.clip(g_balanced, 0, 255).astype(np.uint8)
        b_balanced = np.clip(b_balanced, 0, 255).astype(np.uint8)
        
        return cv2.merge([r_balanced, g_balanced, b_balanced])
    
    def apply_shades_of_gray(self, img, minkowski_norm=6):
        """
        应用灰度阴影的白平衡
        确保数据类型一致性
        """
        img = self._normalize_image(img)
        
        # 转换为浮点数进行计算
        img_float = img.astype(np.float32) / 255.0
        r, g, b = cv2.split(img_float)
        
        # 计算Minkowski范数
        r_norm = np.power(np.mean(np.power(r, minkowski_norm) + np.finfo(float).eps), 1/minkowski_norm)
        g_norm = np.power(np.mean(np.power(g, minkowski_norm) + np.finfo(float).eps), 1/minkowski_norm)
        b_norm = np.power(np.mean(np.power(b, minkowski_norm) + np.finfo(float).eps), 1/minkowski_norm)
        
        # 计算增益系数
        gains = np.sqrt(1/(3 * np.array([r_norm, g_norm, b_norm]) + np.finfo(float).eps))
        gains = gains / (np.max(gains) + np.finfo(float).eps)
        
        # 应用增益并确保数据类型一致
        r_balanced = np.clip(r * gains[0] * 255, 0, 255).astype(np.uint8)
        g_balanced = np.clip(g * gains[1] * 255, 0, 255).astype(np.uint8)
        b_balanced = np.clip(b * gains[2] * 255, 0, 255).astype(np.uint8)
        
        return cv2.merge([r_balanced, g_balanced, b_balanced])
    
    def evaluate_color_cast(self, img):
        """
        评估图像的色偏程度
        返回评分（越高越好）
        """
        img = self._normalize_image(img)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # 计算色彩统计特征
        a_std = np.std(a)
        b_std = np.std(b)
        a_deviation = abs(np.mean(a) - 128)
        b_deviation = abs(np.mean(b) - 128)
        
        # 计算综合得分
        score = (a_std + b_std) / (a_deviation + b_deviation + 1e-6)
        return score
    
    def apply_white_balance(self, img):
        """
        应用自适应白平衡
        """
        try:
            img = self._normalize_image(img)
            
            if self.white_balance_method == 'gray_world':
                return self.apply_gray_world(img)
            elif self.white_balance_method == 'retinex':
                return self.apply_retinex(img)
            elif self.white_balance_method == 'shades_of_gray':
                return self.apply_shades_of_gray(img)
            else:
                # 自适应选择最佳白平衡方法
                methods = {
                    'gray_world': self.apply_gray_world,
                    'retinex': self.apply_retinex,
                    'shades_of_gray': self.apply_shades_of_gray
                }
                
                results = {}
                scores = {}
                
                for name, method in methods.items():
                    try:
                        balanced = method(img)
                        score = self.evaluate_color_cast(balanced)
                        results[name] = balanced
                        scores[name] = score
                    except Exception as e:
                        print(f"Method {name} failed: {str(e)}")
                        continue
                
                if not scores:
                    print("All white balance methods failed, returning original image")
                    return img
                
                best_method = max(scores.items(), key=lambda x: x[1])[0]
                return results[best_method]
                
        except Exception as e:
            print(f"White balance failed: {str(e)}")
            return img  # 如果处理失败，返回原始图像
    
    def apply_clahe(self, img):
        """
        应用CLAHE增强
        使用LAB色彩空间保持色彩平衡
        """
        img = self._normalize_image(img)
        
        # 转换到LAB色彩空间
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # 应用CLAHE到L通道
        l_clahe = self.clahe.apply(l)
        
        # 合并通道并转换回RGB
        enhanced_lab = cv2.merge([l_clahe, a, b])
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        return enhanced_rgb
    
    def apply_adaptive_gamma(self, img):
        """
        应用自适应gamma校正
        使用局部亮度信息调整gamma值
        """
        img = self._normalize_image(img)
        
        # 转换到HSV空间
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        v = hsv[:,:,2]
        
        # 计算局部亮度
        local_mean = cv2.GaussianBlur(v, (15,15), 0)
        
        # 创建自适应gamma map
        gamma_map = np.ones_like(v, dtype=np.float32) * self.gamma
        gamma_map[local_mean > 128] = 1.0 + (self.gamma - 1.0) * 0.5
        
        # 应用gamma校正
        corrected = np.power(v / 255.0, 1.0 / gamma_map) * 255.0
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        
        # 更新V通道
        hsv[:,:,2] = corrected
        
        # 转换回RGB
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    def enhance(self, image):
        """
        完整的图像增强流程
        """
        try:
            # 保存原始格式供后续还原
            original_format = image.copy()
            
            # 规范化图像格式
            img = self._normalize_image(image)
            
            # 1. 应用自适应白平衡
            balanced = self.apply_white_balance(img)
            
            # 2. 应用CLAHE增强
            clahe_result = self.apply_clahe(balanced)
            
            # 3. 应用自适应gamma校正
            gamma_result = self.apply_adaptive_gamma(balanced)
            
            # 4. 融合结果
            enhanced = cv2.addWeighted(
                clahe_result, self.alpha,
                gamma_result, self.beta,
                0
            )
            
            # 还原到原始格式
            enhanced = self._restore_format(enhanced, original_format)
            
            return enhanced
            
        except Exception as e:
            print(f"Enhancement failed: {str(e)}")
            return image  # 如果处理失败，返回原始图像

def process_image(image, 
                 clahe_clip_limit=1.5,
                 clahe_grid_size=(8, 8),
                 gamma=1.15,
                 alpha=0.7,
                 beta=0.6,
                 white_balance_method='adaptive'):
    """
    便捷的处理函数
    
    Args:
        image: 输入图像
        clahe_clip_limit: CLAHE对比度限制
        clahe_grid_size: CLAHE网格大小
        gamma: gamma校正值
        alpha: CLAHE结果权重
        beta: gamma校正结果权重
        white_balance_method: 白平衡方法
    
    Returns:
        增强后的图像
    """
    processor = EnhancementProcessor(
        clahe_clip_limit=clahe_clip_limit,
        clahe_grid_size=clahe_grid_size,
        gamma=gamma,
        alpha=alpha,
        beta=beta,
        white_balance_method=white_balance_method
    )
    return processor.enhance(image)
# 使用示例
"""
import cv2
import numpy as np

# 读取图像
image = cv2.imread('underwater_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 方法1：使用便捷函数
enhanced = process_image(image)

# 方法2：使用处理器类
processor = EnhancementProcessor()
enhanced = processor.enhance(image)

# 保存结果
enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
cv2.imwrite('enhanced_image.jpg', enhanced_bgr)
"""

def color_balance(img, strength=0.5):
    img = img.astype(np.float32)
    r, g, b = cv2.split(img)
    r_avg, g_avg, b_avg = np.mean(r), np.mean(g), np.mean(b)
    avg = (r_avg + g_avg + b_avg) / 3
    
    r = np.clip(r * (1 - strength + strength * (avg / r_avg)), 0, 1)
    g = np.clip(g * (1 - strength + strength * (avg / g_avg)), 0, 1)
    b = np.clip(b * (1 - strength + strength * (avg / b_avg)), 0, 1)
    
    return cv2.merge((r, g, b))


def adaptive_depth_compensation(depths):
    depth_mean = np.mean(depths)
    depth_std = np.std(depths)
    return 1 / (1 + np.exp(-(depths - depth_mean) / (depth_std + 1e-6)))

def recover_image_improved(img, depths, B, beta_D, water_quality_map):
    eps = 1e-8
    depth_factor = adaptive_depth_compensation(depths)
    res = (img - B) * np.exp(beta_D * np.expand_dims(depths * depth_factor, axis=2))
    res = res / (water_quality_map[:, :, np.newaxis] + eps)
    res = np.clip(res, 0, 1)
    return post_process(res)

    # res = (img - B) * np.exp(beta_D * np.expand_dims(depths, axis=2))
def recover_image(img, depths, B, beta_D, water_quality_map):
    # 限制指数的最大值，防止数值溢出
    max_exponent = 10  # 根据实际情况调整
    exponent = beta_D * np.expand_dims(depths, axis=2)
    exponent = np.clip(exponent, 0, max_exponent)
    
    transmission = np.exp(-exponent)
    direct_signal = (img - B) / (transmission + 1e-6)
    
    # 加入自适应对比度增强
    #p2, p98 = np.percentile(direct_signal, (2, 98))
    #direct_signal_enhanced = exposure.rescale_intensity(direct_signal, in_range=(p2, p98))
    #water_quality_compensation = np.expand_dims(water_quality_map, axis=2)
    #compensated_signal = direct_signal #* water_quality_compensation
    # balanced = color_balance(compensated_signal)
    # compensated_signal = post_process(compensated_signal)
    direct_signal = np.clip(direct_signal, 0, 1)
    return process_image(direct_signal)


def evaluate_image_quality(original, recovered):
    uciqe_score = getUCIQE(recovered)
    ssim_score = ssim(original, recovered, multichannel=True)
    psnr_score = psnr(original, recovered)
    return {
        'UCIQE': uciqe_score,
        'SSIM': ssim_score,
        'PSNR': psnr_score
    }

def construct_neighborhood_map(depths, epsilon=0.05):
    eps = (np.max(depths) - np.min(depths)) * epsilon
    nmap = np.zeros_like(depths).astype(np.int32)
    n_neighborhoods = 1
    while np.any(nmap == 0):
        locs_x, locs_y = np.where(nmap == 0)
        start_index = np.random.randint(0, len(locs_x))
        start_x, start_y = locs_x[start_index], locs_y[start_index]
        q = collections.deque()
        q.append((start_x, start_y))
        while not len(q) == 0:
            x, y = q.pop()
            if np.abs(depths[x, y] - depths[start_x, start_y]) <= eps:
                nmap[x, y] = n_neighborhoods
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    x2, y2 = x + dx, y + dy
                    if 0 <= x2 < depths.shape[0] and 0 <= y2 < depths.shape[1] and nmap[x2, y2] == 0:
                        q.append((x2, y2))
        n_neighborhoods += 1
    zeros_size_arr = sorted(zip(*np.unique(nmap[depths == 0], return_counts=True)), key=lambda x: x[1], reverse=True)
    if len(zeros_size_arr) > 0:
        nmap[nmap == zeros_size_arr[0][0]] = 0 #reset largest background to 0
    return nmap, n_neighborhoods - 1

def find_closest_label(nmap, start_x, start_y):
    mask = np.zeros_like(nmap).astype(np.bool_)
    q = collections.deque()
    q.append((start_x, start_y))
    while not len(q) == 0:
        x, y = q.pop()
        if 0 <= x < nmap.shape[0] and 0 <= y < nmap.shape[1]:
            if nmap[x, y] != 0:
                return nmap[x, y]
            mask[x, y] = True
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                x2, y2 = x + dx, y + dy
                if 0 <= x2 < nmap.shape[0] and 0 <= y2 < nmap.shape[1] and not mask[x2, y2]:
                    q.append((x2, y2))
    return 0

def refine_neighborhood_map(nmap, min_size = 10, radius = 3):
    refined_nmap = np.zeros_like(nmap)
    vals, counts = np.unique(nmap, return_counts=True)
    neighborhood_sizes = sorted(zip(vals, counts), key=lambda x: x[1], reverse=True)
    num_labels = 1
    for label, size in neighborhood_sizes:
        if size >= min_size and label != 0:
            refined_nmap[nmap == label] = num_labels
            num_labels += 1
    for label, size in neighborhood_sizes:
        if size < min_size and label != 0:
            for x, y in zip(*np.where(nmap == label)):
                refined_nmap[x, y] = find_closest_label(refined_nmap, x, y)
    refined_nmap = closing(refined_nmap, square(radius))
    return refined_nmap, num_labels - 1

def load_image_and_depth_map(img_fname, depths_fname, size_limit = 1024):
    depths = Image.open(depths_fname)
    img = Image.fromarray(rawpy.imread(img_fname).postprocess())
    img.thumbnail((size_limit, size_limit), Image.ANTIALIAS)
    depths = depths.resize(img.size, Image.ANTIALIAS)
    return np.float32(img) / 255.0, np.array(depths)

#def estimate_scattering_coefficient(img, depths):
    # 简单估计散射系数
#    return np.mean(img) / np.mean(depths)
import scipy as sp
import scipy.optimize as opt
import scipy.stats as stats


def estimate_scattering_coefficient(img_channel, depths, wavelength, water_type='I'):
    # 根据水体类型和波长获取吸收系数和初始散射系数
    a_coeff = get_absorption_coefficient(water_type, wavelength)
    b_initial = get_scattering_coefficient_from_water_type(water_type, wavelength)
    
    # 估计 B_inf 和 beta_B
    depth_threshold = np.percentile(depths, 90)
    B_inf = estimate_B_inf(img_channel, depths, depth_threshold)
    beta_B = b_initial  # 近似取 beta_B = b_initial

    # 将图像和深度展开为一维数组
    I = img_channel.flatten()
    z = depths.flatten()

    # 构建模型函数
    def model_function(z, b):
        return np.exp(-(a_coeff + b) * z) + B_inf * (1 - np.exp(-beta_B * z))

    # 使用非线性最小二乘拟合，初始猜测为 b_initial
    popt, _ = opt.curve_fit(model_function, z, I, p0=[b_initial], bounds=(0, np.inf))

    scattering_coefficient = popt[0]
    return scattering_coefficient

def get_absorption_coefficient(water_type, wavelength):
    # 定义 I 型水体的吸收系数
    absorption_coefficients = {
        450: 0.015,  # 蓝光
        550: 0.035,  # 绿光
        650: 0.065   # 红光
    }
    return absorption_coefficients.get(wavelength, 0.035)  # 默认取绿光

def get_scattering_coefficient_from_water_type(water_type, wavelength):
    # 定义 I 型水体的散射系数
    scattering_coefficients = {
        450: 0.032,  # 蓝光
        550: 0.020,  # 绿光
        650: 0.010   # 红光
    }
    return scattering_coefficients.get(wavelength, 0.020)  # 默认取绿光

def estimate_B_inf(img_channel, depths, depth_threshold):
    # 选择深度大于阈值的区域
    deep_region_mask = depths > depth_threshold
    # 防止空区域
    if np.sum(deep_region_mask) == 0:
        deep_region_mask = depths > (depth_threshold * 0.9)
    # 计算深度较大区域的平均像素值
    B_inf = np.mean(img_channel[deep_region_mask])
    return B_inf





def estimate_water_quality_map(img, depths):
    brightness = np.mean(img, axis=2)
    normalized_depth = (depths - np.min(depths)) / (np.max(depths) - np.min(depths))
    turbidity = 1 - brightness  # 假设亮度越低，浑浊度越高
    absorption = 0.1 + 0.2 * normalized_depth
    scattering = 0.05 + 0.1 * turbidity
    water_quality = 1 / (absorption + scattering + 1e-6)
    return water_quality

def adaptive_backscatter_model(depth, B_inf, beta_B, gamma, local_std, alpha=0.5):
    return B_inf * (1 - np.exp(-beta_B * (depth**gamma) * (1 + alpha * np.tanh(local_std))))

def compute_local_std(img, kernel_size=5):
    return cv2.GaussianBlur(np.std(img, axis=2), (kernel_size, kernel_size), 0)


def smooth_depth_map(depths, kernel_size=5):
    return cv2.edgePreservingFilter(depths.astype(np.float32), flags=1, sigma_s=kernel_size, sigma_r=0.1)

# 在 run_pipeline 函数中使用


def run_pipeline(img, depths, args):
    wavelengths = {'R': 650, 'G': 550, 'B': 450}
    
    #     # 确保depths是浮点类型，范围在0-1之间
    # if depths.dtype != np.float32 and depths.dtype != np.float64:
    #     print(f'Converting depths from {depths.dtype} to float32', flush=True)
    #     if depths.max() > 1.0:
    #         # 如果depths是0-255范围，归一化到0-1
    #         depths = depths.astype(np.float32) / 255.0
    #     else:
    #         depths = depths.astype(np.float32)
    
    # 确保depths的值在合理范围内
    # depths = np.clip(depths, 0.01, 1.0)  # 避免零深度
    depths = smooth_depth_map(depths)
    print('Estimating backscatter...', flush=True)
    Br, coefs_r = find_backscatter_values_improved(find_backscatter_estimation_points(img[:,:,0], depths), depths, wavelength=620)
    Bg, coefs_g = find_backscatter_values_improved(find_backscatter_estimation_points(img[:,:,1], depths), depths, wavelength=540)
    Bb, coefs_b = find_backscatter_values_improved(find_backscatter_estimation_points(img[:,:,2], depths), depths, wavelength=450)

    print('Constructing neighborhood map...', flush=True)
    nmap, n = construct_neighborhood_map(depths, 0.1)
    nmap, n = refine_neighborhood_map(nmap, 50)

    print('Estimating illumination...', flush=True)
    scattering_coeff_R = estimate_scattering_coefficient(img[:,:,0], depths, 650)
    scattering_coeff_G = estimate_scattering_coefficient(img[:,:,1], depths, 550)
    scattering_coeff_B = estimate_scattering_coefficient(img[:,:,2], depths, 450)
    illR = estimate_illumination_improved(img[:,:,0], Br(depths), depths, nmap, n, scattering_coeff_R, p=args.p, f=args.f)
    illG = estimate_illumination_improved(img[:,:,1], Bg(depths), depths, nmap, n, scattering_coeff_G, p=args.p, f=args.f)
    illB = estimate_illumination_improved(img[:,:,2], Bb(depths), depths, nmap, n, scattering_coeff_B, p=args.p, f=args.f)

    print('Estimating wideband attenuation...', flush=True)
    beta_D_r = estimate_wideband_attenuation_improved(depths, illR, wavelength=620)
    beta_D_g = estimate_wideband_attenuation_improved(depths, illG, wavelength=540)
    beta_D_b = estimate_wideband_attenuation_improved(depths, illB, wavelength=450)

    print('Reconstructing image...', flush=True)
    local_std = compute_local_std(img)
    B = np.stack([
        adaptive_backscatter_model(depths, *coefs_r, local_std),
        adaptive_backscatter_model(depths, *coefs_g, local_std),
        adaptive_backscatter_model(depths, *coefs_b, local_std)
    ], axis=2)

    beta_D = np.stack([beta_D_r, beta_D_g, beta_D_b], axis=2)
    water_quality_map = estimate_water_quality_map(img, depths)
    recovered = recover_image(img, depths, B, beta_D, water_quality_map)
    return recovered

    #B = np.stack([Br(depths), Bg(depths), Bb(depths)], axis=2)

    #quality_metrics = evaluate_image_quality(img, recovered)
    #print(f"Image quality metrics: {quality_metrics}")

#, quality_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run improved SeaThru pipeline')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--depth', type=str, required=True, help='Path to depth map')
    parser.add_argument('--output', type=str, required=True, help='Path to output image')
    parser.add_argument('--p', type=float, default=0.7, help='Illumination estimation parameter')
    parser.add_argument('--f', type=float, default=3.0, help='Illumination estimation parameter')
    parser.add_argument('--min_depth', type=float, default=0.0, help='Minimum depth for backscatter estimation')
    parser.add_argument('--spread_data_fraction', type=float, default=0.01, help='Fraction for data spreading in attenuation estimation')
    parser.add_argument('--l', type=float, default=1.0, help='Attenuation estimation parameter')
    
    args = parser.parse_args()
    
    img, depths = load_image_and_depth_map(args.image, args.depth)
    recovered, metrics = run_pipeline(img, depths, args)
    
    plt.imsave(args.output, recovered)
    print(f"Recovered image saved to {args.output}")
    print(f"Final image quality metrics: {metrics}")