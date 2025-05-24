"""
UCIQE
======================================
Trained coefficients are c1=0.4680, c2=0.2745, c3=0.2576.
UCIQE= c1*var_chr+c2*con_lum+c3*aver_sat
var_chr   is σc : the standard deviation of chroma
con_lum is conl: the contrast of luminance
aver_sat  is μs : the average of saturation
coe_metric=[c1, c2, c3]are weighted coefficients.
---------------------------------------------------------
When you want to use the uciqe function, you must give the values of two parameters,
one is the nargin value you calculated and the location and name format of the image
you want to calculate the uciqe value.The format of the function is UCIQE.uciqe(nargin,loc)
---------------------------------------------------------
The input image must be RGB image
======================================
"""
import os

import cv2
import numpy as np
from natsort import natsort
from PIL import Image
import numpy as np

def getUCIQE(img):
    # 确保输入是一个PIL Image对象
    if isinstance(img, Image.Image):
        img_array = np.array(img)
    else:
        # 如果已经是Numpy数组，直接使用
        img_array = img

    # 归一化并转换为uint8
    img_array = (np.round(img_array * 255.0)).astype(np.uint8)
    # img_array = (np.round(img_array * 255.0)).astype(np.uint8)
    
    # 将图像从RGB转换到LAB
    img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2Lab)
    l, a, b = cv2.split(img_lab.astype('float'))

    # 归一化
    l /= 255
    a = a / 255 - 0.5
    b = b / 255 - 0.5

    # 计算色度（chroma）
    chroma = np.sqrt(a**2 + b**2)
    # 计算饱和度（saturation）
    saturation = chroma / np.sqrt(chroma**2 + l**2)

    # 平均饱和度
    avg_saturation = np.mean(saturation)
    # 色度方差
    var_chroma = np.sqrt(np.mean((chroma - np.mean(chroma))**2))
    # 亮度对比度
    contrast_l = np.max(l) - np.min(l)

    # 定义系数
    c1, c2, c3 = 0.4680, 0.2745, 0.2576

    # 计算UCIQE
    uciqe = c1 * var_chroma + c2 * contrast_l + c3 * avg_saturation
    print('uciqe:', uciqe)
    return uciqe

def getUCIQE_img(img):
    # 读取图像
    #img = cv2.imread(img_path)
    # 将图像从BGR转换到LAB
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(img_lab.astype('float'))

    # 归一化
    l /= 255
    a = a / 255 - 0.5
    b = b / 255 - 0.5

    # 计算色度（chroma）
    chroma = np.sqrt(a**2 + b**2)
    # 计算饱和度（saturation）
    saturation = chroma / np.sqrt(chroma**2 + l**2)

    # 平均饱和度
    avg_saturation = np.mean(saturation)
    # 色度方差
    var_chroma = np.sqrt(np.mean((chroma - np.mean(chroma))**2))
    # 亮度对比度
    contrast_l = np.max(l) - np.min(l)

    # 定义系数
    c1, c2, c3 = 0.4680, 0.2745, 0.2576

    # 计算UCIQE
    uciqe = c1 * var_chroma + c2 * contrast_l + c3 * avg_saturation
    return uciqe

if __name__ == '__main__':

    img = cv2.imread('/public/home/shaojx8/ADPCC_Code-main/OutputImages_ant/temp.png')
    uciqe = getUCIQE_img(img)
    print('uciqe：', uciqe)
    #error_list_uciqe.append(uciqe)
    #print("UCIQE >> Mean: {:.4f} std: {:.4f}".format(np.mean(np.array(error_list_uciqe)),np.std((np.array(error_list_uciqe)))))
