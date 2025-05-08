import cv2
import numpy as np
from scipy.ndimage import convolve

import os

import cv2
import numpy as np
from natsort import natsort
from PIL import Image
import numpy as np

import cv2
import numpy as np

# 定义常量参数
UICM_CB_MEAN_WEIGHT = -0.0268
UICM_CB_STD_WEIGHT = 0.1586
UICM_CR_MEAN_WEIGHT = -0.0268
UICM_CR_STD_WEIGHT = 0.1586

UIQM_UICM_WEIGHT = 0.0282
UIQM_UISM_WEIGHT = 0.2953
UIQM_UICONM_WEIGHT = 3.5753

def calculate_uicm(ycbcr_image):
    cb = ycbcr_image[:, :, 1]
    cr = ycbcr_image[:, :, 2]
    
    cb_mean = np.mean(cb)
    cr_mean = np.mean(cr)
    cb_std = np.std(cb)
    cr_std = np.std(cr)
    
    uicm = UICM_CB_MEAN_WEIGHT * np.abs(cb_mean) + UICM_CB_STD_WEIGHT * cb_std + \
           UICM_CR_MEAN_WEIGHT * np.abs(cr_mean) + UICM_CR_STD_WEIGHT * cr_std
    return uicm

def calculate_uism(gray_image):
    gx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    
    uism = np.mean(magnitude)
    return uism

def calculate_uiconm(ycbcr_image):
    y_channel = ycbcr_image[:, :, 0]
    uiconm = np.std(y_channel)
    return uiconm

def calculate_uiqm(image):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    #image =image/255
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    uicm = calculate_uicm(ycbcr_image)
    uism = calculate_uism(gray_image)
    uiconm = calculate_uiconm(ycbcr_image)
    
    uiqm = UIQM_UICM_WEIGHT * uicm + UIQM_UISM_WEIGHT * uism + UIQM_UICONM_WEIGHT * uiconm
    print("UIQM Value:", uiqm)
    return uiqm

def UICM(img):
    R = img[:, :, 2]
    G = img[:, :, 1]
    B = img[:, :, 0]
    RG = R - G
    YB = (R + G) / 2 - B

    K = img.shape[0] * img.shape[1]

    # R-G channel
    RG1 = np.sort(RG.flatten())
    alpha = 0.1
    RG1 = RG1[int(alpha * K):int((1 - alpha) * K)]
    meanRG = np.mean(RG1)
    deltaRG = np.std(RG1)

    # Y-B channel
    YB1 = np.sort(YB.flatten())
    YB1 = YB1[int(alpha * K):int((1 - alpha) * K)]
    meanYB = np.mean(YB1)
    deltaYB = np.std(YB1)

    uicm = -0.0268 * np.sqrt(meanRG ** 2 + meanYB ** 2) + 0.1586 * np.sqrt(deltaRG ** 2 + deltaYB ** 2)
    return uicm

def UISM(img):
    hx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    hy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    SobelR = np.abs(convolve(img[:, :, 2], hx) + convolve(img[:, :, 2], hy))
    SobelG = np.abs(convolve(img[:, :, 1], hx) + convolve(img[:, :, 1], hy))
    SobelB = np.abs(convolve(img[:, :, 0], hx) + convolve(img[:, :, 0], hy))

    EME = lambda x: np.log(np.max(x) / np.min(x)) if np.min(x) != 0 else 0
    EMER = np.mean([EME(SobelR[i:i+5, j:j+5]) for i in range(0, img.shape[0], 5) for j in range(0, img.shape[1], 5)])
    EMEG = np.mean([EME(SobelG[i:i+5, j:j+5]) for i in range(0, img.shape[0], 5) for j in range(0, img.shape[1], 5)])
    EMEB = np.mean([EME(SobelB[i:i+5, j:j+5]) for i in range(0, img.shape[0], 5) for j in range(0, img.shape[1], 5)])

    lambdaR = 0.299
    lambdaG = 0.587
    lambdaB = 0.114
    uism = lambdaR * EMER + lambdaG * EMEG + lambdaB * EMEB
    return uism

def AME(im):
    max_val = np.max(im)
    min_val = np.min(im)
    if max_val != min_val and (max_val != 0 or min_val != 0):
        # 防止除以零
        denominator = max_val + min_val
        if denominator == 0:
            return 0
        else:
            value = (max_val - min_val) / denominator
            # 使用np.log1p来提高数值稳定性
            return np.log1p(value) * value
    else:
        return 0


def UIConM(img):
    R = img[:, :, 2]
    G = img[:, :, 1]
    B = img[:, :, 0]
    patchsz = 5
    k1, k2 = img.shape[0] // patchsz, img.shape[1] // patchsz

    AMEER = np.mean([AME(R[i:i+patchsz, j:j+patchsz]) for i in range(0, img.shape[0], patchsz) for j in range(0, img.shape[1], patchsz)])
    AMEEG = np.mean([AME(G[i:i+patchsz, j:j+patchsz]) for i in range(0, img.shape[0], patchsz) for j in range(0, img.shape[1], patchsz)])
    AMEEB = np.mean([AME(B[i:i+patchsz, j:j+patchsz]) for i in range(0, img.shape[0], patchsz) for j in range(0, img.shape[1], patchsz)])

    uiconm = AMEER + AMEEG + AMEEB
    return uiconm


def UIQM(img):
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    img = img/255
    uicm = UICM(img)
    uism = UISM(img)
    uiconm = UIConM(img)
    uiqm = c1 * uicm + c2 * uism + c3 * uiconm
    print(f"UIQM Score: {uiqm}")
    return uiqm



def calculate_uiqm_from_pil(img):
    #image_cv = np.array(image_pil)
    if isinstance(img, Image.Image):
        img_array = np.array(img)
    else:
        # 如果已经是Numpy数组，直接使用
        img_array = img
    return UIQM(img_array)

# 测试用例
img = cv2.imread('/public/home/shaojx8/ADPCC_Code-main/OutputImages/26.png')  # 更换为你的测试图像路径
uiqm_score = UIQM(img)
print(f"UIQM Score: {uiqm_score}")
