"""
2022.4.17
author:alian
function:
核查.json标签数量
"""
import numpy as np
import scipy.special
import os
import cv2

path = r'D:\Users\zr159\Desktop\Ultra-Fast-Lane-Detection-master\datasets\label'  # 实例图像的路径
a = os.listdir(path)
print(a)
for i in a:
    img = cv2.imread(os.path.join(path,i),-1)
    print(set((img.flatten())))
