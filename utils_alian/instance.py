"""
2022.4.17
author:alian
function:
根据labelme标注的文件生成实例掩膜
"""
import argparse
import glob
import json
import os
import os.path as ops
import cv2
import numpy as np


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, help='The origin path of image')
    parser.add_argument('--json_dir', type=str, help='The origin path of json')
    return parser.parse_args()

# 获得原始图像（.png），二值化图像以及实例分割图像
def process_json_file(img_path, json_path, instance_dst_dir):
    """
    :param img_path: 原始图像路径
    :param json_path: 标签文件路径
    :param instance_dst_dir:实例图像保存路径
    :return:
    """

    assert ops.exists(img_path), '{:s} not exist'.format(img_path)
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    instance_image = np.zeros([image.shape[0], image.shape[1]], np.uint8)
    with open(json_path, 'r',encoding='utf8') as file:
        info_dict = json.load(file)
        for ind,info in enumerate(info_dict['shapes']):
            contours = info['points']
            contours = np.array(contours,dtype=int)
            # 绘制多边形
            cv2.fillPoly(instance_image, [contours], (ind+1, ind+1, ind+1)) #  * 50 + 20

        instance_image_path = img_path.replace(os.path.dirname(img_path),instance_dst_dir)
        cv2.imwrite(instance_image_path.replace('.jpg','.png'), instance_image)  # 实例分割图像
# 训练库构建
def process_tusimple_dataset(img_dir,json_dir):
    """
	:param json_dir: 标签文件路径
    :param img_dir: 原始图像路径
    :return:
    """
    gt_instance_dir = ops.join(os.path.dirname(img_dir), 'label')  # 与原始图片文件夹在同级父目录下
    os.makedirs(gt_instance_dir, exist_ok=True)

    for img_path in glob.glob('{:s}/*.jpg'.format(img_dir)):
        json_path = img_path.replace(img_dir,json_dir).replace('.jpg','.json')
        process_json_file(img_path, json_path, gt_instance_dir)
    return

if __name__ == '__main__':
    img_dir = r'D:\Users\zr159\Desktop\Ultra-Fast-Lane-Detection-master\datasets\1'  # 原始图片路径
    json_dir = r'D:\Users\zr159\Desktop\Ultra-Fast-Lane-Detection-master\datasets\json'  # 标注文件路径
    process_tusimple_dataset(img_dir,json_dir)
