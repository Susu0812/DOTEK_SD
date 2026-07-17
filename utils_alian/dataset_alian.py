"""
2022.4.20
author:alian
function：
Ultra-Fast-Lane-Detection：https://github.com/cfzd/Ultra-Fast-Lane-Detection
加载自定义数据集
"""
import torch
from PIL import Image
import pdb
import numpy as np
import glob
from data.mytransforms import find_start_pos
import torch.utils.data


def loader_func(path):
    return Image.open(path)

# 加载自定义的数据集
class ClsDataset(torch.utils.data.Dataset):
    def __init__(self, path, img_transform=None, target_transform=None, simu_transform=None, griding_num=50,
                 load_name=False,
                 row_anchor=None, use_aux=False, segment_transform=None, num_lanes=4):
        super(ClsDataset, self).__init__()
        self.img_transform = img_transform  #
        self.target_transform = target_transform
        self.segment_transform = segment_transform  #
        self.simu_transform = simu_transform  #
        self.path = path
        self.griding_num = griding_num  # 网格列数：100
        self.load_name = load_name
        self.use_aux = use_aux  # 是否使用语义标签
        self.num_lanes = num_lanes  # 车道线的数量 2
        self.row_anchor = row_anchor  # 锚框出现的行位置，默认总行数为288
        self.row_anchor.sort()
        self.img_paths = sorted(glob.glob(f'{path}/pic/*.jpg'))
        self.label_paths = sorted(glob.glob(f'{path}/label/*.png'))

    def __getitem__(self, index):  # 获得图像数组
        img_path = self.img_paths[index]
        label_path = self.label_paths[index]

        img = loader_func(img_path)
        label = loader_func(label_path)

        # labels = glob.glob('%s/label/*.png'%self.path)
        # label_path = labels[index]
        # label = loader_func(label_path)  # loader_func读取图像函数

        # imgs = glob.glob('%s/pic/*.jpg'%self.path)
        # img_path = imgs[index]
        # img = loader_func(img_path)  # 读取图像

        if self.simu_transform is not None:
            img, label = self.simu_transform(img, label)
        #
        lane_pts = self._get_index(label)  # 获得包含车道线的坐标[车道线数，56，2]（在原始图像上的，还未网格化）
        # 网格化，将包含车道线的列值进行网格化
        w, h = img.size
        cls_label = self._grid_pts(lane_pts, self.griding_num, w)
        if self.img_transform is not None:
            img = self.img_transform(img)
        # make the coordinates to classification label
        if self.use_aux:
            assert self.segment_transform is not None
            seg_label = self.segment_transform(label)
            return img, cls_label, seg_label

        if self.load_name:
            return img, cls_label

        return img, cls_label  # img：[3,288,800],cls_label:[56,2],seg_label:[36,100]

    def __len__(self):
        # return len(glob.glob('%s/label/*.png'%self.path))
        return len(self.label_paths)

    def _grid_pts(self, pts, num_cols, w):  # 获得网格数，返回[56,2]
        """
        pts:包含车道线的坐标[车道线，56，2]
        num_cols:网格列数
        w:图片的列数（像素单位）
        function：
        将图片按指定的列数划分网格数，并获得车道线所在的网格位置
        """
        # pts : numlane, n, 2
        num_lane, n, n2 = pts.shape  # [2,56,2]
        col_sample = np.linspace(0, w - 1, num_cols)  # 列间隔（在w=1920的原始图像上，分成100列，则每列的间隔为）
        # 在w=1920的原始图像上，分成100列，则包含车道线的坐标落在那一列上
        assert n2 == 2
        to_pts = np.zeros((n, num_lane))  # [56,2]
        for i in range(num_lane):
            pti = pts[i, :, 1]  # 包含车道线坐标的列值
            to_pts[:, i] = np.asarray(
                [int(pt // (col_sample[1] - col_sample[0])) if pt != -1 else num_cols for pt in pti])
        return to_pts.astype(int)

    def _get_index(self, label):  # 获取包含车道线的坐标，返回[车道线数,56,2]
        """
        label:1920*1080

        """
        w, h = label.size
        # 行锚框映射[64,68...284]-->[240,255,...1065]
        if h != 288:  # 若图像高度不为预设的288，则将行锚框row_anchors等比缩放到原始图像尺寸中[0,288]-->[0,1080]
            scale_f = lambda x: int((x * 1.0 / 288) * h)  # 定义一种匿名函数
            sample_tmp = list(map(scale_f, self.row_anchor))  # 根据函数和自变量做映射

        # 第一步骤：得到all_idx[2,56,2]的数组，包含车道线像素的坐标（在原始图像上的）
        all_idx = np.zeros((self.num_lanes, len(sample_tmp), 2))  # [2,56,2]
        # 实质是：获取车道线所在的坐标位置
        for i, r in enumerate(sample_tmp):  # 遍历行锚框[240,255,...1065]
            label_r = np.asarray(label)[int(round(r))]  # 获取该行的灰度值
            for lane_idx in range(1, self.num_lanes + 1):   # 遍历车道标签id，车道标签[1,2,...,n] 0一般为背景标签
                pos = np.where(label_r == lane_idx)[0]
                if len(pos) == 0:
                    all_idx[lane_idx - 1, i, 0] = r
                    all_idx[lane_idx - 1, i, 1] = -1
                    continue  # 继续下一步的循环，不再往下走
                pos = np.mean(pos)  # 车道线具有一定的宽度，取列均值
                all_idx[lane_idx - 1, i, 0] = r  # 获取行值
                all_idx[lane_idx - 1, i, 1] = pos  # 获取列值

        # 第2步骤all_idx_cp：下面的步骤为数据增强：将车道线坐标（在原始图像上的）延申至边缘
        all_idx_cp = all_idx.copy()
        for i in range(self.num_lanes):
            if np.all(all_idx_cp[i, :, 1] == -1): # 若没找到车道线则继续找，找到了就执行下一步
                continue
            # if there is no lane

            valid = all_idx_cp[i, :, 1] != -1  # 获得包含车道线的索引:仅包含true和false的列表
            # get all valid lane points' index
            valid_idx = all_idx_cp[i, valid, :] # 获得包含车道线的点
            # get all valid lane points
            if valid_idx[-1, 0] == all_idx_cp[0, -1, 0]:
                # 若最底部的有效车道线点在图像边缘，则不进行延申操作（图像增强）
                continue
            if len(valid_idx) < 6:  # 有效的车道线点至少6个
                continue

            valid_idx_half = valid_idx[len(valid_idx) // 2:, :]  # 取一半的有效点
            p = np.polyfit(valid_idx_half[:, 0], valid_idx_half[:, 1], deg=1)  # 用一次函数进行拟合，返回一次函数的系数
            start_line = valid_idx_half[-1, 0]  # 最底部的有效车道线点
            pos = find_start_pos(all_idx_cp[i, :, 0], start_line) + 1

            fitted = np.polyval(p, all_idx_cp[i, pos:, 0])  # 根据系数和自变量返回应变量数组
            fitted = np.array([-1 if y < 0 or y > w - 1 else y for y in fitted])

            assert np.all(all_idx_cp[i, pos:, 1] == -1)  # 条件为True时执行
            all_idx_cp[i, pos:, 1] = fitted
        if -1 in all_idx[:, :, 0]:  # 所有的行坐标
            pdb.set_trace()  # 程序终止
        return all_idx_cp  # [4,56,2]

