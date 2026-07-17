"""
2022.4.20
author:alian
function：
自定义数据加载器
"""
# 导入库
import torch
import torchvision.transforms as transforms
import torch.utils.data
# 导入项目源码中的文件
import data.mytransforms as mytransforms
from data.constant import tusimple_row_anchor, my_row_anchor
from utils_alian.dataset_alian import ClsDataset
from utils_alian.finetune_utils import LowLightExposureSampler

# data/mytransforms.py 或 utils_alian/transforms_alian.py

class RepeatGrayToRGB:
    """将单通道灰度图重复为三通道 RGB"""
    def __call__(self, x):
        return x.repeat(3, 1, 1)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def build_train_sampler(dataset, distributed, low_light_exposure=1,
                        seed=20260716):
    if distributed:
        if low_light_exposure != 1:
            raise ValueError('low-light exposure sampler does not support distributed mode')
        return torch.utils.data.distributed.DistributedSampler(dataset)
    if low_light_exposure > 1:
        return LowLightExposureSampler(
            dataset.img_paths,
            prefix='lowlight_camera_full_rgb_',
            exposure=low_light_exposure,
            seed=seed,
        )
    return torch.utils.data.RandomSampler(dataset)


def get_train_loader(batch_size, data_root, griding_num, use_aux, distributed,
                     num_lanes, num_workers=8, low_light_exposure=1,
                     seed=20260716):
    target_transform = transforms.Compose([
        # mytransforms.FreeScaleMask((192, 256)),  # 图像缩放功能
        mytransforms.FreeScaleMask((288, 384)),
        mytransforms.MaskToTensor(),
    ])
    segment_transform = transforms.Compose([
        # mytransforms.FreeScaleMask((24, 32)),
        mytransforms.FreeScaleMask((36, 48)),
        mytransforms.MaskToTensor(),
    ])
    img_transform = transforms.Compose([
        # transforms.Resize((192, 256)),
        transforms.Resize((288, 384)),
        transforms.ToTensor(),
        transforms.Grayscale(),
        RepeatGrayToRGB(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    simu_transform = mytransforms.Compose2([
        mytransforms.RandomRotate(6),
        mytransforms.RandomUDoffsetLABEL(100),
        mytransforms.RandomLROffsetLABEL(200),
        mytransforms.RandomHorizontalFlip(prob=0.5)  # 添加水平翻转，50%概率
    ])
    # 自定义数据集
    train_dataset = ClsDataset(data_root,
                                   img_transform=img_transform, target_transform=target_transform,
                                   simu_transform=simu_transform,
                                   griding_num=griding_num,
                                   row_anchor=my_row_anchor,
                                   segment_transform=segment_transform, use_aux=use_aux, num_lanes=num_lanes)

    sampler = build_train_sampler(
        train_dataset,
        distributed=distributed,
        low_light_exposure=low_light_exposure,
        seed=seed,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader

def get_val_loader(batch_size, data_root, griding_num, use_aux, distributed,
                   num_lanes, num_workers=8):
    target_transform = transforms.Compose([
        # mytransforms.FreeScaleMask((192, 256)),  # 图像缩放功能
        mytransforms.FreeScaleMask((288, 384)),
        mytransforms.MaskToTensor(),
    ])
    segment_transform = transforms.Compose([
        # mytransforms.FreeScaleMask((24, 32)),  # 36 48  24 32
        mytransforms.FreeScaleMask((36, 48)),
        mytransforms.MaskToTensor(),
    ])
    img_transform = transforms.Compose([
        # transforms.Resize((192, 256)),
        transforms.Resize((288, 384)),
        transforms.ToTensor(),
        transforms.Grayscale(),
        RepeatGrayToRGB(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # 验证数据集不使用数据增强
    val_dataset = ClsDataset(data_root,
                               img_transform=img_transform, target_transform=target_transform,
                               simu_transform=None,  # 不使用模拟变换
                               griding_num=griding_num,
                               row_anchor=my_row_anchor,
                               segment_transform=segment_transform, use_aux=use_aux, num_lanes=num_lanes)

    if distributed:
        sampler = SeqDistributedSampler(val_dataset, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(val_dataset)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return val_loader

class SeqDistributedSampler(torch.utils.data.distributed.DistributedSampler):  # 分布式采样
    '''
    Change the behavior of DistributedSampler to sequential distributed sampling.
    The sequential sampling helps the stability of multi-thread testing, which needs multi-thread file io.
    Without sequentially sampling, the file io on thread may interfere other threads.
    '''
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        super().__init__(dataset, num_replicas, rank, shuffle)
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
        num_per_rank = int(self.total_size // self.num_replicas)

        # sequential sampling
        indices = indices[num_per_rank * self.rank : num_per_rank * (self.rank + 1)]
        assert len(indices) == self.num_samples

        return iter(indices)
