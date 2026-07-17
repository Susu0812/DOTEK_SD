import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ECA(nn.Module):
    """为车道线检测优化的ECA模块"""
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        # 自适应确定卷积核大小，针对车道线特征优化
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        k = max(3, min(k, 9))  # 车道线检测推荐核大小范围
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class SDAA_H(nn.Module):
    '''
    空间方向感知注意力模块 (Spatial Directional Awareness Attention, SDAA)
    '''
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        
        # 共享第一个卷积（省参数）
        self.fc_down = nn.Conv2d(channels, hidden, 1, bias=False)
        self.relu = nn.ReLU6(inplace=True)
        
        # 分支上采样
        self.fc_v_up = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, x):
        v = x.mean(dim=3, keepdim=True)   # (B, C, H, 1)
        v = self.relu(self.fc_down(v))
        att_v = torch.sigmoid(self.fc_v_up(v))
        att = att_v
        return x * att

class SDAA_W(nn.Module):
    '''
    空间方向感知注意力模块 (Spatial Directional Awareness Attention, SDAA)
    '''
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        
        # 共享第一个卷积（省参数）
        self.fc_down = nn.Conv2d(channels, hidden, 1, bias=False)
        self.relu = nn.ReLU6(inplace=True)
        
        # 分支上采样
        self.fc_v_up = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, x):
        v = x.mean(dim=2, keepdim=True)   # (B, C, W, 1)
        v = self.relu(self.fc_down(v))
        att_v = torch.sigmoid(self.fc_v_up(v))
        att = att_v
        return x * att

class DFEUModule(nn.Module):
    """
    方向特征增强单元 (Directional Feature Enhancement Unit, DFEU)  
    """
    def __init__(self, channels: int, expansion: int = 2, kernel_size: int = 3):
        super().__init__()
        mid = int(channels * expansion)
        
        # Expand
        self.expand = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU6(inplace=True)
        )
        
        # Longitudinal conv: (K, 1) → captures vertical continuity
        self.dw_long = nn.Conv2d(
            mid, mid, 
            kernel_size=(kernel_size, 1), 
            padding=(kernel_size//2, 0), 
            groups=mid, 
            bias=False
        )
        # Lateral edge conv: (1, K) → detects left/right boundaries
        self.dw_edge = nn.Conv2d(
            mid, mid, 
            kernel_size=(1, kernel_size), 
            padding=(0, kernel_size//2), 
            groups=mid, 
            bias=False
        )
        self.sdaa_h = SDAA_H(mid)
        self.sdaa_w = SDAA_W(mid)
        
        # Project back
        self.project = nn.Sequential(
            # nn.BatchNorm2d(mid),
            nn.ReLU6(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        x = self.expand(x)
        x = self.sdaa_w(self.dw_long(x)) + self.sdaa_h(self.dw_edge(x))
        x = self.project(x)
        return x

class DPD(nn.Module):
    """
    方向保持型下采样 (Direction-Preserving Downsampling, DPD)
    轻量级方向感知下采样模块，专为车道线检测设计。
    - 使用 3x1 和 1x3 深度卷积捕获方向信息
    - 通过 PixelUnshuffle 实现无损下采样
    - 中间通道压缩控制计算量
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # 压缩通道以控制 PixelUnshuffle 后的膨胀
        hidden_ch = in_ch // 4  # 可调：//2, //4, //8
        self.out_ch = out_ch
        
        if self.out_ch < 96:
            # 方向性深度卷积（轻量！）
            self.dw_v = nn.Conv2d(in_ch, in_ch, kernel_size=(3, 1), padding=(1, 0), groups=in_ch, bias=False)
            self.dw_h = nn.Conv2d(in_ch, in_ch, kernel_size=(1, 3), padding=(0, 1), groups=in_ch, bias=False)
            
            # 融合 + 激活
            self.bn_fuse = nn.BatchNorm2d(in_ch)
            self.act = nn.ReLU6(inplace=True)
        
        # 通道压缩（关键！避免 unshuffle 后通道爆炸）
        self.compress = nn.Conv2d(in_ch, hidden_ch, kernel_size=1, bias=False)
        
        # 无损下采样
        self.unshuffle = nn.PixelUnshuffle(2)  # [B, hidden_ch, H, W] → [B, 4*hidden_ch, H/2, W/2]
        
        # 输出投影
        self.expand = nn.Conv2d(4 * hidden_ch, out_ch, kernel_size=1, bias=False)
        self.bn_out = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        # 方向特征提取
        if self.out_ch < 96:
            x_dir = self.act(self.bn_fuse(self.dw_v(x) + self.dw_h(x)))
        else:
            x_dir = x
        
        # 通道压缩 + PixelUnshuffle 下采样
        x_comp = self.compress(x_dir)
        x_unshuffled = self.unshuffle(x_comp)  # [B, 4*hidden_ch, H/2, W/2]
        
        # 输出
        x_out = self.bn_out(self.expand(x_unshuffled))
        return x_out

class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, kernel_size: int, expansion: int = 2):
        super().__init__()
        self.dwconv = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=kernel_size//2, groups=in_ch, bias=False)

        self.att = ECA(in_ch)

        self.dw = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            DFEUModule(in_ch, expansion=expansion, kernel_size=kernel_size)
        )

    def forward(self, x):   
        x = self.dwconv(x) + x
        x = self.att(x) + x
        x = self.dw(x) + x
        return x

class LDAE_Net(nn.Module):
    '''
    Lightweight Direction-Aware Enhancement Neural Network (LDAE-Net)   轻量级方向感知增强神经网络
    '''
    def __init__(self, block_list=[2, 2, 6, 2], base_channels=[48, 96, 192, 384], 
        kernel_size=[3, 3, 3, 3], expansion=[2, 2, 2, 2]):
        """
        Args:
            block_list (list): 每个 stage 中 DFEUModule 模块的数量，长度为 4，如 [2, 2, 6, 2]
            base_channels (list): 每个 stage 的输出通道数，如 [48, 96, 192, 384]
        """
        super(LDAE_Net, self).__init__()
        assert len(block_list) == 4, "block_list must have 4 elements"
        assert len(base_channels) == 4, "base_channels must have 4 elements"

        # Stem: 
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels[0]),
            nn.ReLU6(inplace=True),
            DPD(base_channels[0], base_channels[0])
        )

        stage1_layers = []
        for i in range(block_list[0]):
            stage1_layers.append(BasicBlock(base_channels[0], kernel_size=kernel_size[0], expansion=expansion[0]))
        self.stage1 = nn.Sequential(*stage1_layers)

        stage2_layers = []
        stage2_layers.append(DPD(base_channels[0], base_channels[1]))  # 下采样到128通道
        for i in range(block_list[1]):
            stage2_layers.append(BasicBlock(base_channels[1], kernel_size=kernel_size[1], expansion=expansion[1]))
        self.stage2 = nn.Sequential(*stage2_layers)

        stage3_layers = []
        stage3_layers.append(DPD(base_channels[1], base_channels[2]))  # 下采样到256通道
        for i in range(block_list[2]):
            stage3_layers.append(BasicBlock(base_channels[2], kernel_size=kernel_size[2], expansion=expansion[2]))
        self.stage3 = nn.Sequential(*stage3_layers)

        stage4_layers = []
        stage4_layers.append(DPD(base_channels[2], base_channels[3]))  # 下采样到512通道
        for i in range(block_list[3]):
            stage4_layers.append(BasicBlock(base_channels[3], kernel_size=kernel_size[3], expansion=expansion[3]))
        self.stage4 = nn.Sequential(*stage4_layers)

    def forward(self, x):
        out = self.stem(x)
        out1 = self.stage1(out)
        out2 = self.stage2(out1)
        out3 = self.stage3(out2)
        out4 = self.stage4(out3)
        return out2, out3, out4
