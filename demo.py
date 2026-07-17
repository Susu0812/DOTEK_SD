"""
2022.04.20
author:alian
车道线检测
测试自定义的数据集，并保存成检测结果图
H,W：原图尺寸；h:行锚框数，w:单元格数，C：车道线数
"""
# 导入项目源码中的文件
from model.model import parsingNet
from utils.dist_utils import dist_print
from data.constant import tusimple_row_anchor
# 导入库
import scipy.special, tqdm
import torchvision.transforms as transforms
from PIL import Image
import os,glob,cv2,argparse
import numpy as np
import torch.utils.data


class TestDataset(torch.utils.data.Dataset):  # 加载测试数据集----------------------------------------------------------
    def __init__(self, path, img_transform=None):
        super(TestDataset, self).__init__()
        self.path = path
        self.img_transform = img_transform
        self.img_list = glob.glob('%s/*.png'%self.path)

    def __getitem__(self, index):
        name = glob.glob('%s/*.png'%self.path)[index]
        img = Image.open(name)

        if self.img_transform is not None:
            img = self.img_transform(img)
        return img, name

    def __len__(self):
        return len(self.img_list)

class TestDataset_jpg(torch.utils.data.Dataset):  # 加载测试数据集----------------------------------------------------------
    def __init__(self, path, img_transform=None):
        super(TestDataset_jpg, self).__init__()
        self.path = path
        self.img_transform = img_transform
        self.img_list = glob.glob('%s/*.jpg'%self.path)

    def __getitem__(self, index):
        name = glob.glob('%s/*.jpg'%self.path)[index]
        img = Image.open(name)

        if self.img_transform is not None:
            img = self.img_transform(img)
        return img, name

    def __len__(self):
        return len(self.img_list)


def parse_opt():  # 参数指定-------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='18', help='骨干网络')
    parser.add_argument('--model', type=str, default='./culane_18.pth', help='模型路径')  # 设置
    parser.add_argument('--dataset', type=str, default='datasets', help='数据集名称')
    parser.add_argument('--source', type=str, default='datasets/pic', help='测试路径')  # 设置
    parser.add_argument('--savepath', type=str, default='datasets/test2', help='保存路径')  # 设置
    parser.add_argument('--save_video', type=bool, default=False, help='保存为视频')
    parser.add_argument('--griding_num', type=int, default=200, help='网格数')   #200,18,4     100,56,4
    parser.add_argument('--num_row_anchors', type=int, default=18, help='锚框行')
    parser.add_argument('--num_lanes', type=int, default=4, help='车道数')
    opt = parser.parse_args()
    return opt


def run(opt):
    dist_print('start testing...')
    backbone, model, dataset, source, savepath = opt.backbone, opt.model, opt.dataset, opt.source, opt.savepath
    save_video, griding_num, num_row_anchors, num_lanes = opt.save_video, opt.griding_num, opt.num_row_anchors, opt.num_lanes
    assert opt.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']

    net = parsingNet(pretrained=False, backbone=backbone, cls_dim=(griding_num + 1, num_row_anchors, num_lanes),
                     use_aux=False).cuda()

    state_dict = torch.load(model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v
    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    datasets = TestDataset_jpg(source, img_transform=img_transforms)
    # 不再需要硬编码的 img_w, img_h
    row_anchor = tusimple_row_anchor

    for dataset_item in zip(datasets):  # 注意：你的原代码这里可能有点问题，我帮你修正了循环
        loader = torch.utils.data.DataLoader(dataset_item, batch_size=1, shuffle=False, num_workers=0)

        # 视频保存逻辑应该放在外层，为每个数据集创建一个视频文件
        # if save_video:
        #     fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        #     # 视频尺寸需要是动态的，或者所有图片尺寸一致
        #     # vout = cv2.VideoWriter(dataset_name + '.avi', fourcc, 30.0, (img_w, img_h))

        for i, data in enumerate(tqdm.tqdm(loader)):
            imgs, names = data
            imgs = imgs.cuda()
            with torch.no_grad():
                pred = net(imgs)

            out_j = pred[0].data.cpu().numpy()
            out_j = out_j[:, ::-1, :]
            prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
            idx = np.arange(griding_num) + 1
            idx = idx.reshape(-1, 1, 1)
            loc = np.sum(prob * idx, axis=0)
            out_j = np.argmax(out_j, axis=0)
            loc[out_j == griding_num] = 0
            out_j = loc

            # --- 关键修改部分 ---
            img = cv2.imdecode(np.fromfile(os.path.join(names[0]), dtype=np.uint8),
                               cv2.IMREAD_COLOR)
            # 动态获取当前图像的真实高和宽
            img_h, img_w, _ = img.shape

            grids = np.linspace(0, 800 - 1, griding_num)
            grid = grids[1] - grids[0]

            for i in range(out_j.shape[1]):
                if np.sum(out_j[:, i] != 0) > 2:
                    for k in range(out_j.shape[0]):
                        if out_j[k, i] > 0:
                            # 现在这里的 img_w 和 img_h 是当前图片的真实尺寸，计算结果将是正确的
                            point = (int(out_j[k, i] * grid * img_w / 800) - 1,
                                     int(img_h * (row_anchor[num_row_anchors - 1 - k] / 288)) - 1)
                            cv2.circle(img, point, 5, (0, 0, 255), -1)

            # 保存逻辑（视频保存逻辑如果需要，也需要在这里处理尺寸问题）
            if save_video:
                # pass # 如果要保存视频，需要确保所有图片尺寸一致，或者在第一次循环时初始化vout
                print("Video saving is not fully implemented for varying image sizes.")
            else:
                cv2.imwrite(os.path.join(savepath, os.path.basename(names[0])), img)

        # if save_video and 'vout' in locals():
        #     vout.release()

if __name__ == "__main__":
    import torch.backends.cudnn
    torch.backends.cudnn.benchmark = True  # 加速
    opt = parse_opt()  # 指定参数
    run(opt)
