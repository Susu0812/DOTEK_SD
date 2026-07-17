"""
2022.4.20
author:alian
function:
训练参数配置
"""

import argparse,datetime,os

def get_args():  # 配置训练参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='./datasets/newdata/train/',help='数据库路径')
    parser.add_argument('--val_source', type=str, default='./datasets/newdata/test/',help='验证数据集路径')
    parser.add_argument('--log_path', type=str, default='./logs/', help='模型保存路径')
    parser.add_argument('--epoch', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--optimizer', type=str, default='Adam', help='优化器：[SGD,Adam]')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减系数')
    parser.add_argument('--momentum', type=float, default=0.9, help='动量')
    parser.add_argument('--scheduler', type=str, default='cos', help='调度器:[multi, cos]')
    parser.add_argument('--gamma', type=float, default=0.1, help='模型预热')
    parser.add_argument('--warmup', type=str, default='linear', help='模型预热')
    parser.add_argument('--warmup_iters', type=int, default=100, help='模型预热')
    parser.add_argument('--backbone', type=str, default='18', help='网络骨干')
    parser.add_argument('--use_aux', type=bool, default=True, help='是否使用语义标签')
    parser.add_argument('--griding_num', type=int, default=50, help='网格列数')
    parser.add_argument('--row_anchor', type=int, default=18, help='行锚框')
    parser.add_argument('--num_lanes', type=int, default=1, help='车道数')
    parser.add_argument('--sim_loss_w', type=int, default=0, help='loss')
    parser.add_argument('--shp_loss_w', type=int, default=0, help='loss')
    parser.add_argument('--resume',  type=str, default=None, help='继续训练')
    parser.add_argument('--finetune', type=str, default=None,
                        help='load model weights only and create a fresh optimizer')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='micro-batches accumulated per optimizer update')
    parser.add_argument('--amp', action='store_true',
                        help='enable CUDA automatic mixed precision')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='DataLoader worker process count')
    parser.add_argument('--low_light_exposure', type=int, default=1,
                        help='times each reviewed low-light image appears per epoch')
    parser.add_argument('--seed', type=int, default=20260716,
                        help='training and sampler random seed')
    parser.add_argument('--auto_backup', action='store_true', help='automatically backup current code in the log path')
    parser.add_argument('--distributed',type=bool, default=False, help='分布式训练')
    opt = parser.parse_args()
    return opt


def validate_finetune_options(opt):
    positive_fields = (
        'batch_size', 'accumulation_steps', 'low_light_exposure', 'epoch'
    )
    for field in positive_fields:
        if getattr(opt, field) <= 0:
            raise ValueError(f'{field} must be positive')
    if opt.num_workers < 0:
        raise ValueError('num_workers must not be negative')
    if opt.finetune is not None:
        if opt.resume is not None:
            raise ValueError('finetune and resume cannot be used together')
        if opt.batch_size * opt.accumulation_steps != 64:
            raise ValueError('fine-tuning requires effective batch size 64')
    return opt

def get_work_dir(opt):  # 模型保存路径
    now = datetime.datetime.now().strftime('%m%d_%H%M')  # 获得当前时间
    hyper_param_str = '_lr_%1.0e_b_%d' % (opt.learning_rate, opt.batch_size)
    work_dir = os.path.join(opt.log_path, now+hyper_param_str)
    return work_dir
