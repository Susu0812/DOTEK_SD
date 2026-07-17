import onnxruntime
import numpy as np
import cv2
from PIL import Image
import scipy.special
import argparse

# 从data.constant导入行锚点配置
from data.constant import culane_row_anchor, tusimple_row_anchor, my_row_anchor

def preprocess_image(image_path, target_size=(288, 800)):
    """预处理输入图像 - 基于demo_custom.py的变换"""
    # 读取图像
    image = Image.open(image_path)
    
    # 应用与demo_custom.py相同的变换
    # 1. 调整大小到(288, 800)
    # print(f"原始图像大小: {image}")
    image = image.resize((target_size[1], target_size[0]))
    
    # 2. 转换为张量并确保float32类型
    image = np.array(image).astype(np.float32) / 255.0
    
    # 3. 如果是灰度图，转换为3通道
    if len(image.shape) == 3 and image.shape[2] == 3:
        # 转换为灰度图
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # 将灰度图复制三份转为RGB图
        image = np.stack([gray_image, gray_image, gray_image], axis=2).astype(np.float32)
    
    # 4. 转换为CHW格式
    if image.shape[2] == 3:
        image = np.transpose(image, (2, 0, 1))
    
    # 5. 应用与训练时相同的归一化参数，确保使用float32
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    image = (image - mean) / std
    
    # 6. 确保最终数据类型为float32
    image = image.astype(np.float32)
    
    # 7. 添加批次维度
    image = np.expand_dims(image, axis=0)

    print(f"预处理后图像形状: {image.shape}, 数据类型: {image.dtype}")
    print(image.dtype)
    
    return image

def inference_onnx(model_path, image_path):
    """使用ONNX模型进行推理"""
    # 创建ONNX Runtime会话
    session = onnxruntime.InferenceSession(model_path)
    
    # 预处理图像
    input_data = preprocess_image(image_path)

    # 获取输入名称
    input_name = session.get_inputs()[0].name
    
    # 运行推理
    outputs = session.run(None, {input_name: input_data})
    print(outputs[0].shape)
    
    # 输出通常是[group_cls]，即车道线分类结果
    group_cls = outputs[0]
    
    return group_cls

def visualize_lanes(image_path, lane_points, output_path=None):
    """可视化车道线检测结果"""
    # 读取原始图像
    img = cv2.imread(image_path)
    
    # 单车道使用单一颜色
    color = (0, 0, 255)  # 红色
    
    for lane_idx, dots in enumerate(lane_points):
        # 只处理第一条有效车道线
        if dots:  # 如果有检测点
            for dot in dots:
                cv2.circle(img, dot, 5, color, -1)
            break  # 只显示第一条车道线
    
    # 保存或显示结果
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"结果已保存到: {output_path}")
    
    return img

def postprocess_lane_detection(output, griding_num=200, num_row_anchors=18, num_lanes=1, 
                              original_size=(1640, 590)):
    """
    后处理车道线检测输出 - 针对单车道优化
    """
    # 使用culane行锚点配置
    row_anchor = tusimple_row_anchor
    img_w, img_h = original_size
    
    # 复制demo_custom.py的后处理逻辑
    out_j = output[0]  # 取第一个批次的输出 [201, 18, 4]
    out_j = out_j[:, ::-1, :]  # 将第二维度倒序
    
    # softmax计算概率
    prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)  # [200, 18, 4]
    
    # 计算位置
    idx = np.arange(griding_num) + 1  # 1-200
    idx = idx.reshape(-1, 1, 1)  # [200, 1, 1]
    loc = np.sum(prob * idx, axis=0)  # [18, 4]
    
    # 找到最大值索引
    out_j = np.argmax(out_j, axis=0)  # [18, 4]
    
    # 将最大值索引等于griding_num的位置归零（背景）
    loc[out_j == griding_num] = 0
    
    # 计算网格间隔
    grids = np.linspace(0, 800 - 1, griding_num)
    grid = grids[1] - grids[0]
    
    # 准备返回的车道线点列表
    lane_points = []
    
    # 只处理第一条车道线（索引0）
    lane_dots = []
    if np.sum(loc[:, 0] != 0) > 2:  # 有效车道线点大于2个
        for row_idx in range(loc.shape[0]):  # 遍历每个行锚点
            if loc[row_idx, 0] > 0:
                # 计算原始图像坐标
                x = int(loc[row_idx, 0] * grid * img_w / 800) - 1
                y = int(img_h * (row_anchor[num_row_anchors - 1 - row_idx] / 288)) - 1
                lane_dots.append((x, y))
    lane_points.append(lane_dots)
    
    return lane_points, loc

def parse_opt():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./ultra_fast_lane_detection.onnx', help='ONNX模型路径')
    parser.add_argument('--source', type=str, default='./datasets/all_640_480/pic/DJI_20250814100401_0005_D_frame_01750.jpg', help='测试图像路径')
    parser.add_argument('--savepath', type=str, default='./datasets/show/output.jpg', help='保存路径')
    parser.add_argument('--griding_num', type=int, default=100, help='网格数')
    parser.add_argument('--num_row_anchors', type=int, default=56, help='行锚点数')
    parser.add_argument('--num_lanes', type=int, default=1, help='车道数')
    parser.add_argument('--original_width', type=int, default=640, help='原始图像宽度')
    parser.add_argument('--original_height', type=int, default=480, help='原始图像高度')
    return parser.parse_args()

def main():
    # 解析参数
    opt = parse_opt()
    
    try:
        # 运行推理
        print("开始ONNX推理...")
        output = inference_onnx(opt.model, opt.source)
        print(f"原始输出形状: {output.shape}")
        
        # 后处理
        lane_points, _ = postprocess_lane_detection(
            output, 
            griding_num=opt.griding_num,
            num_row_anchors=opt.num_row_anchors,
            num_lanes=opt.num_lanes,
            original_size=(opt.original_width, opt.original_height)
        )
        
        print("推理完成!")
        print(f"检测到 {len([p for p in lane_points if p])} 条有效车道线")

        print(lane_points)
        # 可视化结果
        result_image = visualize_lanes(opt.source, lane_points, opt.savepath)
        
        print(f"结果已处理并保存")
        
    except Exception as e:
        print(f"推理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()