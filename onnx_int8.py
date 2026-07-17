import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import onnx
from onnxruntime.quantization import (
    quantize_static, 
    CalibrationDataReader, 
    QuantType, 
    QuantFormat, 
    CalibrationMethod
)

class RepeatGrayToRGB(object):
    def __call__(self, img):
        return img.repeat(3, 1, 1)

class LaneDetDataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder, input_name='input', 
                 max_samples=None, shuffle=True):
        self.image_folder = calibration_image_folder
        self.input_name = input_name
        self.enum_data = None
        
        if not os.path.exists(calibration_image_folder):
            raise ValueError(f"校准图片文件夹不存在: {calibration_image_folder}")

        self.datas = [os.path.join(calibration_image_folder, f) 
                      for f in os.listdir(calibration_image_folder) 
                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        if shuffle:
            import random
            random.seed(42)
            random.shuffle(self.datas)
        
        if max_samples is not None:
            self.datas = self.datas[:max_samples]
        
        print(f"📊 校准数据统计:")
        print(f"   - 总图片数: {len(os.listdir(calibration_image_folder))}")
        print(f"   - 使用数量: {len(self.datas)}")

        self.transform = transforms.Compose([
            transforms.Resize((288, 384)),
            transforms.ToTensor(),
            transforms.Grayscale(),
            RepeatGrayToRGB(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        self.current_idx = 0

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(self.datas)
        
        try:
            image_path = next(self.enum_data)
            self.current_idx += 1
            
            if self.current_idx % 100 == 0:
                print(f"   校准进度: {self.current_idx}/{len(self.datas)}")
                
        except StopIteration:
            return None

        try:
            img = Image.open(image_path).convert('RGB') 
        except Exception as e:
            print(f"⚠️ 无法读取图片 {image_path}: {e}")
            return self.get_next()

        img_tensor = self.transform(img)
        img_numpy = img_tensor.cpu().numpy()
        img_numpy = np.expand_dims(img_numpy, axis=0).astype(np.float32)

        return {self.input_name: img_numpy}


def main():
    fp32_model_path = 'resnet_288_384.onnx'       
    int8_model_path = 'resnet_288_384_int8.onnx'  
    calib_folder = './datasets/all_640_480/pic'   

    if not os.path.exists(fp32_model_path):
        print(f"❌ 错误：找不到输入模型 {fp32_model_path}")
        return

    # 校准数据量设置
    MAX_CALIBRATION_SAMPLES = 1000  # 使用1000张
    
    data_reader = LaneDetDataReader(
        calib_folder, 
        input_name='input',
        max_samples=MAX_CALIBRATION_SAMPLES,
        shuffle=True
    )

    print("\n🚀 开始进行静态量化...")
    
    # ==========================================
    # 修复：移除了 optimize_model 参数
    # ==========================================
    quantize_static(
        model_input=fp32_model_path,
        model_output=int8_model_path,
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QDQ, 
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,  # 逐通道量化，精度更高
    )
    
    print(f"\n✅ 量化完成！INT8 模型已保存至: {int8_model_path}")


if __name__ == '__main__':
    main()