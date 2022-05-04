import onnxoptimizer
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from nets import get_model_from_name

if __name__ == "__main__":

    # 训练模型路径
    modelPath = "./logs/ep099-loss0.076-val_loss0.003.pth"
    # 输出模型路径
    savePath = "model_data/best-class.onnx"

    # 类别数量
    num_classes=5

    # 图片参数
    imageChannel=3
    imageWidth=224
    imageHeight=224
    input_shape = [imageChannel, imageWidth,  imageHeight]


    # 加载模型
    device  = torch.device('cpu')
    backbone='resnet50'
    model=''
    if backbone != "vit":
        model  = get_model_from_name[backbone](num_classes = num_classes, pretrained = True)
    else:
        model  = get_model_from_name[backbone](input_shape = input_shape, num_classes = num_classes, pretrained = True)

    model.load_state_dict(torch.load(modelPath, map_location=device))
    # 生产模型
    model = model.eval()


    # 导出参数
    ## 输入数据结构
    input_shape2 = (imageChannel,  imageWidth, imageHeight)
    dummy_input = (
        torch.randn(1, *input_shape2).to(device)
    )
    ## 输入输出参数
    input_names = ["input"]
    output_names = ["output"]

    # torch.onnx.export(model, (torch.rand(1, 3, 224, 224).to(device), torch.rand(1, 3, 224, 224).to(device)), args.out_path, input_names=['input'],
    #                   output_names=['output'], export_params=True)
    torch.onnx.export(model,
                      dummy_input,
                      savePath,
                      input_names=input_names,
                      output_names=output_names,
                      export_params=True 
                    #   verbose=True,                                        
                    #   keep_initializers_as_inputs=True
                      )    
    
    