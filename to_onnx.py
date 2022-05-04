import onnxoptimizer
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
# import tensorrt as trt

from nets import get_model_from_name

if __name__ == "__main__":

    # 训练模型路径
    modelPath = "./logs/ep011-loss0.087-val_loss0.011.pth"
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
        model  = get_model_from_name[backbone](num_classes = num_classes, pretrained = False)
    else:
        model  = get_model_from_name[backbone](input_shape = input_shape, num_classes = num_classes, pretrained = False)

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

    # torch.onnx.export(model,
    #                   dummy_input,
    #                   savePath,
    #                   input_names=input_names,
    #                   output_names=output_names,
    #                   export_params=True
    #                   )   
    im=dummy_input
    f=savePath
    opset=12
    # if trt.__version__[0] == '7': # TensorRT 7 handling https://github.com/ultralytics/yolov5/issues/6012
    #     opset=12
    # else: # TensorRT >= 8
    #     opset=13

    torch.onnx.export(
        model, 
        im, 
        f, 
        verbose=False, 
        opset_version=opset,
        # training=TrainingMode.EVAL,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        # dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
        #                                 'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
        #                                 } if dynamic else None
                                        )                       
    
    