from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from torch import nn

from nets import get_model_from_name
from utils.utils import (cvtColor, get_classes, letterbox_image,
                         preprocess_input)


imgPath='D:/env/x5/test2/2.png'

img2 = cv2.imread(imgPath)
img2 = cv2.resize(img2, (224,224))
cv2.normalize(img2,img2, 0,255,cv2.NORM_MINMAX)

image = Image.open(imgPath)
image = cvtColor(image)
image_data1  = letterbox_image(image,             [224, 224],             False)
        # image_data  = np.transpose(
        #         np.expand_dims(
        #             preprocess_input(
        #                 np.array(image_data, np.float32)
        #                 )
        #             , 0
        #         )
        #         , (0, 3, 1, 2)
        #     )    
a = np.array(image_data1, np.float32)
b = preprocess_input(a)
c = np.expand_dims(b, 0)
image_data  = np.transpose(c, (0, 3, 1, 2))
print(image_data)



