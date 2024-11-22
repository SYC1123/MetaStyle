# 读取数据,数据都是2D的npz文件
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


path=r'G:\Pycharm_project\MetaStyle\dataset\BRATS-2018\train_npz_data\t2_ss_train\sample14079_0.npz'
data = np.load(path)
image = data['image']
label = data['label']
print(f"image_shape:{image.shape}")
print(f"label_shape:{label.shape}")