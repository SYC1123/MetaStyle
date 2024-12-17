import torch
from torch.utils.data import DataLoader
import medpy.metric.binary as mmb
from code.dataloader.dataloader import MyBraTSDataset
from code.model.test.meta_unet import UNet
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn as nn

#
#
# import torch
# from torch import nn
#
#
# # 下采样过程
# class DownsampleLayer(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(DownsampleLayer, self).__init__()
#         self.Conv_BN_ReLU_2 = nn.Sequential(
#             nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3), stride=(1, 1), padding=1),
#             nn.BatchNorm2d(out_ch),  # 归一化处理
#             nn.ReLU(),
#             nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(3, 3), stride=(1, 1), padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU()
#         )
#         # 相当于池化层处理了
#         self.downsample = nn.Sequential(
#             nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(3, 3), stride=(2, 2), padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         """
#         :param x:
#         :return: out输出到深层，out_2输入到下一层，
#         """
#         out = self.Conv_BN_ReLU_2(x)
#         out_2 = self.downsample(out)
#         return out, out_2
#
#
# # 上采样过程
# class UpSampleLayer(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(UpSampleLayer, self).__init__()
#
#         self.Conv_BN_ReLU_2 = nn.Sequential(
#             nn.Conv2d(in_channels=in_ch, out_channels=out_ch * 2, kernel_size=(3, 3), stride=(1, 1), padding=1),
#             nn.BatchNorm2d(out_ch * 2),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=out_ch * 2, out_channels=out_ch * 2, kernel_size=(3, 3), stride=(1, 1), padding=1),
#             nn.BatchNorm2d(out_ch * 2),
#             nn.ReLU()
#         )
#
#         self.upsample = nn.Sequential(
#             # 逆卷积操作
#             nn.ConvTranspose2d(in_channels=out_ch * 2, out_channels=out_ch, kernel_size=(3, 3), stride=(2, 2),
#                                padding=1,
#                                output_padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU()
#         )
#
#     def forward(self, x, out):
#         """
#         :param x: 输入卷积层
#         :param out:与上采样层进行cat
#         :return:
#         """
#         x_out = self.Conv_BN_ReLU_2(x)
#         x_out = self.upsample(x_out)
#         cat_out = torch.cat((x_out, out), dim=1)  # 特征矩阵合成
#         return cat_out
#
#
# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()
#         # 下采样
#         self.d1 = DownsampleLayer(1, 64)
#         self.d2 = DownsampleLayer(64, 128)
#         self.d3 = DownsampleLayer(128, 256)
#         self.d4 = DownsampleLayer(256, 512)
#         # 上采样
#         self.u1 = UpSampleLayer(512, 512)
#         self.u2 = UpSampleLayer(1024, 256)
#         self.u3 = UpSampleLayer(512, 128)
#         self.u4 = UpSampleLayer(256, 64)
#
#         # 输出
#         self.o = nn.Sequential(
#             nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=1),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x):
#         out_1, out1 = self.d1(x)  # out_1 (64, 256, 256) out1 (64, 128,128)
#         out_2, out2 = self.d2(out1)  # out_3 (128, 128, 128), out3 (128, 64, 64)
#         out_3, out3 = self.d3(out2)  # out_4 (256, 64, 64) out4 (256, 32, 32)
#         out_4, out4 = self.d4(out3)  # out_5 (512, 32, 32) out5 (512, 16, 16) 其中 out5不参与下一级的上采样了
#
#         out5 = self.u1(out4, out_4)  # out6 (512,16,16)->(1024,16,16)->(512,32,32)
#         out6 = self.u2(out5, out_3)  # out7 (1024, 32, 32)->(512,64,64)->(256,64,64)
#         out7 = self.u3(out6, out_2)  # out8 (512, 64, 64)->(256,128,128)->(128,128,128)
#         out8 = self.u4(out7, out_1)  # out9 (256,128,128)->(128,256,256)->(64,256,256)
#         out = self.o(out8)  # (1, 256 ,256)
#         return out




def dice(pred, mask):
    pred = pred.view(-1)
    mask = mask.view(-1)
    intersection = (pred * mask).sum()
    return (2. * intersection) / (pred.sum() + mask.sum() + 1e-6)


def HD95(pred, mask):
    pred = pred.cpu().detach().numpy()
    mask = mask.cpu().detach().numpy()
    hd95 = mmb.hd95(pred, mask)
    return hd95


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # Flatten the tensors to treat them as 1D vectors
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)

        # Compute the intersection
        intersection = (preds_flat * targets_flat).sum()

        # Compute the Dice coefficient
        dice_coef = (2. * intersection + self.smooth) / (preds_flat.sum() + targets_flat.sum() + self.smooth)

        # Compute the Dice loss
        dice_loss = 1 - dice_coef

        return dice_loss


batch_size = 16

# 测试
domain_test = 'flair'
test_path = f'G:\VS_project\Brats-Demo\processed_2d_test_num_bezier\\{domain_test}'
test_case = os.listdir(test_path)
testdataset = MyBraTSDataset(test_path, test_case, mode='test')
testdataloder = DataLoader(testdataset, batch_size=batch_size, shuffle=True, drop_last=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 初始化模型、优化器和损失函数
model = UNet().to(device)

# 加载模型
model.load_state_dict(torch.load('./new_meta_style_unet_40.pth', map_location=device))
criterion = DiceLoss()

# 测试
model.eval()

dice_list = []
hd95_list = []
loss_list = []
dice_val = []
loss_val = []

# 验证
model.eval()
with torch.no_grad():
    for i, (img, mask) in enumerate(testdataloder):
        img, mask = img.to(device), mask.to(device)
        _,pred = model(img)
        loss = criterion(pred, mask)
        loss_list.append(loss.item())
        dice_score = dice(pred, mask)
        hd95 = HD95(pred, mask)
        hd95_list.append(hd95)
        dice_list.append(dice_score.cpu().detach().numpy())
loss = np.mean(loss_list)
dice_score = np.mean(dice_list)
hd95 = np.mean(hd95_list)
loss_val.append(loss)
dice_val.append(dice_score)
print('loss:{}, dice:{}, hd95:{}'.format(loss, dice_score, hd95))

# flair loss:0.49074236907457053, dice:0.5092337131500244, hd95:4.94004208490208