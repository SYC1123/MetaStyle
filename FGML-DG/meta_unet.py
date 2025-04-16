""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ops import *
# from code.util.losses import dice_loss
from meta_style import StyleFeatureBank


# from model.meta.ops import *

class ConvD(nn.Module):
    def __init__(self, inplanes, planes, norm='in', first=False):
        super(ConvD, self).__init__()

        # self.meta_loss = meta_loss
        # self.meta_step_size = meta_step_size
        # self.stop_gradient = stop_gradient

        self.first = first

        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, bias=True)
        self.in1 = normalization(planes, norm)

        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.in2 = normalization(planes, norm)

        self.conv3 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.in3 = normalization(planes, norm)

    def forward(self, x, meta_loss=None, meta_step_size=0.001, stop_gradient=False):
        if not self.first:
            x = maxpool2D(x, kernel_size=2)

        # layer 1 conv, in
        x = conv2d(x, self.conv1.weight, self.conv1.bias, stride=1, padding=1, meta_loss=meta_loss,
                   meta_step_size=meta_step_size, stop_gradient=stop_gradient)
        x = self.in1(x)

        # layer 2 conv, in, lrelu
        y = conv2d(x, self.conv2.weight, self.conv2.bias, stride=1, padding=1, meta_loss=meta_loss,
                   meta_step_size=meta_step_size, stop_gradient=stop_gradient)
        y = self.in2(y)
        y = lrelu(y)

        # layer 3 conv, in, lrelu
        z = conv2d(y, self.conv3.weight, self.conv3.bias, stride=1, padding=1, meta_loss=meta_loss,
                   meta_step_size=meta_step_size, stop_gradient=stop_gradient)
        z = self.in3(z)
        z = lrelu(z)

        return z


class ConvU(nn.Module):
    def __init__(self, planes, norm='in', first=False):
        super(ConvU, self).__init__()

        # self.meta_loss = meta_loss
        # self.meta_step_size = meta_step_size
        # self.stop_gradient = stop_gradient

        self.first = first
        if not self.first:
            self.conv1 = nn.Conv2d(2 * planes, planes, 3, 1, 1, bias=True)
            self.in1 = normalization(planes, norm)

        self.conv2 = nn.Conv2d(planes, planes // 2, 1, 1, 0, bias=True)
        self.in2 = normalization(planes // 2, norm)

        self.conv3 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.in3 = normalization(planes, norm)

    def forward(self, x, prev, meta_loss=None, meta_step_size=0.001, stop_gradient=False):
        # layer 1 conv, in, lrelu
        if not self.first:
            x = conv2d(x, self.conv1.weight, self.conv1.bias, stride=1, padding=1, meta_loss=meta_loss,
                       meta_step_size=meta_step_size, stop_gradient=stop_gradient)
            x = self.in1(x)
            x = lrelu(x)

        # upsample, layer 2 conv, bn, relu
        y = upsample(x)
        y = conv2d(y, self.conv2.weight, self.conv2.bias, stride=1, padding=0, meta_loss=meta_loss,
                   meta_step_size=meta_step_size, stop_gradient=stop_gradient)
        y = self.in2(y)
        y = lrelu(y)

        # concatenation of two layers
        y = torch.cat([prev, y], 1)

        # layer 3 conv, bn
        y = conv2d(y, self.conv3.weight, self.conv3.bias, stride=1, padding=1, meta_loss=meta_loss,
                   meta_step_size=meta_step_size, stop_gradient=stop_gradient)
        y = self.in3(y)
        y = lrelu(y)

        return y


class UNet(nn.Module):
    def __init__(self, c=1, n=32, num_classes=1, norm='in'):
        super(UNet, self).__init__()
        self.style_bank = StyleFeatureBank()
        self.mode='meta-train'

        # meta_loss = None
        # meta_step_size = 0.001
        # stop_gradient = False

        self.convd1 = ConvD(c, n, norm, first=True)
        self.convd2 = ConvD(n, 2 * n, norm)
        self.convd3 = ConvD(2 * n, 4 * n, norm)
        self.convd4 = ConvD(4 * n, 8 * n, norm)
        self.convd5 = ConvD(8 * n, 16 * n, norm)

        self.convu4 = ConvU(16 * n, norm, first=True)
        self.convu3 = ConvU(8 * n, norm)
        self.convu2 = ConvU(4 * n, norm)
        self.convu1 = ConvU(2 * n, norm)

        # 分割输出
        self.seg1 = nn.Conv2d(2 * n, 128, 1)

        # 输出
        self.o = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x, meta_loss=None, meta_step_size=0.001, stop_gradient=False):
        # self.meta_loss = meta_loss
        # self.meta_step_size = meta_step_size
        # self.stop_gradient = stop_gradient
        # 前两个先不加

        x1 = self.convd1(x)
        x2 = self.convd2(x1)
        x3 = self.convd3(x2, meta_loss, meta_step_size, stop_gradient)
        x4 = self.convd4(x3, meta_loss, meta_step_size, stop_gradient)
        x5 = self.convd5(x4, meta_loss, meta_step_size, stop_gradient)
        # 获取特征统计量
        mean, sig = self.style_bank.compute_statistics(x2)
        # print(f"mean,sig,x2{mean.shape, sig.shape, x2.shape}")
        # mean, sig, x2(torch.Size([1, 32, 1, 1]), torch.Size([1, 32, 1, 1]), torch.Size([1, 32, 64, 64]))

        if self.mode in ['meta-train' , 're-train', 'get_source']:
            # print(self.mode)
            # 元训练特征统计量加入特征bank
            if self.mode=='meta-train':
                self.style_bank.add_statistics(mean, sig)
            y4 = self.convu4(x5, x4, meta_loss, meta_step_size, stop_gradient)
            y3 = self.convu3(y4, x3, meta_loss, meta_step_size, stop_gradient)
            y2 = self.convu2(y3, x2, meta_loss, meta_step_size, stop_gradient)
            y1 = self.convu1(y2, x1, meta_loss, meta_step_size, stop_gradient)

            seg_out = conv2d(y1, self.seg1.weight, self.seg1.bias, meta_loss=meta_loss,
                             meta_step_size=meta_step_size, stop_gradient=stop_gradient, kernel_size=None,
                             stride=1, padding=0)
            seg_out=self.o(seg_out)
            # seg_out = F.sigmoid(seg_out)

            return x2, seg_out
            # return seg_out

        elif self.mode == 'meta-test':
            # 解码器
            y4 = self.convu4(x5, x4, meta_loss, meta_step_size, stop_gradient)
            y3 = self.convu3(y4, x3, meta_loss, meta_step_size, stop_gradient)
            y2 = self.convu2(y3, x2, meta_loss, meta_step_size, stop_gradient)
            y1 = self.convu1(y2, x1, meta_loss, meta_step_size, stop_gradient)

            # 将元训练阶段保存的风格统计量与当前样本的统计量进行混合
            mean_vector, std_vector = self.style_bank.get_vector()
            # print(
            #     f"mean_vector,std_vector,mean,sig,y1{mean_vector.shape, std_vector.shape, mean.shape, sig.shape, y1.shape}")
            # 随机从 a 中选择 26 个样本
            train_batch_size = mean_vector.shape[0]
            test_batch_size = mean.shape[0]
            idx = torch.randperm(train_batch_size)[:test_batch_size]
            mean_vector = mean_vector[idx, :, :, :]  # 从 a 中随机选择 26 个样本
            std_vector = std_vector[idx, :, :, :]
            mixed_mean, mixed_std = self.style_bank.mix_statistics(mean, sig, mean_vector, std_vector)
            meta_styled_y = self.style_bank.apply_meta_style(y1, mixed_mean, mixed_std)

            seg_out = conv2d(meta_styled_y, self.seg1.weight, self.seg1.bias, meta_loss=meta_loss,
                             meta_step_size=meta_step_size, stop_gradient=stop_gradient, kernel_size=None,
                             stride=1, padding=0)
            seg_out=self.o(seg_out)

            # seg_out = F.sigmoid(seg_out)

            return x2, seg_out

        elif self.mode == 'eval':
            y4 = self.convu4(x5, x4)
            y3 = self.convu3(y4, x3)
            y2 = self.convu2(y3, x2)
            y1 = self.convu1(y2, x1)
            seg_out = self.seg1(y1)
            seg_out = self.o(seg_out)

            return seg_out

        else:
            raise ValueError(f"Invalid mode: {self.mode}. Expected 'meta-train' or 'meta-test'.")

    def set_mode(self, mode):
        self.mode = mode

    # 计算浅层特征的均值和方差
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


def dice(pred, mask):
    pred = pred.view(-1)
    mask = mask.view(-1)
    intersection = (pred * mask).sum()
    return (2. * intersection) / (pred.sum() + mask.sum() + 1e-6)

def compute_mean_and_std(features):
    mean = features.mean(dim=[2, 3], keepdim=True)  # 均值 [batch_size, channels, 1, 1]
    std = features.std(dim=[2, 3], keepdim=True)  # 标准差 [batch_size, channels, 1, 1]
    return mean, std


# 动态权重计算
def compute_dynamic_weight(source_mean, source_std, target_mean, target_std):
    # 计算源域与目标域间的风格差异
    delta_mean = torch.abs(source_mean - target_mean).mean()
    delta_std = torch.abs(source_std - target_std).mean()
    delta_style = delta_mean + delta_std

    # 使用归一化后的风格差异计算权重 (1 为归一化上限)
    weight = delta_style / (delta_style + 1e-8)  # 避免除零
    return weight


# 风格对齐损失
def style_alignment_loss(source_mean, source_std, target_mean, target_std, weight):
    mean_loss = (source_mean - target_mean).pow(2).mean()
    std_loss = (source_std - target_std).pow(2).mean()
    return weight * (mean_loss + std_loss)
