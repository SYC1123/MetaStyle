import os
from collections import OrderedDict

import numpy as np
import torch.utils.data
# import torchsnooper
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

from code.dataloader.dataloader import BraTSDataset, MyBraTSDataset
from code.model.test.meta_unet import UNet
from code.model.unet import Unet2D
from code.util.config import load_config
from sklearn.model_selection import KFold

from code.util.losses import dice_loss, ReconstructionLoss
from code.util.util import save_model
import torchvision.utils as vutils
import torch.optim as optim
import medpy.metric.binary as mmb
import torch
from torch import nn


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


# 计算浅层特征的均值和方差
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


if __name__ == '__main__':
    # train_FM()

    batch_size = 16
    domain = 't2'
    train_path = f'F:\MetaStyleDate\BRATS-2018\processed_2d_train_bezier_num_name\\{domain}'

    # train_case是一个列表，0到13291
    train_case = [i for i in range(13291)]
    print(f"train_case:{train_case[-1]}")


    dataset = MyBraTSDataset(train_path, train_case)
    traindataloder = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 初始化模型、优化器和损失函数
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = DiceLoss()

    loss_train = []
    loss_val = []
    dice_train = []
    dice_val = []

    # 开始训练
    model.train()
    for epoch in range(50):  # 训练 10 个 epoch
        loss_list = []
        dice_list = []
        for i, (img0, img1, img2, img3, img4, img5, label) in enumerate(
                tqdm(traindataloder, total=len(traindataloder), desc=f'Epoch {epoch + 1}')):
            optimizer.zero_grad()
            # 获取源域特征
            source_data = img0.to(device)
            source_features, segmentation_output = model(source_data)
            source_mean, source_std = compute_mean_and_std(source_features)

            # 初始化风格对齐损失
            total_style_loss = 0.0

            # 相似域
            # for similar_domain in images[1]:
            similar_domain = img1.to(device)
            target_features, _ = model(similar_domain)
            target_mean, target_std = compute_mean_and_std(target_features)

            # 动态计算权重
            dynamic_weight = compute_dynamic_weight(source_mean, source_std, target_mean, target_std)

            # 计算风格对齐损失
            style_loss = style_alignment_loss(source_mean, source_std, target_mean, target_std, dynamic_weight)
            total_style_loss += style_loss

            # 不相似域
            # for dissimilar_domain in images[3]:
            dissimilar_domain = img3.to(device)
            target_features, _ = model(dissimilar_domain)
            target_mean, target_std = compute_mean_and_std(target_features)

            # 动态计算权重
            dynamic_weight = compute_dynamic_weight(source_mean, source_std, target_mean, target_std)

            # 计算风格对齐损失
            style_loss = style_alignment_loss(source_mean, source_std, target_mean, target_std, dynamic_weight)
            total_style_loss += style_loss

            # 分割任务损失 (使用 BCE 作为示例)
            # _, segmentation_output = model(source_data)
            label = label.to(device)
            # segmentation_loss = dice_loss(segmentation_output, label)
            segmentation_loss = criterion(segmentation_output, label)

            # 总损失
            total_loss = segmentation_loss + total_style_loss

            # print(f"Epoch {epoch + 1}, Total Loss: {total_loss.item():.4f}, Style Loss: {total_style_loss.item():.4f}")

            # meta-test部分
            model.mode = 'meta-test'
            # 获取源域特征
            # source_data = images[0].to(device)
            source_features, segmentation_output = model(source_data, mode='meta-test', meta_loss=total_loss)
            # print('segmentation_output:', segmentation_output.shape)
            source_mean, source_std = compute_mean_and_std(source_features)

            # 初始化风格对齐损失
            total_style_loss = 0.0

            # 相似域
            # for similar_domain in images[2]:
            similar_domain = img2.to(device)
            target_features, _ = model(similar_domain, mode='meta-test', meta_loss=total_loss)
            target_mean, target_std = compute_mean_and_std(target_features)

            # 动态计算权重
            dynamic_weight = compute_dynamic_weight(source_mean, source_std, target_mean, target_std)

            # 计算风格对齐损失
            style_loss = style_alignment_loss(source_mean, source_std, target_mean, target_std, dynamic_weight)
            total_style_loss += style_loss

            # 不相似域
            # for dissimilar_domain in images[4]:
            dissimilar_domain = img4.to(device)
            target_features, _ = model(dissimilar_domain, mode='meta-test', meta_loss=total_loss)
            target_mean, target_std = compute_mean_and_std(target_features)

            # 动态计算权重
            dynamic_weight = compute_dynamic_weight(source_mean, source_std, target_mean, target_std)

            # 计算风格对齐损失
            style_loss = style_alignment_loss(source_mean, source_std, target_mean, target_std, dynamic_weight)
            total_style_loss += style_loss

            # 分割任务损失 (使用 BCE 作为示例)
            # segmentation_loss = dice_loss(segmentation_output, label)
            segmentation_loss = criterion(segmentation_output, label)

            dice_score = dice(segmentation_output, label)
            dice_list.append(dice_score.cpu().detach().numpy())

            # 总损失
            total_loss = segmentation_loss + total_style_loss
            loss_list.append(total_loss.item())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        loss = np.mean(loss_list)
        dice_score = np.mean(dice_list)
        if (epoch + 1) % 10 == 0:
            # 保存模型
            torch.save(model.state_dict(), f'./meta_style_unet_{epoch + 1}.pth')
        # print(f"Epoch {epoch + 1}, Total Loss: {loss:.4f}")
        print(f"Epoch {epoch + 1}, Total Loss: {loss:.4f}, Dice Score: {dice_score:.4f}")