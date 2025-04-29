""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ops import *
# from code.util.losses import dice_loss
# from meta_style import StyleFeatureBank
from StyleFeatureBank import StyleFeatureBank

# 钩子函数定义（用于自动提取统计）
def hook_fn(module, input, output, style_bank, layer_name):
    mean, std = style_bank.compute_statistics(output)  # 使用 StyleFeatureBank 的 compute_statistics
    style_bank.add_statistics(layer_name, mean, std)


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
        self.mode='meta-train'
        self.epoch_fill=30
        self.style_bank=StyleFeatureBank()  # 初始化风格特征库
        self.style_bank_old=StyleFeatureBank()  # 初始化旧风格特征库

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

        # 注册钩子到指定层（例如 convd2 和 convd3），可选。如果启用，钩子会自动提取统计
        # self.hook_handles = []
        # self.hook_handles.append(self.convd2.register_forward_hook(
        #     lambda module, input, output: hook_fn(module, input, output, self.style_bank, 'convd2') if self.mode != 'get_source' else None))
        # self.hook_handles.append(self.convd3.register_forward_hook(
        #     lambda module, input, output: hook_fn(module, input, output, self.style_bank, 'convd3')if self.mode != 'get_source' else None))

    def forward(self, x, task_id=None,epoch=None,meta_loss=None, meta_step_size=0.05, stop_gradient=False):
        if self.mode in ['meta-train' , 're-train', 'get_source']:
            # 下采样路径
            x1 = self.convd1(x, meta_loss, meta_step_size, stop_gradient)
            x2 = self.convd2(x1, meta_loss, meta_step_size, stop_gradient)
            # 元训练特征统计量加入特征bank
            if self.mode=='meta-train' and epoch>=self.epoch_fill:
                mean, std = self.style_bank.compute_statistics(x2)  # 手动计算 x2 统计（可选，如果钩子已覆盖，可以注释）
                if task_id==0:
                    self.style_bank.add_statistics('convd2', mean, std)  # 添加到 style_bank（使用自定义层名避免冲突）
                if task_id!= 0 and epoch>self.epoch_fill:
                    # 获取并混合统计：使用 style_bank 的方法
                    # print(f'任务 {task_id} ,使用旧风格特征库')
                    mean_vector, std_vector = self.style_bank_old.get_vector('convd2')  # 从 convd2 层获取；可以扩展到其他层
                    mixed_mean, mixed_std = self.style_bank_old.mix_statistics(mean, std, mean_vector,std_vector)
                    # 应用混合统计
                    x2 = self.style_bank_old.apply_meta_style(x2, mixed_mean, mixed_std)
            x3 = self.convd3(x2, meta_loss, meta_step_size, stop_gradient)
            x4 = self.convd4(x3, meta_loss, meta_step_size, stop_gradient)
            x5 = self.convd5(x4, meta_loss, meta_step_size, stop_gradient)

            # 上采样路径
            y4 = self.convu4(x5, x4, meta_loss, meta_step_size, stop_gradient)
            y3 = self.convu3(y4, x3, meta_loss, meta_step_size, stop_gradient)
            y2 = self.convu2(y3, x2, meta_loss, meta_step_size, stop_gradient)
            y1 = self.convu1(y2, x1, meta_loss, meta_step_size, stop_gradient)

            seg_out = conv2d(y1, self.seg1.weight, self.seg1.bias, meta_loss=meta_loss,
                             meta_step_size=meta_step_size, stop_gradient=stop_gradient, kernel_size=None,
                             stride=1, padding=0)
            seg_out=self.o(seg_out)

            return x2, seg_out

        elif self.mode == 'meta-test':
            # 下采样路径
            x1 = self.convd1(x, meta_loss, meta_step_size, stop_gradient)  # 钩子不在这里触发，因为 convd1 没有注册

            # if task_id!= 0:
            #     # 获取并混合统计：使用 style_bank 的方法
            #     mean_vector, std_vector = self.style_bank_old.get_vector('convd2')  # 从 convd2 层获取；可以扩展到其他层
            #     # print('mean_vector:', mean_vector.shape) # [total_batch, 64, 1, 1] [24, 64, 1, 1]
            #     # print('std_vector:', std_vector.shape) # [total_batch, 64, 1, 1] [24, 64, 1, 1]

            x2 = self.convd2(x1, meta_loss, meta_step_size, stop_gradient)  # 钩子会自动捕获 convd2 输出
            if task_id!= 0:
                mean, std = self.style_bank.compute_statistics(x2)  # 手动计算 x2 统计（可选，如果钩子已覆盖，可以注释）
                self.style_bank.add_statistics('convd2', mean, std)  # 添加到 style_bank（使用自定义层名避免冲突）
            # if task_id!= 0:
            #     target_mean, target_std = self.style_bank_old.compute_statistics(x2)  # 计算当前X2的统计作为目标
            #     # print('target_mean:', target_mean.shape) # [batch/2, 64, 1, 1] [4, 64, 1, 1]
            #     # print('target_std:', target_std.shape) # [batch/2, 64, 1, 1] [4, 64, 1, 1]
            #     mixed_mean, mixed_std = self.style_bank_old.mix_statistics(target_mean, target_std, mean_vector, std_vector)
            #     # print('mixed_mean:', mixed_mean.shape) # [batch/2, 64, 1, 1] [4, 64, 1, 1]
            #     # print('mixed_std:', mixed_std.shape) # [batch/2, 64, 1, 1] [4, 64, 1, 1]
            #     # 应用混合统计
            #     x2 = self.style_bank_old.apply_meta_style(x2, mixed_mean, mixed_std)
            #
            # if task_id!= 0:
            #     mean_vector, std_vector = self.style_bank_old.get_vector('convd3')  # 假设从 convd2 层获取；可以扩展到其他层
                # print('mean_vector:', mean_vector.shape) # [total_batch, 128, 1, 1] [24, 128, 1, 1]
                # print('std_vector:', std_vector.shape)  # [total_batch, 128, 1, 1] [24, 128, 1, 1]
            # meta_loss=None

            x3 = self.convd3(x2, meta_loss, meta_step_size, stop_gradient)  # 钩子会自动捕获 convd3 输出

            # if task_id!= 0:
            #     target_mean, target_std = self.style_bank_old.compute_statistics(x3)  # 计算当前X3的统计作为目标
            #     # print('target_mean:', target_mean.shape) # [batch/2, 128, 1, 1] [4, 128, 1, 1]
            #     # print('target_std:', target_std.shape) # [batch/2, 128, 1, 1] [4, 128, 1, 1]
            #     mixed_mean, mixed_std = self.style_bank_old.mix_statistics(target_mean, target_std, mean_vector, std_vector)
            #     # print('mixed_mean:', mixed_mean.shape) # [batch/2, 128, 1, 1] [4, 128, 1, 1]
            #     # print('mixed_std:', mixed_std.shape) # [batch/2, 128, 1, 1] [4, 128, 1, 1]
            #     # 应用混合统计
            #     x3 = self.style_bank_old.apply_meta_style(x3, mixed_mean, mixed_std)

            x4 = self.convd4(x3, meta_loss, meta_step_size, stop_gradient)
            x5 = self.convd5(x4, meta_loss, meta_step_size, stop_gradient)

            # 上采样路径
            y4 = self.convu4(x5, x4, meta_loss, meta_step_size, stop_gradient)
            y3 = self.convu3(y4, x3, meta_loss, meta_step_size, stop_gradient)
            y2 = self.convu2(y3, x2, meta_loss, meta_step_size, stop_gradient)
            y1 = self.convu1(y2, x1, meta_loss, meta_step_size, stop_gradient)


            seg_out = conv2d(y1, self.seg1.weight, self.seg1.bias, meta_loss=meta_loss,
                             meta_step_size=meta_step_size, stop_gradient=stop_gradient, kernel_size=None,
                             stride=1, padding=0)
            seg_out=self.o(seg_out)

            return x2, seg_out

        elif self.mode == 'eval':
            # 下采样路径
            x1 = self.convd1(x, meta_loss, meta_step_size, stop_gradient)
            x2 = self.convd2(x1, meta_loss, meta_step_size, stop_gradient)
            x3 = self.convd3(x2, meta_loss, meta_step_size, stop_gradient)
            x4 = self.convd4(x3, meta_loss, meta_step_size, stop_gradient)
            x5 = self.convd5(x4, meta_loss, meta_step_size, stop_gradient)
            # 上采样路径（无 meta 参数）
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

    def save_style_bank(self, path):
        """保存 style_bank 中的统计到文件"""
        self.style_bank.save_style_bank(path)

    def load_style_bank(self, path):
        """加载保存的统计"""
        self.style_bank_old.load_style_bank(path)

    def enable_hooks(self, enable=True):
        """启用或禁用钩子（可选，用于控制是否自动提取统计）"""
        if enable:
            # 注册钩子（如果尚未注册）
            if not self.hook_handles:
                self.hook_handles.append(self.convd2.register_forward_hook(
                    lambda module, input, output: hook_fn(module, input, output, self.style_bank, 'convd2')))
                self.hook_handles.append(self.convd3.register_forward_hook(
                    lambda module, input, output: hook_fn(module, input, output, self.style_bank, 'convd3')))
        else:
            # 移除钩子
            for handle in self.hook_handles:
                handle.remove()
            self.hook_handles = []

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

#
# def dice(pred, mask):
#     pred = pred.view(-1)
#     mask = mask.view(-1)
#     intersection = (pred * mask).sum()
#     return (2. * intersection) / (pred.sum() + mask.sum() + 1e-6)
#
# def compute_mean_and_std(features):
#     mean = features.mean(dim=[2, 3], keepdim=True)  # 均值 [batch_size, channels, 1, 1]
#     std = features.std(dim=[2, 3], keepdim=True)  # 标准差 [batch_size, channels, 1, 1]
#     return mean, std
#
#
# # 动态权重计算
# def compute_dynamic_weight(source_mean, source_std, target_mean, target_std):
#     # 计算源域与目标域间的风格差异
#     delta_mean = torch.abs(source_mean - target_mean).mean()
#     delta_std = torch.abs(source_std - target_std).mean()
#     delta_style = delta_mean + delta_std
#
#     # 使用归一化后的风格差异计算权重 (1 为归一化上限)
#     weight = delta_style / (delta_style + 1e-8)  # 避免除零
#     return weight
#
#
# # 风格对齐损失
# def style_alignment_loss(source_mean, source_std, target_mean, target_std, weight):
#     mean_loss = (source_mean - target_mean).pow(2).mean()
#     std_loss = (source_std - target_std).pow(2).mean()
#     return weight * (mean_loss + std_loss)
