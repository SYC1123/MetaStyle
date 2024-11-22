import torch
from torch import nn


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    target = target.squeeze(1)  # 变成 [102, 240, 240]
    
    # 将单通道目标转换为双通道 one-hot 编码
    # 假设 target 中前景为 1，背景为 0
    target_one_hot = torch.zeros_like(score)  # 创建一个与 score 形状相同的全零张量
    # print(f"target_one_hot.shape:{target_one_hot.shape}")
    # print("target.shape:", target.shape)
    target_one_hot[:, 1, ...] = target  # 前景通道为 target
    target_one_hot[:, 0, ...] = 1 - target  # 背景通道为 1 - target

    # 对 score 应用 Sigmoid 激活
    score = torch.sigmoid(score)  # 将预测值映射到 [0, 1]
    
    loss = 0
    for i in range(target_one_hot.shape[1]):
        intersect = torch.sum(score[:, i, ...] * target_one_hot[:, i, ...])
        z_sum = torch.sum(score[:, i, ...] )
        y_sum = torch.sum(target_one_hot[:, i, ...] )
        loss += (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss * 1.0 / target_one_hot.shape[1]

    return loss


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, original, reconstructed):
        return self.mse_loss(reconstructed, original)


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
