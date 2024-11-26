import torch
from torch.utils.data import DataLoader
import medpy.metric.binary as mmb
from code.dataloader.dataloader import MyBraTSDataset
from code.model.test.meta_unet import UNet
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn as nn


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
domain_test = 't1'
test_path = f'G:\VS_project\Brats-Demo\processed_2d_test_num_bezier\{domain_test}'
test_case = os.listdir(test_path)
testdataset = MyBraTSDataset(test_path, test_case, mode='test')
testdataloder = DataLoader(testdataset, batch_size=batch_size, shuffle=True, drop_last=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 初始化模型、优化器和损失函数
model = UNet().to(device)

# 加载模型
model.load_state_dict(torch.load('./unet_newnum_onechannel_feedback_20.pth', map_location=device))
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
        _, pred = model(img)
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
