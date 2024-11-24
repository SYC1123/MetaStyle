import torch
from torch.utils.data import DataLoader
import medpy.metric.binary as mmb
from code.dataloader.dataloader import MyBraTSDataset
from code.model.test.meta_unet import UNet
import numpy as np
import os
import matplotlib.pyplot as plt


def dice(pred, mask):
    pred = pred.view(-1)
    mask = mask.view(-1)
    intersection = (pred * mask).sum()
    return (2. * intersection) / (pred.sum() + mask.sum() + 1e-6)


def dice_coefficient(pred_y, mask, smooth=1e-6):
    """
    计算二分类任务的 Dice 系数

    参数:
    - pred_y: 模型预测结果，形状为 (batch_size, height, width)
    - mask: 真实标签，形状为 (batch_size, height, width)
    - smooth: 防止除零的小平滑项

    返回:
    - Dice 系数 (标量)
    """
    # 确保输入是浮点数类型
    pred_y = pred_y.astype(np.float32)
    mask = mask.astype(np.float32)

    # 计算交集和总面积
    intersection = np.sum(pred_y * mask, axis=(1, 2))  # 对每个样本计算交集
    pred_sum = np.sum(pred_y, axis=(1, 2))  # 对每个样本计算预测总面积
    mask_sum = np.sum(mask, axis=(1, 2))  # 对每个样本计算真实总面积

    # 计算每个样本的 Dice 系数
    dice = (2.0 * intersection + smooth) / (pred_sum + mask_sum + smooth)

    # 返回平均 Dice 系数
    return np.mean(dice)



batch_size = 16

# 测试
domain_test = 'flair'
test_path = f'F:\MetaStyleDate\BRATS-2018\processed_2d_test_bezier\{domain_test}'
test_case = os.listdir(test_path)
testdataset = MyBraTSDataset(test_path, test_case, mode='test')
testdataloder = DataLoader(testdataset, batch_size=batch_size, shuffle=True, drop_last=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 初始化模型、优化器和损失函数
model = UNet().to(device)

# 加载模型
model.load_state_dict(torch.load('./meta_style_unet_newnum_50.pth', map_location=device))

# 测试
model.eval()
total_dice = 0
total_hd = 0
total_asd = 0

dice_list = []
loss_list = []
# 验证
with torch.no_grad():
    for idx, (img, mask) in enumerate(testdataloder):
        img = img.to(device)
        # mask转ndarray
        mask = mask.squeeze(1)
        mask = mask.cpu().detach().numpy()
        _, output = model(img)

        pred_y = torch.sigmoid(output)  # 如果使用 sigmoid，保留原始输出
        pred_y = torch.argmax(pred_y, dim=1)
        pred_y = pred_y.cpu().detach().numpy()

        # print(f"pred_y_shape:{pred_y.shape}")
        # print(f"mask_shape:{mask.shape}")

        dice_score = dice_coefficient(pred_y, mask)
        dice_list.append(dice_score)
dice_mean = np.mean(dice_list)
print(f"mean_dice:{dice_mean}")
        # 可视化
        # plt.imshow(pred_y[0], cmap='gray')
        # plt.show()
        # plt.imshow(mask[0], cmap='gray')
        # plt.show()
        # break
        # print(np.unique(pred_y))
        # if pred_y.sum() == 0 or mask.sum() == 0:
        #     total_dice += 0
        #     total_hd += 100
        #     total_asd += 100
        # else:
        #     total_dice += mmb.dc(pred_y, mask)
        #     total_hd += mmb.hd95(pred_y, mask)
        #     total_asd += mmb.asd(pred_y, mask)
        #
        # # 记录日志
        # print('Domain: {}, Dice: {}, HD: {}, ASD: {}'.format(
        #     domain_test,
        #     round(100 * total_dice / (idx + 1), 2),
        #     round(total_hd / (idx + 1), 2),
        #     round(total_asd / (idx + 1), 2)
        # ))
