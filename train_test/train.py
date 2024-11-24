import torch
from torch import nn


# 下采样过程
class DownsampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownsampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(out_ch),  # 归一化处理
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        # 相当于池化层处理了
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        """
        :param x:
        :return: out输出到深层，out_2输入到下一层，
        """
        out = self.Conv_BN_ReLU_2(x)
        out_2 = self.downsample(out)
        return out, out_2


# 上采样过程
class UpSampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpSampleLayer, self).__init__()

        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch * 2, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(out_ch * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch * 2, out_channels=out_ch * 2, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(out_ch * 2),
            nn.ReLU()
        )

        self.upsample = nn.Sequential(
            # 逆卷积操作
            nn.ConvTranspose2d(in_channels=out_ch * 2, out_channels=out_ch, kernel_size=(3, 3), stride=(2, 2),
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x, out):
        """
        :param x: 输入卷积层
        :param out:与上采样层进行cat
        :return:
        """
        x_out = self.Conv_BN_ReLU_2(x)
        x_out = self.upsample(x_out)
        cat_out = torch.cat((x_out, out), dim=1)  # 特征矩阵合成
        return cat_out


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # 下采样
        self.d1 = DownsampleLayer(1, 64)
        self.d2 = DownsampleLayer(64, 128)
        self.d3 = DownsampleLayer(128, 256)
        self.d4 = DownsampleLayer(256, 512)
        # 上采样
        self.u1 = UpSampleLayer(512, 512)
        self.u2 = UpSampleLayer(1024, 256)
        self.u3 = UpSampleLayer(512, 128)
        self.u4 = UpSampleLayer(256, 64)

        # 输出
        self.o = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out_1, out1 = self.d1(x)  # out_1 (64, 256, 256) out1 (64, 128,128)
        out_2, out2 = self.d2(out1)  # out_3 (128, 128, 128), out3 (128, 64, 64)
        out_3, out3 = self.d3(out2)  # out_4 (256, 64, 64) out4 (256, 32, 32)
        out_4, out4 = self.d4(out3)  # out_5 (512, 32, 32) out5 (512, 16, 16) 其中 out5不参与下一级的上采样了

        out5 = self.u1(out4, out_4)  # out6 (512,16,16)->(1024,16,16)->(512,32,32)
        out6 = self.u2(out5, out_3)  # out7 (1024, 32, 32)->(512,64,64)->(256,64,64)
        out7 = self.u3(out6, out_2)  # out8 (512, 64, 64)->(256,128,128)->(128,128,128)
        out8 = self.u4(out7, out_1)  # out9 (256,128,128)->(128,256,256)->(64,256,256)
        out = self.o(out8)  # (1, 256 ,256)
        return out


# 读取数据,数据都是2D的npz文件
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


# 显示图像和掩码
def show_image_and_mask(image, mask):
    plt.figure(figsize=(12, 6))

    # 显示图像
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Image')
    plt.axis('off')

    # 显示掩码
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.axis('off')

    plt.show()


class MyDataset(Dataset):
    def __init__(self, data_path, modes='train'):
        if modes == 'train':  # 取70%的数据作为训练集
            self.data_list = os.listdir(data_path)
        elif modes == 'val':  # 取20%的数据作为验证集
            self.data_list = os.listdir(data_path)
        else:  # 取10%的数据作为测试集
            self.data_list = os.listdir(data_path)
        self.data_path = data_path

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.data_path, self.data_list[idx]))
        # print(os.path.join(self.data_path, self.data_list[idx]))
        img = data['image']
        mask = data['label']

        img = np.array(img)
        mask = np.array(mask)

        # print(type(img),type(mask))  # <class 'numpy.ndarray'> <class 'numpy.ndarray'>

        # # img转换到-1到0之间
        img = (img - np.min(img)) / (np.max(img) - np.min(img))


        img = Image.fromarray(img)
        mask = Image.fromarray(mask)

        def mask_to_tensor(mask, size=(128, 128)):
            mask = np.array(mask)  # 转换为numpy数组
            mask = torch.tensor(mask, dtype=torch.float32)  # 转换为PyTorch张量
            mask = mask.unsqueeze(0)  # 增加通道维度
            return mask

        # print(type(img),type(mask))  # <class 'PIL.Image.Image'> <class 'PIL.Image.Image'>
        transform = transforms.Compose([
            transforms.Resize((128, 128))
        ])
        transform1 = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        img = transform1(img)
        mask = transform(mask)
        mask = mask_to_tensor(mask)

        # print('image',np.array(img).max(),np.array(img).min())
        # print('mask',np.array(mask).max(),np.array(mask).min())
        return img, mask


# 数据文件夹

train_path = r'F:\MetaStyleDate\BRATS-2018\processed_2d_train_bezier_newnum\t2'
# val_path = './processed_2d_val_bezier/t1ce'
# test_path = './processed_2d_test_bezier/t1ce'

print('train')
taindata = MyDataset(train_path, modes='train')
# print('val')
# valdata = MyDataset(val_path, modes='val')
# print('test')
# testdata = MyDataset(test_path, modes='test')

batch_size = 128

train_dataloader = DataLoader(taindata, batch_size=batch_size, shuffle=True)
# val_dataloader = DataLoader(valdata, batch_size=batch_size, shuffle=False)
# test_dataloader = DataLoader(testdata, batch_size=batch_size, shuffle=False)
for i, (img, mask) in enumerate(train_dataloader):
    print(img.size(), mask.size())
    break

# 训练模型
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable


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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet().to(device)
criterion = DiceLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

loss_train = []
loss_val = []
dice_train = []
dice_val = []

for epoch in range(20):
    loss_list = []
    dice_list = []
    # 训练
    net.train()
    for i, (img, mask) in enumerate(train_dataloader):
        img, mask = img.to(device), mask.to(device)

        # 可视化
        # plt.subplot(121)
        # plt.imshow(img[0][0].cpu().detach().numpy(), cmap='gray')
        # plt.subplot(122)
        # plt.imshow(mask[0][0].cpu().detach().numpy(), cmap='gray')
        # plt.show()

        optimizer.zero_grad()
        pred = net(img)  # 输入input_var而不是img，因为需要计算梯度，所以需要Variable

        loss = criterion(pred, mask)
        loss_list.append(loss.item())

        dice_score = dice(pred, mask)
        dice_list.append(dice_score.cpu().detach().numpy())

        loss.backward()

        optimizer.step()

    loss = np.mean(loss_list)
    dice_score = np.mean(dice_list)
    loss_train.append(loss)
    dice_train.append(dice_score)
    print('epoch:{}, loss:{}, dice:{}'.format(epoch, loss, dice_score))

    loss_list = []
    dice_list = []


torch.save(net.state_dict(), './unet-t2-newnum.pth')