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

from train_FM_1.FM2_dataloader import MyBraTSDataset1


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


def compute_feedback_metrics(model, tasks, source_metric):
    """
    计算元测试反馈信号
    这里测试的是源非相似域的性能
    """
    feedback_signals = {0: [], 1: [], 2: []}
    domain_gaps = {0: [], 1: [], 2: []}
    model.eval()
    with torch.no_grad():
        for data1, data2, data3 in zip(tasks[0], tasks[1], tasks[2]):
            img3, label = data1
            img4, label = data2
            img5, label = data3

            label = label.to(device)
            img3 = img3.to(device)
            _, pred = model(img3)
            dice_score = dice(pred, label)
            feedback_signals[0].append(dice_score.cpu().detach().numpy())

            img4 = img4.to(device)
            _, pred = model(img4)
            dice_score = dice(pred, label)
            feedback_signals[1].append(dice_score.cpu().detach().numpy())

            img5 = img5.to(device)
            _, pred = model(img5)
            dice_score = dice(pred, label)
            feedback_signals[2].append(dice_score.cpu().detach().numpy())

    feedback_signals = {k: np.mean(v) for k, v in feedback_signals.items()}

    # 跨域性能偏差
    # avg_target_metric = sum(feedback_signals.values()) / len(feedback_signals)
    # print("Average target metric:", avg_target_metric)
    # print("Source metric:", source_metric)
    # 计算源域与每一个目标域之间的性能差距
    for k, v in feedback_signals.items():
        domain_gaps[k] = source_metric - v
    return feedback_signals, domain_gaps


def compute_source_metric(model, source_dataloader):
    """
    计算源域性能
    """
    loss_val = []
    dice_val = []
    loss_list = []
    dice_list = []
    model.eval()
    with torch.no_grad():
        for i, (img, mask) in enumerate(source_dataloader):
            img, mask = img.to(device), mask.to(device)
            _, pred = model(img)
            loss = criterion(pred, mask)
            loss_list.append(loss.item())
            dice_score = dice(pred, mask)
            dice_list.append(dice_score.cpu().detach().numpy())
    loss = np.mean(loss_list)
    dice_score = np.mean(dice_list)
    loss_val.append(loss)
    dice_val.append(dice_score)
    # 写入log.txt
    # with open('log.txt', 'a') as f:
    #     f.write(f"Source metric: {dice_score:.4f}\n")
    # print('val—Epoch:{}, loss:{}, dice:{}'.format(epoch, loss, dice_score))
    return dice_score


def reinforce_training(model, optimizer, traindataloder, total_loss, w_style, domain_gap, eta=0.01):
    """
    根据反馈信号强化训练
    """
    model.train()
    # w_style的梯度被释放了，需要重新添加梯度
    w_style = w_style.clone().detach().requires_grad_(True).to(device)
    adjusted_w_style = w_style + eta * domain_gap  # 动态调整风格对齐权重

    loss_list = []
    dice_list = []
    for i, (img0, img1, img2, img3, img4, img5, label) in enumerate(
            tqdm(traindataloder, total=len(traindataloder), desc=f'Epoch {epoch + 1}')):
        # optimizer.zero_grad()
        # 获取源域特征
        source_data = img0.to(device)
        label = label.to(device)

        # meta-test部分
        model.mode = 'meta-test'
        # 获取源域特征
        total_loss = total_loss.clone().detach().requires_grad_(True).to(device)
        # print('total_loss:', total_loss)
        # print(type(total_loss))
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
        style_loss1 = style_alignment_loss(source_mean, source_std, target_mean, target_std, dynamic_weight)
        total_style_loss += style_loss1

        # 不相似域
        # for dissimilar_domain in images[4]:
        dissimilar_domain = img4.to(device)
        target_features, _ = model(dissimilar_domain, mode='meta-test', meta_loss=total_loss)
        target_mean, target_std = compute_mean_and_std(target_features)

        # 动态计算权重
        dynamic_weight = compute_dynamic_weight(source_mean, source_std, target_mean, target_std)

        # 计算风格对齐损失
        style_loss1 = style_alignment_loss(source_mean, source_std, target_mean, target_std, dynamic_weight)
        total_style_loss += style_loss1

        # 分割任务损失 (使用 BCE 作为示例)
        # segmentation_loss = dice_loss(segmentation_output, label)
        segmentation_loss1 = criterion(segmentation_output, label)

        dice_score = dice(segmentation_output, label)
        dice_list.append(dice_score.cpu().detach().numpy())

        # 总损失
        total_loss1 = segmentation_loss1 + total_style_loss * adjusted_w_style
        loss_list.append(total_loss1.item())

        optimizer.zero_grad()
        total_loss1.backward()
        optimizer.step()

    loss = np.mean(loss_list)
    dice_score = np.mean(dice_list)
    # 写入log.txt
    with open('log.txt', 'a') as f:
        f.write(f"ReTrain—Epoch {epoch + 1}, Total Loss: {loss:.4f}, Dice Score: {dice_score:.4f}\n")
    # print(f"train—Epoch {epoch + 1}, Total Loss: {loss:.4f}, Dice Score: {dice_score:.4f}")

    return adjusted_w_style


def train_one_epoch(model, optimizer, traindataloder, epoch):
    model.train()
    loss_list = []
    dice_list = []
    dynamic_weight = 0.0
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

        # print('total_loss:', total_loss)
        # print(type(total_loss))

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

        # optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    loss = np.mean(loss_list)
    # loss = torch.mean(torch.stack(loss_list))
    dice_score = np.mean(dice_list)

    if (epoch + 1) % 10 == 0:
        # 保存模型
        torch.save(model.state_dict(), f'./meta_style_unet_{epoch + 1}.pth')
    # print(f"Epoch {epoch + 1}, Total Loss: {loss:.4f}")
    # 写入log.txt
    with open('log.txt', 'a') as f:
        f.write(f"train—Epoch {epoch + 1}, Total Loss: {loss:.4f}, Dice Score: {dice_score:.4f}\n")
    # print(f"train—Epoch {epoch + 1}, Total Loss: {loss:.4f}, Dice Score: {dice_score:.4f}")
    return total_loss, dynamic_weight


def train_one_epoch_1(model, optimizer, epoch, sourcetraindataloader, task_id, traindataloder=None):
    model.train()
    loss_list = []
    dice_list = []
    # 写入log.txt
    with open('log.txt', 'a') as f:
        f.write(f"当前开始训练第{task_id}个任务\n")

    if traindataloder is None:  # 当前只有源域
        for i, (img0, label) in enumerate(
                tqdm(sourcetraindataloader, total=len(sourcetraindataloader), desc=f'Epoch {epoch + 1}')):
            # TODO: 检查梯度
            optimizer.zero_grad()
            # 获取源域特征
            source_data = img0.to(device)
            label = label.to(device)
            source_features, segmentation_output = model(source_data)
            source_mean, source_std = compute_mean_and_std(source_features)  # 得到源域的均值和方差

            total_style_loss = 0.0

            segmentation_loss = criterion(segmentation_output, label)
            total_loss = segmentation_loss.clone()

            # 该部分是meta-test
            model.mode = 'meta-test'
            source_features, segmentation_output = model(source_data, mode='meta-test', meta_loss=total_loss)
            meta_loss = criterion(segmentation_output, label)

            dice_score = dice(segmentation_output, label)
            dice_list.append(dice_score.cpu().detach().numpy())

            loss_list.append(meta_loss.item())

            meta_loss.backward()
            optimizer.step()

    else:  # 当前有源域和目标域
        for source, target in zip(sourcetraindataloader, traindataloder):
            optimizer.zero_grad()
            img0, label = source
            img1, label1 = target
            # 获取源域特征
            source_data = img0.to(device)
            label = label.to(device)
            source_features, _ = model(source_data)
            source_mean, source_std = compute_mean_and_std(source_features)  # 得到源域的均值和方差

            total_style_loss = 0.0
            # for target_domain:
            target_data = img1.to(device)
            target_features, segementation_output = model(target_data)
            target_mean, target_std = compute_mean_and_std(target_features)

            # 动态计算权重
            # 当前域和源域之间的偏差权重
            dynamic_weight = compute_dynamic_weight(source_mean, source_std, target_mean, target_std)

            # 计算风格对齐损失
            style_loss = style_alignment_loss(source_mean, source_std, target_mean, target_std, dynamic_weight)
            total_style_loss += style_loss

            # 分割任务损失
            segmentation_loss = criterion(segementation_output, label)

            # 总损失
            total_loss = segmentation_loss + total_style_loss

            # 该部分是meta-test
            model.mode = 'meta-test'
            source_features, segmentation_output = model(source_data, mode='meta-test', meta_loss=total_loss)
            meta_loss = criterion(segmentation_output, label)

            dice_score = dice(segmentation_output, label)
            dice_list.append(dice_score.cpu().detach().numpy())

            loss_list.append(meta_loss.item())
            meta_loss.backward()
            optimizer.step()

    loss = np.mean(loss_list)
    dice_score = np.mean(dice_list)
    if (epoch + 1) % 10 == 0:
        # 保存模型
        torch.save(model.state_dict(), f'./meta_style_unet_{epoch + 1}.pth')
    # 写入log.txt
    with open('log.txt', 'a') as f:
        f.write(f"train—Epoch {epoch + 1}, Total Loss: {loss:.4f}, Dice Score: {dice_score:.4f}\n")


def reinforce_training_1(model, optimizer, tasks, domain_gaps, sourcetraindataloader, eta=0.01):
    """
    根据反馈信号强化训练
    """
    model.train()

    for task_id in range(len(tasks)):
        for data1, data2 in zip(sourcetraindataloader, tasks[task_id]):
            img0, label = data1
            img1, label = data2
            optimizer.zero_grad()

            # 获取源域特征
            source_data = img0.to(device)
            label = label.to(device)
            source_features, _ = model(source_data)
            source_mean, source_std = compute_mean_and_std(source_features)  # 得到源域的均值和方差

            total_style_loss = 0.0

            # for target_domain:
            target_data = img1.to(device)
            target_features, segementation_output = model(target_data)
            target_mean, target_std = compute_mean_and_std(target_features)

            # 动态计算权重
            # 当前域和源域之间的偏差权重
            dynamic_weight = compute_dynamic_weight(source_mean, source_std, target_mean, target_std)

            # 计算风格对齐损失
            style_loss = style_alignment_loss(source_mean, source_std, target_mean, target_std, dynamic_weight)
            total_style_loss += style_loss

            # 分割任务损失
            segmentation_loss = criterion(segementation_output, label)

            # 总损失
            total_loss = segmentation_loss + total_style_loss * domain_gaps[task_id]

            total_loss.backward()
            optimizer.step()


if __name__ == '__main__':
    # train_FM()

    batch_size = 8
    domain = 't2'
    train_path = f'G:\VS_project\Brats-Demo\processed_2d_train_num_bezier\{domain}'
    val_path = f'G:\VS_project\Brats-Demo\processed_2d_val_num_bezier\{domain}'

    # train_case是一个列表，0到13291
    train_case = [i for i in range(len(os.listdir(train_path)) // 6)]
    print(f"train_case:{train_case[-1]}")
    train_case = train_case[:100]

    tasks = []
    for domainid in range(6):  # 源域0，相似域1，2，不相似域3，4，5
        traindataset = MyBraTSDataset1(train_path, train_case, domainid, mode='train')
        traindataloder = DataLoader(traindataset, batch_size=batch_size, shuffle=True, drop_last=True)
        tasks.append(traindataloder)

    val_case = os.listdir(val_path)
    val_case = val_case[:50]
    valdataset = MyBraTSDataset(val_path, val_case, mode='val')
    valdataloder = DataLoader(valdataset, batch_size=batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 初始化模型、优化器和损失函数
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = DiceLoss()

    for epoch in range(50):  # 训练 10 个 epoch
        total_loss = 0.0
        for task_id in range(len(tasks)):
            # 第一轮是源域的
            traindataloder = tasks[task_id]
            if task_id == 0:
                train_one_epoch_1(model, optimizer, epoch, traindataloder, task_id, traindataloder=None)
            else:
                train_one_epoch_1(model, optimizer, epoch, tasks[0], task_id, traindataloder)

        source_metric = compute_source_metric(model, valdataloder)
        # 写入log.txt
        with open('log.txt', 'a') as f:
            f.write(f"Source metric: {source_metric}\n")

        feedback_signals, domain_gaps = compute_feedback_metrics(model, tasks[3:], source_metric)
        # 输出反馈信号，写入log.txt
        with open('log.txt', 'a') as f:
            f.write(
                f"Epoch {epoch + 1}, Feedback signals from target domains: {feedback_signals}, Domain gap: {domain_gaps}\n")

        # 强化训练阶段,强化非相似域的训练
        reinforce_training_1(
            model=model,
            optimizer=optimizer,
            tasks=tasks[3:],
            domain_gaps=domain_gaps,
            sourcetraindataloader=tasks[0],
            eta=0.1  # 动态调整学习率
        )
        # # 再次验证泛化能力
        source_metric = compute_source_metric(model, valdataloder)
        # 写入log.txt
        with open('log.txt', 'a') as f:
            f.write(f"Updated source metric: {source_metric}\n")
            f.write("************************************\n")
