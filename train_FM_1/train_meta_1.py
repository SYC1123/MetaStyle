import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from code.model.test.meta_unet import UNet
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
    # print('delta_mean:', delta_mean)
    delta_std = torch.abs(source_std - target_std).mean()
    # print('delta_std:', delta_std)
    sensitivity=3.0
    # 使用对数进行压缩差异
    delta_mean_log = torch.log1p(delta_mean * sensitivity)  # 对差异使用对数
    delta_std_log = torch.log1p(delta_std * sensitivity)

    # 总的风格差异
    delta_style = delta_mean_log + delta_std_log

    # delta_style = delta_mean + delta_std

    # 使用归一化后的风格差异计算权重
    # print('delta_style:', delta_style)
    weight = 1.0 - torch.exp(-delta_style).cpu().detach().numpy()
    return weight


def style_alignment_loss(source_mean, source_std, target_mean, target_std, weight):
    mean_loss = ((source_mean - target_mean)/(source_mean + target_mean + 1e-6)).pow(2).mean()
    std_loss = ((source_std - target_std)/(source_std + target_std + 1e-6)).pow(2).mean()
    return weight * (mean_loss + std_loss)


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, source_features, target_features, labels):
        """
        :param source_features: 来自源域的特征表示
        :param target_features: 来自目标域的特征表示
        :param labels: 标签，0表示源域和目标域属于同一类别，1表示不同类别
        :return: 对比损失
        """
        # 计算源域和目标域特征之间的欧氏距离
        # print('source_features:', source_features.shape)
        # print('target_features:', target_features.shape)
        source_features_flatten = source_features.view(source_features.size(0), -1)  # 展平为 [batch_size, C*H*W]
        target_features_flatten = target_features.view(target_features.size(0), -1)  # 展平为 [batch_size, C*H*W]
        # 对特征进行L2归一化
        source_features_flatten = F.normalize(source_features_flatten, p=2, dim=1)
        target_features_flatten = F.normalize(target_features_flatten, p=2, dim=1)

        euclidean_distance = F.pairwise_distance(source_features_flatten, target_features_flatten)

        # euclidean_distance = F.pairwise_distance(source_features, target_features, keepdim=True)
        # print('euclidean_distance:', euclidean_distance.shape)
        # print('labels:', labels.shape)
        # 计算对比损失
        loss = 0.5 * (labels.float() * euclidean_distance.pow(2) +
                      (1 - labels.float()) * F.relu(self.margin - euclidean_distance).pow(2))

        return loss.mean()

def calculate_gap(metric):
    return 1-np.exp(np.log10(metric))


def compute_feedback_metrics(model, tasks):
    """
    计算元测试反馈信号
    这里测试的是源非相似域的性能
    """
    feedback_signals = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    domain_gaps = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    model.eval()
    with torch.no_grad():
        for data1, data2, data3,data4,data5,data6 in zip(tasks[0], tasks[1], tasks[2], tasks[3], tasks[4], tasks[5]):
            img1, label = data1
            img2, label = data2
            img3, label = data3
            img4, label = data4
            img5, label = data5
            img6, label = data6

            label = label.to(device)

            img1 = img1.to(device)
            _, pred = model(img1)
            dice_score = dice(pred, label)
            feedback_signals[0].append(dice_score.cpu().detach().numpy())

            img2 = img2.to(device)
            _, pred = model(img2)
            dice_score = dice(pred, label)
            feedback_signals[1].append(dice_score.cpu().detach().numpy())

            img3 = img3.to(device)
            _, pred = model(img3)
            dice_score = dice(pred, label)
            feedback_signals[2].append(dice_score.cpu().detach().numpy())

            img4 = img4.to(device)
            _, pred = model(img4)
            dice_score = dice(pred, label)
            feedback_signals[3].append(dice_score.cpu().detach().numpy())

            img5 = img5.to(device)
            _, pred = model(img5)
            dice_score = dice(pred, label)
            feedback_signals[4].append(dice_score.cpu().detach().numpy())

            img6 = img6.to(device)
            _, pred = model(img6)
            dice_score = dice(pred, label)
            feedback_signals[5].append(dice_score.cpu().detach().numpy())


    feedback_signals = {k: np.mean(v) for k, v in feedback_signals.items()}

    # 计算源域与每一个目标域之间的性能差距
    for k, v in feedback_signals.items():
        domain_gaps[k] = calculate_gap(v)
        # domain_gaps[k] = abs(source_metric - v)
    return feedback_signals, domain_gaps


def compute_source_metric(model, source_dataloader):
    """
    计算源域性能
    """
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
    return dice_score


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
            # # 可视化img0   img1
            # plt.imshow(img0[0][0].cpu().detach().numpy(), cmap='gray')
            # plt.title('img0')
            # plt.show()
            # plt.imshow(img1[0][0].cpu().detach().numpy(), cmap='gray')
            # plt.title('img1')
            # plt.show()
            # 获取源域特征
            source_data = img0.to(device)
            label = label.to(device)
            label1 = label1.to(device)
            with torch.no_grad():
                source_features, _ = model(source_data)
                # # 可视化特征
                # plt.imshow(source_features[0][0].cpu().detach().numpy())
                # plt.title('source_features')
                # plt.show()

                source_mean, source_std = compute_mean_and_std(source_features)  # 得到源域的均值和方差
                # print('source_mean:', source_mean)
                # print('source_std:', source_std)
            #
            # total_style_loss = 0.0
            # for target_domain:
            target_data = img1.to(device)
            target_features, segementation_output = model(target_data)
            # # 可视化特征
            # plt.imshow(target_features[0][0].cpu().detach().numpy())
            # plt.title('target_features')
            # plt.show()

            target_mean, target_std = compute_mean_and_std(target_features)
            # task_id = 1,2 是相似域，3,4,5是不相似域
            if task_id == 1 or task_id == 2:
                labels = torch.ones(source_features.size(0), 1).to(device)
            else:
                labels = torch.zeros(source_features.size(0), 1).to(device)

            # print('target_mean:', target_mean)
            # print('target_std:', target_std)

            # # 动态计算权重
            # # 当前域和源域之间的偏差权重
            dynamic_weight = compute_dynamic_weight(source_mean, source_std, target_mean, target_std)
            # print('dynamic_weight:', dynamic_weight)
            #
            # # 计算风格对齐损失
            # style_loss = style_alignment_loss(source_mean, source_std, target_mean, target_std, dynamic_weight)
            # print('style_loss:', style_loss)
            # total_style_loss += style_loss

            # 对比损失
            style_loss = criterion1(source_features, target_features, labels)
            # print('style_loss:', style_loss)

            # 分割任务损失
            segmentation_loss = criterion(segementation_output, label1)
            # print('segmentation_loss:', segmentation_loss)

            # 总损失
            total_loss = (1-dynamic_weight)*segmentation_loss + dynamic_weight*style_loss


            # 该部分是meta-test
            model.mode = 'meta-test'
            source_features, segmentation_output = model(target_data, mode='meta-test', meta_loss=total_loss)
            meta_loss = criterion(segmentation_output, label1)

            dice_score = dice(segmentation_output, label1)
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
    source_metric = compute_source_metric(model, testdataloder)
    print('source_metric:', source_metric)
    with open('log.txt', 'a') as f:
        f.write(f"train—Epoch {epoch + 1}, Total Loss: {loss:.4f}, Dice Score: {dice_score:.4f}\n")



def reinforce_training(model, optimizer, epoch, tasks,domain_gaps):
    model.train()
    loss_list = []
    dice_list = []

    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    # print(len(tasks[0]))
    for i in range(len(tasks[0])):  # 假设所有DataLoader长度相同
        batch_data = []
        batch_targets = []

        # 根据权重从每个DataLoader中采样数据
        for j, dataloader in enumerate([tasks[0], tasks[1], tasks[2], tasks[3], tasks[4], tasks[5]]):
            if random.random() < domain_gaps[j]:
                data, target = next(iter(dataloader))
                batch_data.append(data)
                batch_targets.append(target)
        if len(batch_data) == 0:
            continue
        batch_data = torch.cat(batch_data, dim=0)
        batch_targets = torch.cat(batch_targets, dim=0)

        optimizer.zero_grad()
        batch_data = batch_data.to(device)
        batch_targets = batch_targets.to(device)

        # print('batch_data:', batch_data.shape)

        # 前向传播
        _,output = model(batch_data,mode='re-train')
        loss = criterion(output, batch_targets)

        dice_score = dice(output, batch_targets)
        dice_list.append(dice_score.cpu().detach().numpy())

        loss_list.append(loss.item())

        # 反向传播和优化
        loss.backward()
        optimizer.step()

    loss = np.mean(loss_list)
    dice_score = np.mean(dice_list)
    if (epoch + 1) % 10 == 0:
        # 保存模型
        torch.save(model.state_dict(), f'./meta_style_unet_re_{epoch + 1}.pth')
    with open('log.txt', 'a') as f:
        f.write(f"retrain—Epoch {epoch + 1},Total Loss: {loss:.4f}, Dice Score: {dice_score:.4f}\n")



if __name__ == '__main__':
    # train_FM()
    with open('log.txt', 'a') as f:
        # 画一个开始的分割线
        f.write("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

    batch_size = 8
    domain = 't2'
    train_path = f'G:\VS_project\Brats-Demo\processed_2d_train_num_bezier\\{domain}'
    val_path = f'G:\VS_project\Brats-Demo\processed_2d_val_num_bezier\\{domain}'
    test_path = f'G:\VS_project\Brats-Demo\processed_2d_test_num_bezier\\{domain}'

    # train_case是一个列表，0到13291
    train_case = [i for i in range(len(os.listdir(train_path)) // 6)]
    print(f"train_case:{train_case[-1]}")
    train_case = train_case[:25]

    tasks = []
    for domainid in range(6):  # 源域0，相似域1，2，不相似域3，4，5
        traindataset = MyBraTSDataset1(train_path, train_case, domainid, mode='train')
        traindataloder = DataLoader(traindataset, batch_size=batch_size, drop_last=True)
        tasks.append(traindataloder)

    val_case = [i for i in range(len(os.listdir(val_path)) // 6)]
    val_case = val_case[:25]

    val_tasks=[]
    for domainid in range(6):
        valdataset = MyBraTSDataset1(val_path, val_case, domainid, mode='val')
        valdataloder = DataLoader(valdataset, batch_size=batch_size, drop_last=True)
        val_tasks.append(valdataloder)
    # valdataset = MyBraTSDataset1(val_path, val_case,None, mode='val')
    # valdataloder = DataLoader(valdataset, batch_size=batch_size,drop_last=True)

    test_case = os.listdir(test_path)
    test_case = test_case[:25]
    testdataset = MyBraTSDataset1(test_path, test_case,None, mode='test')
    testdataloder = DataLoader(testdataset, batch_size=batch_size, drop_last=True)



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 初始化模型、优化器和损失函数
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = DiceLoss()
    # 计算对比损失
    criterion1 = ContrastiveLoss(margin=1.0)

    for epoch in range(50):  # 训练 10 个 epoch
        for task_id in range(len(tasks)):
            # 第一轮是源域的
            traindataloder = tasks[task_id]
            if task_id == 0:
                train_one_epoch_1(model, optimizer, epoch, traindataloder, task_id, traindataloder=None)
            else:
                train_one_epoch_1(model, optimizer, epoch, tasks[0], task_id, traindataloder)

        source_metric = compute_source_metric(model, testdataloder)
        # 写入log.txt
        with open('log.txt', 'a') as f:
            f.write(f"Source metric: {source_metric}\n")

        feedback_signals, domain_gaps = compute_feedback_metrics(model, val_tasks)
        # 输出反馈信号，写入log.txt
        with open('log.txt', 'a') as f:
            f.write(
                f"Epoch {epoch + 1}, Feedback signals from target domains: {feedback_signals}, Domain gap: {domain_gaps}\n")


        # 强化训练阶段,强化非相似域的训练
        for epoch in range(10):
            reinforce_training(model, optimizer, epoch, tasks, domain_gaps)
            _, domain_gaps = compute_feedback_metrics(model, val_tasks)

        # # 再次验证泛化能力
        source_metric = compute_source_metric(model, testdataloder)
        # 写入log.txt
        with open('log.txt', 'a') as f:
            f.write(f"Updated source metric: {source_metric}\n")
            f.write("************************************\n")
        # 保存模型
        torch.save(model.state_dict(), f'./final_model.pth')
