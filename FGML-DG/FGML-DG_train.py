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

from meta_unet import UNet
from FM2_dataloader import MyBraTSDataset1


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
    model.set_mode('eval')
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
            pred = model(img1)
            dice_score = dice(pred, label)
            feedback_signals[0].append(dice_score.cpu().detach().numpy())

            img2 = img2.to(device)
            pred = model(img2)
            dice_score = dice(pred, label)
            feedback_signals[1].append(dice_score.cpu().detach().numpy())

            img3 = img3.to(device)
            pred = model(img3)
            dice_score = dice(pred, label)
            feedback_signals[2].append(dice_score.cpu().detach().numpy())

            img4 = img4.to(device)
            pred = model(img4)
            dice_score = dice(pred, label)
            feedback_signals[3].append(dice_score.cpu().detach().numpy())

            img5 = img5.to(device)
            pred = model(img5)
            dice_score = dice(pred, label)
            feedback_signals[4].append(dice_score.cpu().detach().numpy())

            img6 = img6.to(device)
            pred = model(img6)
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
    # loss_list = []
    dice_list = []
    model.eval()
    model.set_mode('eval')
    with torch.no_grad():
        for i, (img, mask) in enumerate(source_dataloader):
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            # loss = criterion(pred, mask)
            # loss_list.append(loss.item())
            dice_score = dice(pred, mask)
            dice_list.append(dice_score.cpu().detach().numpy())
    # loss = np.mean(loss_list)
    dice_score = np.mean(dice_list)
    return dice_score


def train_one_epoch(model, optimizer, epoch, dataloaders,test_dataloader, task_ids, device):
    model.train()
    loss_list = []
    dice_list = []

    # dataloaders 是一个字典，键是 task_id，值是对应的 DataLoader
    # 例如：dataloaders = {0: source_dataloader, 1: domain1_dataloader, ..., 5: domain5_dataloader}
    # task_ids 是一个列表，包含所有任务 ID，例如 [0, 1, 2, 3, 4, 5]

    # 写入 log.txt
    with open('log.txt', 'a') as f:
        f.write(f"Epoch {epoch + 1} 开始训练\n")

    for task_id in task_ids:  # 遍历每个任务（域）
        dataloader = dataloaders[task_id]  # 获取当前任务的 DataLoader
        with open('log.txt', 'a') as f:
            f.write(f"当前任务 ID: {task_id} (域 {task_id})\n")
        # print('当前任务 ID:', task_id)
        if task_id == 0:  # 源域
            for batch_source in tqdm(dataloader, desc=f'Epoch {epoch + 1}, Task {task_id}', total=len(dataloader)):
            # for batch_source in dataloader:
                # 仅使用源域数据
                img, label = batch_source  # 假设批量数据是 (img, label)，img 和 label 都是 tensor
                img = img.to(device)
                label = label.to(device)

                batch_size = img.size(0)

                # 随机分割支持集和查询集（例如，50% 用于支持集，50% 用于查询集）
                # 使用随机索引确保每次分割不同，避免数据泄露
                indices = list(range(batch_size))
                random.shuffle(indices)  # 随机打乱索引
                split_idx = batch_size // 2
                support_indices = indices[:split_idx]
                query_indices = indices[split_idx:]

                support_img = img[support_indices]
                support_label = label[support_indices]
                query_img = img[query_indices]
                query_label = label[query_indices]

                optimizer.zero_grad()

                # print('============= 支持集处理：计算 total_loss =============')
                # 仅使用源域数据，类似于您的原始代码
                model.set_mode('meta-train')
                source_features, seg_output = model(support_img)
                segmentation_loss = criterion(seg_output, support_label)  # 分割损失

                # meta-test 部分在查询集处理，这里先计算 total_loss
                total_loss = segmentation_loss.clone()  # 可以添加其他损失，如果需要

                # print(' ============= 查询集处理：meta-test 模式 =============')
                model.set_mode('meta-test')
                _, meta_seg_output = model(query_img, meta_loss=total_loss)  # 使用查询集数据和 meta_loss
                meta_loss = criterion(meta_seg_output, query_label)  # 计算元损失

                # 计算 Dice 分数
                dice_score = dice(meta_seg_output, query_label)
                dice_list.append(dice_score.cpu().detach().numpy())

                # 记录损失
                loss_list.append(meta_loss.item())

                # 反向传播和优化
                meta_loss.backward()
                optimizer.step()

                # 清零梯度（可选，确保每个批量独立）
                optimizer.zero_grad()
        else:
            # 目标域
            for batch_source,batch_target in tqdm(zip(dataloaders[0],dataloader), desc=f'Epoch {epoch + 1}, Task {task_id}', total=len(dataloader)):
            # for batch_source, batch_target in zip(dataloaders[0], dataloader):
                img_source, label_source = batch_source  # 假设批量数据是 (img, label)，img 和 label 都是 tensor
                img_source = img_source.to(device)
                label_source = label_source.to(device)

                img_target, label_target = batch_target
                img_target = img_target.to(device)
                label_target = label_target.to(device)

                batch_size = img_source.size(0)

                # 随机分割支持集和查询集（例如，50% 用于支持集，50% 用于查询集）
                # 使用随机索引确保每次分割不同，避免数据泄露
                indices = list(range(batch_size))
                random.shuffle(indices)  # 随机打乱索引
                split_idx = batch_size // 2
                support_indices = indices[:split_idx]
                query_indices = indices[split_idx:]

                support_img_source = img_source[support_indices]
                support_label_source = label_source[support_indices]
                query_img_source = img_source[query_indices]
                query_label_source = label_source[query_indices]

                support_img_target = img_target[support_indices]
                support_label_target = label_target[support_indices]
                query_img_target = img_target[query_indices]
                query_label_target = label_target[query_indices]

                optimizer.zero_grad()
                # print(' ============= 支持集处理：计算 total_loss =============')
                # 获取源域特征
                with torch.no_grad():
                    model.set_mode('get_source')
                    source_features, _ = model(support_img_source)
                    source_mean, source_std = compute_mean_and_std(source_features)
                model.set_mode('meta-train')
                # 计算目标域特征（使用支持集的同一数据，但作为目标域模拟）
                target_features, seg_output = model(support_img_target)  # 或使用一个变换后的版本，如果可用

                target_mean, target_std = compute_mean_and_std(target_features)

                # 根据 task_id 调整标签（相似域 vs 不相似域）
                if task_id in [1, 2]:  # 相似域
                    labels = torch.ones(source_features.size(0), 1).to(device)  # 相似，标签为 1
                else:  # 不相似域 (task_id 3,4,5)
                    labels = torch.zeros(source_features.size(0), 1).to(device)  # 不相似，标签为 0

                # 计算风格对齐损失或对比损失
                style_loss = criterion1(source_features, target_features, labels)  # 对比损失

                # 分割损失
                segmentation_loss = criterion(seg_output, support_label_target)

                # 动态权重计算（根据域偏差）
                dynamic_weight = compute_dynamic_weight(source_mean, source_std, target_mean, target_std)

                # 总损失：结合分割损失和风格损失，权重根据动态权重调整
                total_loss = (1 - dynamic_weight) * segmentation_loss + dynamic_weight * style_loss

                # print(' ============= 查询集处理：meta-test 模式 =============')
                model.set_mode('meta-test')
                _, meta_seg_output = model(query_img_target, meta_loss=total_loss)  # 使用查询集数据和 meta_loss
                meta_loss = criterion(meta_seg_output, support_label_target)  # 计算元损失

                # 计算 Dice 分数
                dice_score = dice(meta_seg_output, support_label_target)
                dice_list.append(dice_score.cpu().detach().numpy())

                # 记录损失
                loss_list.append(meta_loss.item())

                # 反向传播和优化
                meta_loss.backward()
                optimizer.step()

                # 清零梯度（可选，确保每个批量独立）
                optimizer.zero_grad()
    model.style_bank.clean_style_feature_bank()
    # =============  epoch 结束处理 =============
    loss = np.mean(loss_list)
    dice_score = np.mean(dice_list)

    # 每 10 个 epoch 保存模型
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f'./meta_style_unet_domain_generalization_epoch_{epoch + 1}.pth')

    # 计算源域指标（如果有测试数据加载器）
    # 假设有一个 test_dataloader 用于源域评估
    source_metric = compute_source_metric(model, test_dataloader)  # 需要定义或传入 test_dataloader
    print(f'Epoch {epoch + 1}, Source Metric: {source_metric}')

    # 写入 log.txt
    with open('log.txt', 'a') as f:
        f.write(f"Epoch {epoch + 1}, Average Meta Loss: {loss:.4f}, Average Dice Score: {dice_score:.4f}\n")


def reinforce_training(model, optimizer, epoch, tasks,domain_gaps):
    model.train()
    loss_list = []
    dice_list = []

    model.train()
    model.set_mode('re-train')
    total_loss = 0.0
    optimizer.zero_grad()
    # print(len(tasks[0]))
    for i in range(len(tasks[0])):  # 假设所有DataLoader长度相同
        batch_data = []
        batch_targets = []

        # 根据权重从每个DataLoader中采样数据
        for j, dataloader in enumerate([tasks[0], tasks[1], tasks[2], tasks[3], tasks[4], tasks[5]]):
            task_len=len(dataloader)
            num=int(domain_gaps[j]*task_len)
            if num==0:
                num=1
            for _ in range(num):
                data, target = next(iter(dataloader))
                batch_data.append(data)
                batch_targets.append(target)

        #     if random.random() < domain_gaps[j]:
        #         data, target = next(iter(dataloader))
        #         batch_data.append(data)
        #         batch_targets.append(target)
        # if len(batch_data) == 0:
        #     continue
        batch_data = torch.cat(batch_data, dim=0)
        batch_targets = torch.cat(batch_targets, dim=0)

        optimizer.zero_grad()
        batch_data = batch_data.to(device)
        batch_targets = batch_targets.to(device)

        # print('batch_data:', batch_data.shape)

        # 前向传播
        _,output = model(batch_data)
        loss = criterion(output, batch_targets)

        dice_score = dice(output, batch_targets)
        dice_list.append(dice_score.cpu().detach().numpy())

        loss_list.append(loss.item())

        # 反向传播和优化
        loss.backward()
        optimizer.step()

    loss = np.mean(loss_list)
    dice_score = np.mean(dice_list)
    # if (epoch + 1) % 10 == 0:
    #     # 保存模型
    #     torch.save(model.state_dict(), f'./meta_style_unet_re_{epoch + 1}.pth')
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

    val_case = [i for i in range(len(os.listdir(val_path)) // 6)]
    val_case = val_case[:25]

    test_case = os.listdir(test_path)
    test_case = test_case[:25]


    task_ids = [0, 1, 2, 3, 4, 5]  # 所有任务 ID

    # 训练数据加载器
    dataloaders_train={}
    for domainid in range(6):  # 源域0，相似域1，2，不相似域3，4，5
        traindataset = MyBraTSDataset1(train_path, train_case, domainid, mode='train')
        traindataloder = DataLoader(traindataset, batch_size=batch_size, drop_last=True)
        # tasks.append(traindataloder)
        dataloaders_train[domainid] = traindataloder

    # 验证数据加载器
    dataloaders_val={}
    for domainid in range(6):
        valdataset = MyBraTSDataset1(val_path, val_case, domainid, mode='val')
        valdataloder = DataLoader(valdataset, batch_size=batch_size, drop_last=True)
        dataloaders_val[domainid] = valdataloder

    # 测试数据加载器
    testdataset = MyBraTSDataset1(test_path, test_case,None, mode='test')
    testdataloder = DataLoader(testdataset, batch_size=batch_size, drop_last=True)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 初始化模型、优化器和损失函数
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = DiceLoss()
    # 计算对比损失
    criterion1 = ContrastiveLoss(margin=1.0)

    # 训练循环
    num_epochs = 100
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, epoch, dataloaders_train, testdataloder,task_ids, device)

        # source_metric = compute_source_metric(model, testdataloder)
        # # 写入log.txt
        # with open('log.txt', 'a') as f:
        #     f.write(f"Source metric: {source_metric}\n")

    #     feedback_signals, domain_gaps = compute_feedback_metrics(model, dataloaders_val)
    #     # 输出反馈信号，写入log.txt
    #     with open('log.txt', 'a') as f:
    #         f.write(
    #             f"Epoch {epoch + 1}, Feedback signals from target domains: {feedback_signals}, Domain gap: {domain_gaps}\n")
    #
    #
    #     # 强化训练阶段,强化非相似域的训练
    #     for epoch in range(10):
    #         reinforce_training(model, optimizer, epoch, dataloaders_train, domain_gaps)
    #         _, domain_gaps = compute_feedback_metrics(model, dataloaders_val)
    #
    #     # # 再次验证泛化能力
    #     source_metric = compute_source_metric(model, testdataloder)
    #     # 写入log.txt
    #     with open('log.txt', 'a') as f:
    #         f.write(f"Updated source metric: {source_metric}\n")
    #         f.write("************************************\n")
    # # 保存模型
    # torch.save(model.state_dict(), f'./final_model.pth')
