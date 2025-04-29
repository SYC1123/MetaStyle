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
from StyleFeatureBank import StyleFeatureBank


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
    # print('source_mean:', source_mean.shape)
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
# def compute_dynamic_weight(source_mean, source_std, target_mean, target_std):
#     """
#     计算源域与目标域间的风格差异，并返回动态权重。
#
#     参数:
#     - source_mean: torch.Tensor, shape [4, 64, 1, 1], 源域均值
#     - source_std: torch.Tensor, shape [4, 64, 1, 1], 源域标准差
#     - target_mean: torch.Tensor, shape [4, 64, 1, 1], 目标域均值
#     - target_std: torch.Tensor, shape [4, 64, 1, 1], 目标域标准差
#
#     返回:
#     - weight: torch.Tensor, shape [4] 或标量, 动态权重 (基于 KL 散度)
#
#     注意: 标准差应大于 0, 以避免数值不稳定。添加了小 epsilon 处理。
#     """
#
#     # 添加小 epsilon 避免 log(0) 或除以 0 的问题 (标准差最小值为 1e-5)
#     epsilon = 1e-5
#     source_std = torch.clamp(source_std, min=epsilon)
#     target_std = torch.clamp(target_std, min=epsilon)
#
#     # 逐通道计算 KL 散度
#     # KL 公式: KL = log(sigma_t / sigma_s) + (sigma_s^2 + (mu_s - mu_t)^2) / (2 * sigma_t^2) - 0.5
#     kl_div = torch.log(target_std / source_std) + (source_std ** 2 + (source_mean - target_mean) ** 2) / (
#                 2 * (target_std ** 2)) - 0.5
#
#     # kl_div 形状: [4, 64, 1, 1], 在通道维度 (dim=1) 上取平均, 得到 [4, 1, 1]
#     kl_div_mean_over_channels = kl_div.mean(dim=1, keepdim=True)  # 平均过 64 个通道
#
#     # 现在 kl_div_mean_over_channels 形状为 [4, 1, 1], 可以挤压以获得 [4] 或进一步处理
#     kl_div_aggregated = kl_div_mean_over_channels.squeeze()  # 形状变为 [4], 每个样本一个值
#
#     # 将 KL 散度转换为动态权重 (例如, exp(-KL) 使权重在 [0,1] 范围内, 差异大时权重小)
#     weight = torch.exp(-kl_div_aggregated)  # 或使用 weight = 1 / (1 + kl_div_aggregated) 等变体
#
#     # 返回批次级权重 (形状 [4]), 或取平均以获得标量权重 (取决于您的使用场景)
#     return weight.mean()  # shape [4], 可以根据需要修改为全局标量: return weight.mean()


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
    # model.load_style_bank('style_bank_epoch_last.pth')  # 加载最近保存的风格银行（可选，根据需要）
    with torch.no_grad():
        for i, (img, mask) in enumerate(source_dataloader):
            img, mask = img.to(device), mask.to(device)
            pred = model(img,None)
            # loss = criterion(pred, mask)
            # loss_list.append(loss.item())
            dice_score = dice(pred, mask)
            dice_list.append(dice_score.cpu().detach().numpy())
    # loss = np.mean(loss_list)
    dice_score = np.mean(dice_list)
    return dice_score

def train_one_epoch(model, optimizer, epoch, dataloaders, test_dataloader, task_ids, device):
    split_rat=0.7  # 支持集和查询集的比例
    model.train()  # 设置模型为训练模式
    loss_list = []
    dice_list = []
    style_loss_list = []  # 新增：记录风格损失，便于监控
    segmentation_loss_list = [] # 新增：记录分割损失，便于监控

    # 写入 log.txt：记录 epoch 开始
    with open('log.txt', 'a') as f:
        f.write(f"Epoch {epoch + 1} 开始训练\n")

    for task_id in task_ids:  # 遍历每个任务（域）
        dataloader = dataloaders[task_id]  # 获取当前任务的 DataLoader
        # print(len(dataloader)) # 3
        with open('log.txt', 'a') as f:
            f.write(f"当前任务 ID: {task_id} (域 {task_id})\n")

        if task_id == 0:  # 源域
            # for batch_source in tqdm(dataloader, desc=f'Epoch {epoch + 1}, Task {task_id}', total=len(dataloader)):
            for batch_source in dataloader:
                img, label = batch_source  # 假设批量数据是 (img, label)，img 和 label 都是 tensor
                img = img.to(device)
                label = label.to(device)

                optimizer.zero_grad()

                # 支持集处理：meta-train 模式，计算总损失
                model.set_mode('meta-train')

                source_features, seg_output = model(img,task_id,epoch)  # forward 会自动提取统计并保存到 style_bank
                segmentation_loss = criterion_seg(seg_output, label)  # 使用 Dice 损失或您的分割损失

                # 计算 Dice 分数
                dice_score = dice(seg_output, label)  # 假设 dice 函数已定义
                dice_list.append(dice_score.cpu().item())

                # 记录损失（包括风格损失）
                loss_list.append(segmentation_loss.item())

                # 反向传播和优化
                segmentation_loss.backward()
                optimizer.step()

                # 清零梯度
                optimizer.zero_grad()

        else:  # 目标域
            model.load_style_bank(f'style_bank_{(task_id-1)}.pth')  # 加载上一个任务保存的风格银行
            # for batch_source, batch_target in tqdm(zip(source_dataloader, target_dataloader), desc=f'Epoch {epoch + 1}, Task {task_id}', total=len(target_dataloader)):
            for batch_source, batch_target in zip(dataloaders[0], dataloader):
                img_source, label_source = batch_source
                img_source = img_source.to(device)
                label_source = label_source.to(device)

                img_target, label_target = batch_target
                img_target = img_target.to(device)
                label_target = label_target.to(device)

                batch_size = img_source.size(0)  # 假设源域和目标域批量大小相同

                # 随机分割支持集和查询集（50% 支持集，50% 查询集）
                indices = list(range(batch_size))
                random.shuffle(indices)
                # 分成7:3
                split_idx = int(batch_size * split_rat)
                # split_idx = batch_size // 2
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

                # 支持集处理：计算总损失
                # 获取源域特征（meta-train 模式）
                # 获取源域特征
                with torch.no_grad():
                    model.set_mode('get_source')
                    source_features, source_outputs = model(support_img_source,task_id,epoch)
                    source_mean, source_std = compute_mean_and_std(source_features)

                model.set_mode('meta-train')
                target_features, seg_output = model(support_img_target,task_id,epoch)  # 或使用一个变换后的版本，如果可用
                target_mean, target_std = compute_mean_and_std(target_features)


                # 根据 task_id 调整标签（相似域 vs 不相似域） – 保留您的逻辑
                if task_id in [1, 2]:  # 相似域
                    labels = torch.zeros(source_features.size(0), 1).to(device)  # 标签为 0
                else:  # 不相似域 (task_id 3,4,5)
                    labels = torch.ones(source_features.size(0), 1).to(device)  # 标签为 1

                # 计算风格对齐损失（替换 criterion1）
                # dynamic_weight = compute_dynamic_weight(source_mean, source_std, target_mean, target_std)
                # print(f'与当前task_id:{task_id}的风格差异:', dynamic_weight)
                # style_loss = style_alignment_loss(source_mean, source_std, target_mean, target_std, dynamic_weight)

                # 对比损失
                # style_loss = criterion1(source_features, target_features, labels)

                # 分割损失
                segmentation_loss = criterion_seg(seg_output, support_label_target)  # 使用 Dice 损失或您的分割损失

                # 一致性损失
                consistency_loss = criterion_con(source_outputs, seg_output)

                # dynamic_weight=0
                # 总损失：结合分割损失和风格损失和一致性损失
                # total_loss_support = (1 - dynamic_weight) * segmentation_loss + dynamic_weight * style_loss+ consistency_loss
                # 总损失：结合分割损失和风格损失
                # total_loss_support = (1 - dynamic_weight) * segmentation_loss + dynamic_weight * style_loss
                # 总损失：结合分割损失和一致性损失
                total_loss_support = segmentation_loss + consistency_loss

                # ====================meta-test 模式，使用 meta_loss========================================
                with torch.no_grad():
                    model.set_mode('get_source')
                    source_features, source_outputs = model(query_img_source, task_id)
                    source_mean, source_std = compute_mean_and_std(source_features)

                model.set_mode('meta-test')
                target_features, meta_seg_output = model(query_img_target,task_id,epoch, meta_loss=total_loss_support)  # 使用查询集数据
                target_mean, target_std = compute_mean_and_std(target_features)

                meta_loss = criterion_seg(meta_seg_output, query_label_target)  # 计算元损失

                # 一致性损失
                consistency_loss = criterion_con(source_outputs, meta_seg_output)

                # 根据 task_id 调整标签（相似域 vs 不相似域） – 保留您的逻辑
                if task_id in [1, 2]:  # 相似域
                    labels = torch.zeros(source_features.size(0), 1).to(device)  # 标签为 0
                else:  # 不相似域 (task_id 3,4,5)
                    labels = torch.ones(source_features.size(0), 1).to(device)  # 标签为 1

                # 计算风格对齐损失（替换 criterion1）
                # dynamic_weight = compute_dynamic_weight(source_mean, source_std, target_mean, target_std)
                # print('dynamic_weight:', dynamic_weight)
                # style_loss = style_alignment_loss(source_mean, source_std, target_mean, target_std, dynamic_weight)

                # 对比损失
                # style_loss = criterion1(source_features, target_features, labels)
                dynamic_weight = 0
                # 总损失：结合分割损失和风格损失和一致性损失
                # meta_loss= (1 - dynamic_weight) * meta_loss + dynamic_weight * style_loss + consistency_loss
                # 总损失：结合分割损失和风格损失
                # meta_loss = (1 - dynamic_weight) * meta_loss + dynamic_weight * style_loss
                # 总损失：结合分割损失和一致性损失
                # meta_loss = meta_loss + consistency_loss

                # 计算 Dice 分数
                dice_score = dice(meta_seg_output, query_label_target)
                dice_list.append(dice_score.cpu().item())

                # 记录损失
                loss_list.append(meta_loss.item())
                # style_loss_list.append(style_loss.item())  # 记录风格损失

                # 反向传播和优化
                meta_loss.backward()
                optimizer.step()

                # 清零梯度
                optimizer.zero_grad()

        # task_id 结束处理：保存风格银行和模型
        model.style_bank.save_style_bank(f'style_bank_{task_id}.pth')  # 保存风格统计
        model.style_bank.clear_stats()
    avg_loss = np.mean(loss_list)
    avg_dice = np.mean(dice_list)
    avg_style_loss = np.mean(style_loss_list)  # 计算平均风格损失

    print(f'Epoch {epoch + 1}, Average Meta Loss: {avg_loss:.4f}, Average Style Loss: {avg_style_loss:.4f}, Average Dice Score: {avg_dice:.4f}')
    # 写入 log.txt：包括风格损失
    with open('log.txt', 'a') as f:
        f.write(f"Epoch {epoch + 1}, Average Meta Loss: {avg_loss:.4f}, Average Style Loss: {avg_style_loss:.4f}, Average Dice Score: {avg_dice:.4f}\n")

    # 每 10 个 epoch 保存模型
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f'meta_style_unet_domain_generalization_epoch_{epoch + 1}.pth')

    # 计算源域指标（使用 test_dataloader）
    model.eval()  # 切换到评估模式
    test_dice = compute_source_metric(model, test_dataloader)  # 假设有一个 evaluate_model 函数，返回 Dice 分数
    print(f'Epoch {epoch + 1}, target Metric: {test_dice}')

    with open('log.txt', 'a') as f:
        f.write(f"Epoch {epoch + 1}, Test Dice Score: {test_dice:.4f}\n")

    return avg_loss, avg_dice

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
        loss = criterion_seg(output, batch_targets)

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


def train_one_epoch():
    def train_one_epoch(model, optimizer, epoch, dataloaders, test_dataloader, task_ids, device):
        split_rat = 0.7  # 支持集和查询集的比例
        model.train()  # 设置模型为训练模式
        loss_list = []
        dice_list = []
        style_loss_list = []  # 新增：记录风格损失，便于监控
        segmentation_loss_list = []  # 新增：记录分割损失，便于监控

        # 写入 log.txt：记录 epoch 开始
        with open('log.txt', 'a') as f:
            f.write(f"Epoch {epoch + 1} 开始训练\n")

        for task_id in task_ids:  # 遍历每个任务（域）
            dataloader = dataloaders[task_id]  # 获取当前任务的 DataLoader
            # print(len(dataloader)) # 3
            with open('log.txt', 'a') as f:
                f.write(f"当前任务 ID: {task_id} (域 {task_id})\n")

            if task_id == 0:  # 源域
                # for batch_source in tqdm(dataloader, desc=f'Epoch {epoch + 1}, Task {task_id}', total=len(dataloader)):
                for batch_source in dataloader:
                    img, label = batch_source  # 假设批量数据是 (img, label)，img 和 label 都是 tensor
                    img = img.to(device)
                    label = label.to(device)

                    optimizer.zero_grad()

                    # 支持集处理：meta-train 模式，计算总损失
                    model.set_mode('meta-train')

                    source_features, seg_output = model(img, task_id, epoch)  # forward 会自动提取统计并保存到 style_bank
                    segmentation_loss = criterion_seg(seg_output, label)  # 使用 Dice 损失或您的分割损失

                    # 计算 Dice 分数
                    dice_score = dice(seg_output, label)  # 假设 dice 函数已定义
                    dice_list.append(dice_score.cpu().item())

                    # 记录损失（包括风格损失）
                    loss_list.append(segmentation_loss.item())

                    # 反向传播和优化
                    segmentation_loss.backward()
                    optimizer.step()

                    # 清零梯度
                    optimizer.zero_grad()

            else:  # 目标域
                model.load_style_bank(f'style_bank_{(task_id - 1)}.pth')  # 加载上一个任务保存的风格银行
                # for batch_source, batch_target in tqdm(zip(source_dataloader, target_dataloader), desc=f'Epoch {epoch + 1}, Task {task_id}', total=len(target_dataloader)):
                for batch_source, batch_target in zip(dataloaders[0], dataloader):
                    img_source, label_source = batch_source
                    img_source = img_source.to(device)
                    label_source = label_source.to(device)

                    img_target, label_target = batch_target
                    img_target = img_target.to(device)
                    label_target = label_target.to(device)

                    batch_size = img_source.size(0)  # 假设源域和目标域批量大小相同

                    # 随机分割支持集和查询集（50% 支持集，50% 查询集）
                    indices = list(range(batch_size))
                    random.shuffle(indices)
                    # 分成7:3
                    split_idx = int(batch_size * split_rat)
                    # split_idx = batch_size // 2
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

                    # 支持集处理：计算总损失
                    # 获取源域特征（meta-train 模式）
                    # 获取源域特征
                    with torch.no_grad():
                        model.set_mode('get_source')
                        source_features, source_outputs = model(support_img_source, task_id, epoch)
                        source_mean, source_std = compute_mean_and_std(source_features)

                    model.set_mode('meta-train')
                    target_features, seg_output = model(support_img_target, task_id, epoch)  # 或使用一个变换后的版本，如果可用
                    target_mean, target_std = compute_mean_and_std(target_features)

                    # 根据 task_id 调整标签（相似域 vs 不相似域） – 保留您的逻辑
                    if task_id in [1, 2]:  # 相似域
                        labels = torch.zeros(source_features.size(0), 1).to(device)  # 标签为 0
                    else:  # 不相似域 (task_id 3,4,5)
                        labels = torch.ones(source_features.size(0), 1).to(device)  # 标签为 1

                    # 计算风格对齐损失（替换 criterion1）
                    # dynamic_weight = compute_dynamic_weight(source_mean, source_std, target_mean, target_std)
                    # print(f'与当前task_id:{task_id}的风格差异:', dynamic_weight)
                    # style_loss = style_alignment_loss(source_mean, source_std, target_mean, target_std, dynamic_weight)

                    # 对比损失
                    # style_loss = criterion1(source_features, target_features, labels)

                    # 分割损失
                    segmentation_loss = criterion_seg(seg_output, support_label_target)  # 使用 Dice 损失或您的分割损失

                    # 一致性损失
                    consistency_loss = criterion_con(source_outputs, seg_output)

                    # dynamic_weight=0
                    # 总损失：结合分割损失和风格损失和一致性损失
                    # total_loss_support = (1 - dynamic_weight) * segmentation_loss + dynamic_weight * style_loss+ consistency_loss
                    # 总损失：结合分割损失和风格损失
                    # total_loss_support = (1 - dynamic_weight) * segmentation_loss + dynamic_weight * style_loss
                    # 总损失：结合分割损失和一致性损失
                    total_loss_support = segmentation_loss + consistency_loss

                    # ====================meta-test 模式，使用 meta_loss========================================
                    with torch.no_grad():
                        model.set_mode('get_source')
                        source_features, source_outputs = model(query_img_source, task_id)
                        source_mean, source_std = compute_mean_and_std(source_features)

                    model.set_mode('meta-test')
                    target_features, meta_seg_output = model(query_img_target, task_id, epoch,
                                                             meta_loss=total_loss_support)  # 使用查询集数据
                    target_mean, target_std = compute_mean_and_std(target_features)

                    meta_loss = criterion_seg(meta_seg_output, query_label_target)  # 计算元损失

                    # 一致性损失
                    consistency_loss = criterion_con(source_outputs, meta_seg_output)

                    # 根据 task_id 调整标签（相似域 vs 不相似域） – 保留您的逻辑
                    if task_id in [1, 2]:  # 相似域
                        labels = torch.zeros(source_features.size(0), 1).to(device)  # 标签为 0
                    else:  # 不相似域 (task_id 3,4,5)
                        labels = torch.ones(source_features.size(0), 1).to(device)  # 标签为 1

                    # 计算风格对齐损失（替换 criterion1）
                    # dynamic_weight = compute_dynamic_weight(source_mean, source_std, target_mean, target_std)
                    # print('dynamic_weight:', dynamic_weight)
                    # style_loss = style_alignment_loss(source_mean, source_std, target_mean, target_std, dynamic_weight)

                    # 对比损失
                    # style_loss = criterion1(source_features, target_features, labels)
                    dynamic_weight = 0
                    # 总损失：结合分割损失和风格损失和一致性损失
                    # meta_loss= (1 - dynamic_weight) * meta_loss + dynamic_weight * style_loss + consistency_loss
                    # 总损失：结合分割损失和风格损失
                    # meta_loss = (1 - dynamic_weight) * meta_loss + dynamic_weight * style_loss
                    # 总损失：结合分割损失和一致性损失
                    # meta_loss = meta_loss + consistency_loss

                    # 计算 Dice 分数
                    dice_score = dice(meta_seg_output, query_label_target)
                    dice_list.append(dice_score.cpu().item())

                    # 记录损失
                    loss_list.append(meta_loss.item())
                    # style_loss_list.append(style_loss.item())  # 记录风格损失

                    # 反向传播和优化
                    meta_loss.backward()
                    optimizer.step()

                    # 清零梯度
                    optimizer.zero_grad()

            # task_id 结束处理：保存风格银行和模型
            model.style_bank.save_style_bank(f'style_bank_{task_id}.pth')  # 保存风格统计
            model.style_bank.clear_stats()
        avg_loss = np.mean(loss_list)
        avg_dice = np.mean(dice_list)
        avg_style_loss = np.mean(style_loss_list)  # 计算平均风格损失

        print(
            f'Epoch {epoch + 1}, Average Meta Loss: {avg_loss:.4f}, Average Style Loss: {avg_style_loss:.4f}, Average Dice Score: {avg_dice:.4f}')
        # 写入 log.txt：包括风格损失
        with open('log.txt', 'a') as f:
            f.write(
                f"Epoch {epoch + 1}, Average Meta Loss: {avg_loss:.4f}, Average Style Loss: {avg_style_loss:.4f}, Average Dice Score: {avg_dice:.4f}\n")

        # 每 10 个 epoch 保存模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'meta_style_unet_domain_generalization_epoch_{epoch + 1}.pth')

        # 计算源域指标（使用 test_dataloader）
        model.eval()  # 切换到评估模式
        test_dice = compute_source_metric(model, test_dataloader)  # 假设有一个 evaluate_model 函数，返回 Dice 分数
        print(f'Epoch {epoch + 1}, target Metric: {test_dice}')

        with open('log.txt', 'a') as f:
            f.write(f"Epoch {epoch + 1}, Test Dice Score: {test_dice:.4f}\n")

        return avg_loss, avg_dice


# 辅助函数：创建可重置的迭代器
def cycle_dataloader(dataloader):
    """创建一个在耗尽时会自动重置的迭代器。"""
    while True:
        for data in dataloader:
            yield data
        # 可选：如果需要，在此处添加警告或中断条件，
        # 但对于训练，通常需要循环。


def train_one_epoch_1(model, optimizer, epoch, dataloaders, test_dataloader, task_ids, device):
    """
       使用交错任务采样进行元学习，训练模型一个周期（epoch）。

       Args:
           model: 元学习模型。
           optimizer: 模型参数的优化器。
           epoch: 当前周期编号。
           dataloaders: 将 task_id 映射到其 DataLoader 的字典。
           test_dataloader: 源域测试集的 DataLoader。
           task_ids: 所有任务 ID 的列表 ([0, 1, 2, ...])。
           device: 运行计算的设备 (例如, 'cuda')。
           criterion_seg: 分割损失函数。
           criterion_con: 一致性损失函数。
           dice: 计算 Dice 分数的函数。
           compute_mean_and_std: 计算特征统计量的函数。
       """
    split_rat = 0.7  # 支持集/查询集分割比例
    model.train()  # 设置模型为训练模式

    loss_list = []
    dice_list = []
    # style_loss_list = [] # 如果在元测试中风格损失计算未激活，则保持注释
    segmentation_loss_list = []  # 用于跟踪元损失中的分割部分
    consistency_loss_list = []  # 用于跟踪元损失中的一致性部分

    source_task_id = 0  # 假设任务 0 是源域
    # target_task_ids = [tid for tid in task_ids if tid != source_task_id]  # 获取所有目标域的任务ID

    # --- 创建迭代器 ---
    # 根据最长的数据加载器（或源域）确定迭代次数
    # 通常使用源数据加载器的长度
    num_batches = len(dataloaders[source_task_id]) # data_num/batch_size
    # 或者使用最大长度：num_batches = max(len(dl) for dl in dataloaders.values())

    # 为所有数据加载器创建循环迭代器
    data_iters = {task_id: cycle_dataloader(dataloaders[task_id]) for task_id in task_ids}

    print(f"周期 {epoch + 1}: 正在运行 {num_batches} 次元迭代...")
    with open('log.txt', 'a') as f:
        f.write(f"周期 {epoch + 1} 开始训练 (交错采样)\n")

        # --- 主要元迭代循环 ---
    for batch_idx in tqdm(range(num_batches), desc=f'周期 {epoch + 1} 元迭代'):
        # 1. 采样源域批次
        img_source_full, label_source_full = next(data_iters[source_task_id])
        img_source_full = img_source_full.to(device)
        # print('img_source_full:', img_source_full.shape) # torch.Size([8, 1, 128, 128])
        label_source_full = label_source_full.to(device)

        train_list=[]
        label_list=[]
        # 2. 遍历此元步骤的目标任务
        for task_id in [0,1,2]:
            # 为当前所有任务采样目标域批次
            img_target_full, label_target_full = next(data_iters[task_id])
            train_list.append(img_target_full)
            label_list.append(label_target_full)
        meta_train_batch = torch.cat(train_list, dim=0).to(device)
        # print('train_batch', train_batch.shape) # train_batch torch.Size([48, 1, 128, 128])
        meta_train_label = torch.cat(label_list, dim=0).to(device)

        train_list = []
        label_list = []
        # 2. 遍历此元步骤的目标任务
        for task_id in [3, 4, 5]:
            # 为当前所有任务采样目标域批次
            img_target_full, label_target_full = next(data_iters[task_id])
            train_list.append(img_target_full)
            label_list.append(label_target_full)
        meta_test_batch = torch.cat(train_list, dim=0).to(device)
        # print('train_batch', train_batch.shape) # train_batch torch.Size([48, 1, 128, 128])
        meta_test_label = torch.cat(label_list, dim=0).to(device)


        # --- 此源-目标对的元学习步骤 ---
        optimizer.zero_grad()  # 对每个目标任务的更新清零梯度

        # batch_size = train_batch.size(0)
        # # 分割成支持集和查询集
        # indices = list(range(batch_size))
        # random.shuffle(indices)
        # split_idx = int(batch_size * split_rat)
        # support_indices = indices[:split_idx]
        # query_indices = indices[split_idx:]
        #
        # support_img_source = train_batch[support_indices]
        # support_label_source = label_batch[support_indices]
        #
        # query_img_source = train_batch[query_indices]
        # query_label_source = label_batch[query_indices]


        # --- 内部循环适应 (在支持集上) ---
        # 计算用于元适应的损失 (total_loss_support)
        # 这部分尚不更新主模型权重，
        # 它计算的是 *将要* 指导适应过程的损失。

        # # 获取源域特征（可选，如果一致性损失需要）
        # with torch.no_grad():  # 通常内部循环损失不需要源域的梯度
        #     model.set_mode('get_source')  # 或相关的模式
        #     source_features_support, source_outputs_support = model(support_img_source, task_id, epoch)
        #     # source_mean_support, source_std_support = compute_mean_and_std(source_features_support) # 如果需要

        # 在目标域支持集上进行前向传播
        model.set_mode('meta-train')  # 计算内部损失的模式
        target_features_support, seg_output_support = model(meta_train_batch, 0, epoch)
        # target_mean_support, target_std_support = compute_mean_and_std(target_features_support) # 如果需要

        # 计算支持集上的分割损失
        segmentation_loss_support = criterion_seg(seg_output_support, meta_train_label)

        # 计算支持集上的一致性损失
        # consistency_loss_support = criterion_con(source_outputs_support, seg_output_support)

        # 定义内部循环适应损失 (供 autograd.grad 使用的 meta_loss)
        # 这个损失驱动元测试（meta-test）阶段的 *模拟* 更新
        total_loss_support = segmentation_loss_support
        # 如果其他损失（如风格损失）也应指导适应过程，则添加到这里

        # --- 外部循环更新 (在查询集上) ---
        # 使用 *适应后* 的模型参数计算查询集上的损失
        # 这个损失的梯度将更新 *原始* 模型参数。

        # 获取查询集的源域特征/输出（用于一致性）
        # with torch.no_grad():
        #     model.set_mode('get_source')  # 或相关模式
        #     source_features_query, source_outputs_query = model(query_img_source, task_id, epoch)
        #     # source_mean_query, source_std_query = compute_mean_and_std(source_features_query) # 如果需要

        # 使用适应后的权重在目标域查询集上进行前向传播
        model.set_mode('meta-test')  # 使用 meta_loss 参数的模式
        # 传入内部循环损失 (total_loss_support) 来模拟更新
        target_features_query, meta_seg_output_query = model(meta_test_batch, 0, epoch,
                                                             meta_loss=total_loss_support)
        # target_mean_query, target_std_query = compute_mean_and_std(target_features_query) # 如果需要

        # 计算查询集上的分割损失 (元更新的主要目标)
        meta_seg_loss = criterion_seg(meta_seg_output_query, meta_test_label)

        # 计算查询集上的一致性损失
        # meta_consistency_loss = criterion_con(source_outputs_query, meta_seg_output_query)

        # 计算查询集上的风格损失 (如果在外部循环目标中使用)
        # meta_style_loss = ... # 如果需要在这里计算风格损失

        # 定义用于反向传播的最终元损失
        # 这个损失指导共享的初始参数的更新
        meta_loss = meta_seg_loss
        # meta_loss = meta_loss + meta_style_loss # 如果适用则添加

        # --- 优化 ---
        meta_loss.backward()  # 计算相对于原始参数的梯度
        optimizer.step()  # 更新原始模型参数

        # --- 日志记录 & 指标 ---
        dice_score = dice(meta_seg_output_query, meta_test_label)  # 假设 dice 函数返回单个标量
        dice_list.append(dice_score.cpu().item())
        loss_list.append(meta_loss.item())  # 跟踪最终的元损失
        segmentation_loss_list.append(meta_seg_loss.item())
        # consistency_loss_list.append(meta_consistency_loss.item())
        # if using meta_style_loss: style_loss_list.append(meta_style_loss.item())

        # --- 元迭代结束 ---
        # 注意：风格库的保存/加载可能需要在此处或周期之外进行调整。
        # 原始逻辑是在处理完一个任务的 *所有* 批次后保存。
        # model.style_bank.save_style_bank(f'style_bank_{task_id}.pth') # <-- 现在应该放在哪里？
        # model.style_bank.clear_stats() # <-- 现在应该放在哪里？

        # --- 周期结束 ---
    avg_loss = np.mean(loss_list) if loss_list else 0
    avg_dice = np.mean(dice_list) if dice_list else 0
    avg_seg_loss = np.mean(segmentation_loss_list) if segmentation_loss_list else 0
    avg_cons_loss = np.mean(consistency_loss_list) if consistency_loss_list else 0
    # avg_style_loss = np.mean(style_loss_list) if style_loss_list else 0 # 如果跟踪风格损失，取消注释

    print(
        f'周期 {epoch + 1}, 平均元损失: {avg_loss:.4f}, 平均分割损失: {avg_seg_loss:.4f}, 平均一致性损失: {avg_cons_loss:.4f}, 平均 Dice: {avg_dice:.4f}')
    # print(f'周期 {epoch + 1}, 平均元损失: {avg_loss:.4f}, 平均风格损失: {avg_style_loss:.4f}, 平均 Dice 分数: {avg_dice:.4f}') # 如果使用风格损失

    with open('log.txt', 'a') as f:  # 指定utf-8编码
        f.write(
            f"周期 {epoch + 1}, 平均元损失: {avg_loss:.4f}, 平均分割损失: {avg_seg_loss:.4f}, 平均一致性损失: {avg_cons_loss:.4f}, 平均 Dice: {avg_dice:.4f}\n")
        # f.write(f"周期 {epoch + 1}, 平均元损失: {avg_loss:.4f}, 平均风格损失: {avg_style_loss:.4f}, 平均 Dice 分数: {avg_dice:.4f}\n") # 如果使用风格损失

    # 定期保存模型
    if (epoch + 1) % 10 == 0:
        save_path = f'meta_style_unet_domain_generalization_epoch_{epoch + 1}.pth'
        torch.save(model.state_dict(), save_path)
        print(f"模型已保存至 {save_path}")
        with open('log.txt', 'a') as f:  # 指定utf-8编码
            f.write(f"模型已保存至 {save_path}\n")

    # 在源测试集（或目标测试集，如果可用）上评估
    model.eval()  # 切换到评估模式
    # 假设 compute_source_metric 在 test_dataloader 上评估
    # 您可能需要为不同的域提供单独的评估循环/函数
    test_dice = compute_source_metric(model, test_dataloader)  # 传入 device
    print(f'周期 {epoch + 1}, 目标域测试 Dice: {test_dice:.4f}')
    with open('log.txt', 'a') as f:  # 指定utf-8编码
        f.write(f"周期 {epoch + 1}, 目标域测试 Dice 分数: {test_dice:.4f}\n")

    return avg_loss, avg_dice


def train_one_epoch_2(model, optimizer, epoch, dataloaders, test_dataloader, task_ids, device):
    """
       使用交错任务采样进行元学习，训练模型一个周期（epoch）。

       Args:
           model: 元学习模型。
           optimizer: 模型参数的优化器。
           epoch: 当前周期编号。
           dataloaders: 将 task_id 映射到其 DataLoader 的字典。
           test_dataloader: 源域测试集的 DataLoader。
           task_ids: 所有任务 ID 的列表 ([0, 1, 2, ...])。
           device: 运行计算的设备 (例如, 'cuda')。
           criterion_seg: 分割损失函数。
           criterion_con: 一致性损失函数。
           dice: 计算 Dice 分数的函数。
           compute_mean_and_std: 计算特征统计量的函数。
       """
    split_rat = 0.7  # 支持集/查询集分割比例
    model.train()  # 设置模型为训练模式

    loss_list = []
    dice_list = []
    # style_loss_list = [] # 如果在元测试中风格损失计算未激活，则保持注释
    segmentation_loss_list = []  # 用于跟踪元损失中的分割部分
    consistency_loss_list = []  # 用于跟踪元损失中的一致性部分

    source_task_id = 0  # 假设任务 0 是源域
    target_task_ids = [tid for tid in task_ids if tid != source_task_id]  # 获取所有目标域的任务ID

    # --- 创建迭代器 ---
    # 根据最长的数据加载器（或源域）确定迭代次数
    # 通常使用源数据加载器的长度
    num_batches = len(dataloaders[source_task_id]) # data_num/batch_size
    # 或者使用最大长度：num_batches = max(len(dl) for dl in dataloaders.values())

    # 为所有数据加载器创建循环迭代器
    data_iters = {task_id: cycle_dataloader(dataloaders[task_id]) for task_id in task_ids}

    print(f"周期 {epoch + 1}: 正在运行 {num_batches} 次元迭代...")
    with open('log.txt', 'a') as f:
        f.write(f"周期 {epoch + 1} 开始训练 (交错采样)\n")

        # --- 主要元迭代循环 ---
    for batch_idx in tqdm(range(num_batches), desc=f'周期 {epoch + 1} 元迭代'):
        # 1. 采样源域批次
        img_source_full, label_source_full = next(data_iters[source_task_id])
        img_source_full = img_source_full.to(device)
        # print('img_source_full:', img_source_full.shape) #
        label_source_full = label_source_full.to(device)


        # 2. 遍历此元步骤的目标任务
        for task_id in target_task_ids:
            # 为当前目标任务采样目标域批次
            img_target_full, label_target_full = next(data_iters[task_id])
            img_target_full = img_target_full.to(device)
            label_target_full = label_target_full.to(device)

            # --- 此源-目标对的元学习步骤 ---
            optimizer.zero_grad()  # 对每个目标任务的更新清零梯度

            # 确保批次大小一致（如果原始大小不同）
            batch_size = min(img_source_full.size(0), img_target_full.size(0))
            if batch_size == 0: continue  # 如果批次为空则跳过

            img_source = img_source_full[:batch_size]
            # print('img_source:', img_source.shape)  # torch.Size([8, 1, 128, 128])
            label_source = label_source_full[:batch_size]

            img_target = img_target_full[:batch_size]
            label_target = label_target_full[:batch_size]

            # 分割成支持集和查询集
            indices = list(range(batch_size))
            random.shuffle(indices)
            split_idx = int(batch_size * split_rat)
            support_indices = indices[:split_idx]
            query_indices = indices[split_idx:]

            # 检查分割索引是否有效
            if not support_indices or not query_indices:
                print(f"警告：因批次大小过小 ({batch_size}) 导致支持/查询集为空，跳过任务 {task_id} 的批次。")
                continue

            support_img_source = img_source[support_indices]
            support_label_source = label_source[support_indices]
            query_img_source = img_source[query_indices]
            query_label_source = label_source[query_indices]

            support_img_target = img_target[support_indices]
            support_label_target = label_target[support_indices]
            query_img_target = img_target[query_indices]
            query_label_target = label_target[query_indices]

            # --- 内部循环适应 (在支持集上) ---
            # 计算用于元适应的损失 (total_loss_support)
            # 这部分尚不更新主模型权重，
            # 它计算的是 *将要* 指导适应过程的损失。

            # 获取源域特征（可选，如果一致性损失需要）
            with torch.no_grad():  # 通常内部循环损失不需要源域的梯度
                model.set_mode('get_source')  # 或相关的模式
                source_features_support, source_outputs_support = model(support_img_source, task_id, epoch)
                # source_mean_support, source_std_support = compute_mean_and_std(source_features_support) # 如果需要

            # 在目标域支持集上进行前向传播
            model.set_mode('meta-train')  # 计算内部损失的模式
            target_features_support, seg_output_support = model(support_img_target, task_id, epoch)
            # target_mean_support, target_std_support = compute_mean_and_std(target_features_support) # 如果需要

            # 计算支持集上的分割损失
            segmentation_loss_support = criterion_seg(seg_output_support, support_label_target)

            # 计算支持集上的一致性损失
            # consistency_loss_support = criterion_con(source_outputs_support, seg_output_support)

            # 定义内部循环适应损失 (供 autograd.grad 使用的 meta_loss)
            # 这个损失驱动元测试（meta-test）阶段的 *模拟* 更新
            total_loss_support = segmentation_loss_support
            # 如果其他损失（如风格损失）也应指导适应过程，则添加到这里

            # --- 外部循环更新 (在查询集上) ---
            # 使用 *适应后* 的模型参数计算查询集上的损失
            # 这个损失的梯度将更新 *原始* 模型参数。

            # 获取查询集的源域特征/输出（用于一致性）
            with torch.no_grad():
                model.set_mode('get_source')  # 或相关模式
                source_features_query, source_outputs_query = model(query_img_source, task_id, epoch)
                # source_mean_query, source_std_query = compute_mean_and_std(source_features_query) # 如果需要

            # 使用适应后的权重在目标域查询集上进行前向传播
            model.set_mode('meta-test')  # 使用 meta_loss 参数的模式
            # 传入内部循环损失 (total_loss_support) 来模拟更新
            target_features_query, meta_seg_output_query = model(query_img_target, task_id, epoch,
                                                                 meta_loss=total_loss_support)
            # target_mean_query, target_std_query = compute_mean_and_std(target_features_query) # 如果需要

            # 计算查询集上的分割损失 (元更新的主要目标)
            meta_seg_loss = criterion_seg(meta_seg_output_query, query_label_target)

            # 计算查询集上的一致性损失
            # meta_consistency_loss = criterion_con(source_outputs_query, meta_seg_output_query)

            # 计算查询集上的风格损失 (如果在外部循环目标中使用)
            # meta_style_loss = ... # 如果需要在这里计算风格损失

            # 定义用于反向传播的最终元损失
            # 这个损失指导共享的初始参数的更新
            meta_loss = meta_seg_loss
            # meta_loss = meta_loss + meta_style_loss # 如果适用则添加

            # --- 优化 ---
            meta_loss.backward()  # 计算相对于原始参数的梯度
            optimizer.step()  # 更新原始模型参数

            # --- 日志记录 & 指标 ---
            dice_score = dice(meta_seg_output_query, query_label_target)  # 假设 dice 函数返回单个标量
            dice_list.append(dice_score.cpu().item())
            loss_list.append(meta_loss.item())  # 跟踪最终的元损失
            segmentation_loss_list.append(meta_seg_loss.item())
            # consistency_loss_list.append(meta_consistency_loss.item())
            # if using meta_style_loss: style_loss_list.append(meta_style_loss.item())

        # --- 元迭代结束 ---
        # 注意：风格库的保存/加载可能需要在此处或周期之外进行调整。
        # 原始逻辑是在处理完一个任务的 *所有* 批次后保存。
        # model.style_bank.save_style_bank(f'style_bank_{task_id}.pth') # <-- 现在应该放在哪里？
        # model.style_bank.clear_stats() # <-- 现在应该放在哪里？

        # --- 周期结束 ---
    avg_loss = np.mean(loss_list) if loss_list else 0
    avg_dice = np.mean(dice_list) if dice_list else 0
    avg_seg_loss = np.mean(segmentation_loss_list) if segmentation_loss_list else 0
    avg_cons_loss = np.mean(consistency_loss_list) if consistency_loss_list else 0
    # avg_style_loss = np.mean(style_loss_list) if style_loss_list else 0 # 如果跟踪风格损失，取消注释

    print(
        f'周期 {epoch + 1}, 平均元损失: {avg_loss:.4f}, 平均分割损失: {avg_seg_loss:.4f}, 平均一致性损失: {avg_cons_loss:.4f}, 平均 Dice: {avg_dice:.4f}')
    # print(f'周期 {epoch + 1}, 平均元损失: {avg_loss:.4f}, 平均风格损失: {avg_style_loss:.4f}, 平均 Dice 分数: {avg_dice:.4f}') # 如果使用风格损失

    with open('log.txt', 'a') as f:  # 指定utf-8编码
        f.write(
            f"周期 {epoch + 1}, 平均元损失: {avg_loss:.4f}, 平均分割损失: {avg_seg_loss:.4f}, 平均一致性损失: {avg_cons_loss:.4f}, 平均 Dice: {avg_dice:.4f}\n")
        # f.write(f"周期 {epoch + 1}, 平均元损失: {avg_loss:.4f}, 平均风格损失: {avg_style_loss:.4f}, 平均 Dice 分数: {avg_dice:.4f}\n") # 如果使用风格损失

    # 定期保存模型
    if (epoch + 1) % 10 == 0:
        save_path = f'meta_style_unet_domain_generalization_epoch_{epoch + 1}.pth'
        torch.save(model.state_dict(), save_path)
        print(f"模型已保存至 {save_path}")
        with open('log.txt', 'a') as f:  # 指定utf-8编码
            f.write(f"模型已保存至 {save_path}\n")

    # 在源测试集（或目标测试集，如果可用）上评估
    model.eval()  # 切换到评估模式
    # 假设 compute_source_metric 在 test_dataloader 上评估
    # 您可能需要为不同的域提供单独的评估循环/函数
    test_dice = compute_source_metric(model, test_dataloader)  # 传入 device
    print(f'周期 {epoch + 1}, 目标域测试 Dice: {test_dice:.4f}')
    with open('log.txt', 'a') as f:  # 指定utf-8编码
        f.write(f"周期 {epoch + 1}, 目标域测试 Dice 分数: {test_dice:.4f}\n")

    return avg_loss, avg_dice



def train_one_epoch_3(model, optimizer, epoch, dataloaders,test_dataloader, task_ids, device):
    model.train()
    total_loss_list = []
    seg_loss_list = []
    consistency_loss_list = []
    style_loss_list = [] # 如果在元测试中风格损失计算未激活，则保持注/释
    dice_list = []

    # dataloaders 是一个字典，键是 task_id，值是对应的 DataLoader
    # 例如：dataloaders = {0: source_dataloader, 1: domain1_dataloader, ..., 5: domain5_dataloader}
    # task_ids 是一个列表，包含所有任务 ID，例如 [0, 1, 2, 3, 4, 5]
    if epoch == model.epoch_fill:
        print('开始使用风格库')

    # 写入 log.txt
    with open('log.txt', 'a') as f:
        f.write(f"Epoch {epoch + 1} 开始训练\n")

    for task_id in task_ids:  # 遍历每个任务（域）
        if task_id!=5:
            meta_train_dataloader = dataloaders[task_id]  # 获取当前任务的 DataLoader
            meta_test_dataloader=dataloaders[task_id+1]
        else:
            meta_train_dataloader = dataloaders[task_id]
            meta_test_dataloader=dataloaders[task_id]
        with open('log.txt', 'a') as f:
            f.write(f"当前任务 ID: {task_id} (域 {task_id})\n")
        # print('当前任务 ID:', task_id)
        for batch_train,batch_test  ,batch_source in tqdm(zip(meta_train_dataloader,meta_test_dataloader,dataloaders[0]), desc=f'Epoch {epoch + 1}, Task {task_id}', total=len(meta_train_dataloader)):
        # for batch_source in dataloader:
            # 仅使用源域数据
            img, label = batch_source
            source_img = img.to(device)
            source_label = label.to(device)

            img, label = batch_train
            support_img = img.to(device)
            support_label = label.to(device)

            img,label=batch_test
            query_img = img.to(device)
            query_label = label.to(device)

            # print('============= 支持集处理：计算 total_loss =============')
            model.set_mode('meta-train')
            if task_id!=0:
                model.load_style_bank(f'./StyleBank/style_bank_{task_id-1}.pth')  # 加载风格库

            optimizer.zero_grad()  # 清零梯度，确保从零开始

            source_features, seg_output = model(support_img,task_id=task_id, epoch=epoch)
            segmentation_loss = criterion_seg(seg_output, support_label)  # 分割损失

            segmentation_loss.backward(retain_graph=True)
            model.zero_grad()


            # print(' ============= 查询集处理：meta-test 模式 =============')
            model.set_mode('meta-test')
            meta_seg_feature, meta_seg_output = model(query_img,task_id=task_id, epoch=epoch, meta_loss=segmentation_loss)  # 使用查询集数据和 meta_loss
            meta_seg_source_feature, meta_seg_output_source = model(source_img,task_id=0, epoch=epoch, meta_loss=segmentation_loss)  # 使用查询集数据和 meta_loss

            seg_loss = criterion_seg(meta_seg_output, query_label)

            consistency_loss = criterion_con(meta_seg_output, meta_seg_output_source)

            # 根据 task_id 调整标签（相似域 vs 不相似域）
            if task_id in [1, 2]:  # 相似域
                labels = torch.ones(source_features.size(0), 1).to(device)  # 相似，标签为 1
            else:  # 不相似域 (task_id 3,4,5)
                labels = torch.zeros(source_features.size(0), 1).to(device)  # 不相似，标签为 0

            # 计算风格损失权重
            style_loss = criterion1(meta_seg_source_feature, meta_seg_feature, labels)  # 对比损失
            source_mean, source_std = compute_mean_and_std(source_features)  # 计算源域均值和标准差
            target_mean, target_std = compute_mean_and_std(meta_seg_feature)  # 计算目标域均值和标准差
            weight=compute_dynamic_weight(source_mean, source_std, target_mean, target_std)  # 计算动态权重


            lambda_consistency = 0.1  # 超参数，控制一致性损失的权重（建议从0.1开始调整）
            meta_loss = seg_loss + lambda_consistency * consistency_loss

            # 计算 Dice 分数
            dice_score = dice(meta_seg_output, query_label)
            dice_list.append(dice_score.cpu().detach().numpy())

            # 记录损失
            total_loss_list.append(meta_loss.item())
            seg_loss_list.append(seg_loss.item())
            consistency_loss_list.append(consistency_loss.item())
            style_loss_list.append(style_loss.item())

            # 反向传播和优化
            meta_loss.backward()
            optimizer.step()
            # 清零梯度（可选，确保每个批量独立）
            optimizer.zero_grad()

        model.style_bank.save_style_bank(f'./StyleBank/style_bank_{task_id}.pth')
        model.style_bank.clear_stats()

    # =============  epoch 结束处理 =============
    total_loss = np.mean(total_loss_list)
    seg_loss = np.mean(seg_loss_list)
    consistency_loss = np.mean(consistency_loss_list)
    style_loss = np.mean(style_loss_list)  # 如果使用风格损失
    dice_score = np.mean(dice_list)
    print('Epoch {}: 平均损失: {:.4f}, 平均分割损失: {:.4f}, 平均一致性损失: {:.4f}, 平均风格损失: {:.4f}, 平均 Dice 分数: {:.4f}'.format(
        epoch + 1, total_loss, seg_loss, consistency_loss, style_loss, dice_score))

    # 每 10 个 epoch 保存模型
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f'./CheckPoint/meta_style_unet_domain_generalization_epoch_{epoch + 1}.pth')

    # 计算源域指标（如果有测试数据加载器）
    # 有一个 test_dataloader 用于目标域评估
    source_metric = compute_source_metric(model, test_dataloader)  # 需要定义或传入 test_dataloader
    print(f'Epoch {epoch + 1}, Target Metric: {source_metric}')

    # 写入 log.txt
    with open('log.txt', 'a') as f:
        f.write(f"Epoch {epoch + 1}, 平均损失: {total_loss:.4f}, 平均分割损失: {seg_loss:.4f}, 平均一致性损失: {consistency_loss:.4f}, 平均风格损失: {style_loss:.4f}, 平均 Dice 分数: {dice_score:.4f}\n")



if __name__ == '__main__':
    # train_FM()
    with open('log.txt', 'a') as f:
        # 画一个开始的分割线
        f.write("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

    batch_size = 2
    domain = 't2'
    test_domain = 't1'
    train_path = f'G:\VS_project\Brats-Demo\processed_2d_train_num_bezier\\{domain}'
    val_path = f'G:\VS_project\Brats-Demo\processed_2d_val_num_bezier\\{domain}'
    test_path = f'G:\VS_project\Brats-Demo\processed_2d_test_num_bezier\\{test_domain}'

    index=25
    # train_case是一个列表，0到13291
    train_case = [i for i in range(len(os.listdir(train_path)) // 6)]
    print(f"train_case:{train_case[-1]}")
    train_case = train_case[:index]

    val_case = [i for i in range(len(os.listdir(val_path)) // 6)]
    val_case = val_case[:index]

    test_case = os.listdir(test_path)
    test_case = test_case[:index]


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
    # criterion = DiceLoss()
    criterion_seg = nn.BCELoss()
    # 计算对比损失
    criterion1 = ContrastiveLoss(margin=1.0)
    # 一致性损失
    criterion_con = nn.MSELoss()
    # style_bank = StyleFeatureBank()
    # 启用钩子以自动提取统计
    # model.enable_hooks(True)  # 钩子在 UNet __init__ 中定义，确保自动捕获中间层统计
    # 训练循环
    num_epochs = 100
    for epoch in range(num_epochs):
        train_one_epoch_3(model, optimizer, epoch, dataloaders_train, testdataloder,task_ids, device)

        source_metric = compute_source_metric(model, testdataloder)
        # 写入log.txt
        with open('log.txt', 'a') as f:
            f.write(f"Source metric: {source_metric}\n")

        feedback_signals, domain_gaps = compute_feedback_metrics(model, dataloaders_val)
        # 输出反馈信号，写入log.txt
        with open('log.txt', 'a') as f:
            f.write(
                f"Epoch {epoch + 1}, Feedback signals from target domains: {feedback_signals}, Domain gap: {domain_gaps}\n")


        # 强化训练阶段,强化非相似域的训练
        for epoch in range(10):
            reinforce_training(model, optimizer, epoch, dataloaders_train, domain_gaps)
            _, domain_gaps = compute_feedback_metrics(model, dataloaders_val)

        # # 再次验证泛化能力
        source_metric = compute_source_metric(model, testdataloder)
        # 写入log.txt
        with open('log.txt', 'a') as f:
            f.write(f"Updated source metric: {source_metric}\n")
            f.write("************************************\n")
    # 保存模型
    torch.save(model.state_dict(), f'./final_model.pth')
