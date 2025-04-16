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

                batch_size = img.size(0)

                # 随机分割支持集和查询集（50% 支持集，50% 查询集）
                indices = list(range(batch_size))
                random.shuffle(indices)
                split_idx = batch_size // 2
                support_indices = indices[:split_idx]
                query_indices = indices[split_idx:]

                support_img = img[support_indices]
                support_label = label[support_indices]
                query_img = img[query_indices]
                query_label = label[query_indices]

                optimizer.zero_grad()

                # 支持集处理：meta-train 模式，计算总损失
                model.set_mode('meta-train')

                source_features, seg_output = model(support_img,task_id)  # forward 会自动提取统计并保存到 style_bank
                segmentation_loss = criterion(seg_output, support_label)  # 使用 Dice 损失或您的分割损失

                # 计算风格统计（钩子已自动处理，但这里可以显式获取如果需要调试）
                source_mean, source_std = compute_mean_and_std(source_features)  # 从 source_features 计算

                # 由于是源域，这里可以跳过风格对齐损失或使用内部参考（可选）
                style_loss = torch.tensor(0.0).to(device)  # 源域可能不需要风格对齐，或与历史统计比较
                dynamic_weight = compute_dynamic_weight(source_mean, source_std, source_mean, source_std)  # 自比较，权重接近0
                # print('dynamic_weight:', dynamic_weight)

                total_loss_support = (1 - dynamic_weight) * segmentation_loss + dynamic_weight * style_loss

                # 查询集处理：meta-test 模式，使用 meta_loss
                model.set_mode('meta-test')
                _, meta_seg_output = model(query_img,task_id, meta_loss=total_loss_support)  # 使用查询集数据和 meta_loss
                meta_loss = criterion(meta_seg_output, query_label)  # 计算元损失

                # 计算 Dice 分数
                dice_score = dice(meta_seg_output, query_label)  # 假设 dice 函数已定义
                dice_list.append(dice_score.cpu().item())

                # 记录损失（包括风格损失）
                loss_list.append(meta_loss.item())
                # style_loss_list.append(style_loss.item())  # 记录风格损失

                # 反向传播和优化
                meta_loss.backward()
                optimizer.step()

                # 清零梯度
                optimizer.zero_grad()

        else:  # 目标域
            model.load_style_bank(f'style_bank_epoch_{(task_id-1)}.pth')  # 加载上一个任务保存的风格银行
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

                # 支持集处理：计算总损失
                # 获取源域特征（meta-train 模式）
                # 获取源域特征
                with torch.no_grad():
                    model.set_mode('get_source')
                    source_features, _ = model(support_img_source,task_id)
                    source_mean, source_std = compute_mean_and_std(source_features)
                model.set_mode('meta-train')
                target_features, seg_output = model(support_img_target,task_id)  # 或使用一个变换后的版本，如果可用
                target_mean, target_std = compute_mean_and_std(target_features)


                # 根据 task_id 调整标签（相似域 vs 不相似域） – 保留您的逻辑
                if task_id in [1, 2]:  # 相似域
                    labels = torch.zeros(source_features.size(0), 1).to(device)  # 标签为 0
                else:  # 不相似域 (task_id 3,4,5)
                    labels = torch.ones(source_features.size(0), 1).to(device)  # 标签为 1

                # 计算风格对齐损失（替换 criterion1）
                dynamic_weight = compute_dynamic_weight(source_mean, source_std, target_mean, target_std)
                # print('dynamic_weight:', dynamic_weight)
                # style_loss = style_alignment_loss(source_mean, source_std, target_mean, target_std, dynamic_weight)

                # 对比损失
                style_loss = criterion1(source_features, target_features, labels)

                # 分割损失
                segmentation_loss = criterion(seg_output, support_label_target)

                dynamic_weight=0
                # 总损失：结合分割损失和风格损失
                total_loss_support = (1 - dynamic_weight) * segmentation_loss + dynamic_weight * style_loss

                # 查询集处理：meta-test 模式，使用 meta_loss
                with torch.no_grad():
                    model.set_mode('get_source')
                    source_features, _ = model(query_img_source, task_id)
                    source_mean, source_std = compute_mean_and_std(source_features)
                model.set_mode('meta-test')
                target_features, meta_seg_output = model(query_img_target,task_id, meta_loss=total_loss_support)  # 使用查询集数据
                target_mean, target_std = compute_mean_and_std(target_features)

                meta_loss = criterion(meta_seg_output, query_label_target)  # 计算元损失

                # 根据 task_id 调整标签（相似域 vs 不相似域） – 保留您的逻辑
                if task_id in [1, 2]:  # 相似域
                    labels = torch.zeros(source_features.size(0), 1).to(device)  # 标签为 0
                else:  # 不相似域 (task_id 3,4,5)
                    labels = torch.ones(source_features.size(0), 1).to(device)  # 标签为 1

                # 计算风格对齐损失（替换 criterion1）
                dynamic_weight = compute_dynamic_weight(source_mean, source_std, target_mean, target_std)
                # print('dynamic_weight:', dynamic_weight)
                # style_loss = style_alignment_loss(source_mean, source_std, target_mean, target_std, dynamic_weight)

                # 对比损失
                style_loss = criterion1(source_features, target_features, labels)
                dynamic_weight = 0
                meta_loss= (1 - dynamic_weight) * meta_loss + dynamic_weight * style_loss

                # 计算 Dice 分数
                dice_score = dice(meta_seg_output, query_label_target)
                dice_list.append(dice_score.cpu().item())

                # 记录损失
                loss_list.append(meta_loss.item())
                style_loss_list.append(style_loss.item())  # 记录风格损失

                # 反向传播和优化
                meta_loss.backward()
                optimizer.step()

                # 清零梯度
                optimizer.zero_grad()

        # task_id 结束处理：保存风格银行和模型
        model.style_bank.save_style_bank(f'style_bank_epoch_{task_id}.pth')  # 保存风格统计
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
    print(f'Epoch {epoch + 1}, Source Metric: {test_dice}')

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
    test_domain = 't1'
    train_path = f'G:\VS_project\Brats-Demo\processed_2d_train_num_bezier\\{domain}'
    val_path = f'G:\VS_project\Brats-Demo\processed_2d_val_num_bezier\\{domain}'
    test_path = f'G:\VS_project\Brats-Demo\processed_2d_test_num_bezier\\{test_domain}'

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
    # criterion = DiceLoss()
    criterion = nn.BCELoss()
    # 计算对比损失
    criterion1 = ContrastiveLoss(margin=1.0)
    # style_bank = StyleFeatureBank()
    # 启用钩子以自动提取统计
    model.enable_hooks(True)  # 钩子在 UNet __init__ 中定义，确保自动捕获中间层统计
    # 训练循环
    num_epochs = 100
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, epoch, dataloaders_train, testdataloder,task_ids, device)

        # source_metric = compute_source_metric(model, testdataloder)
        # # 写入log.txt
        # with open('log.txt', 'a') as f:
        #     f.write(f"Source metric: {source_metric}\n")
        #
        # feedback_signals, domain_gaps = compute_feedback_metrics(model, dataloaders_val)
        # # 输出反馈信号，写入log.txt
        # with open('log.txt', 'a') as f:
        #     f.write(
        #         f"Epoch {epoch + 1}, Feedback signals from target domains: {feedback_signals}, Domain gap: {domain_gaps}\n")
        #
        #
        # # 强化训练阶段,强化非相似域的训练
        # for epoch in range(10):
        #     reinforce_training(model, optimizer, epoch, dataloaders_train, domain_gaps)
        #     _, domain_gaps = compute_feedback_metrics(model, dataloaders_val)
        #
        # # # 再次验证泛化能力
        # source_metric = compute_source_metric(model, testdataloder)
        # # 写入log.txt
        # with open('log.txt', 'a') as f:
        #     f.write(f"Updated source metric: {source_metric}\n")
        #     f.write("************************************\n")
    # 保存模型
    torch.save(model.state_dict(), f'./final_model.pth')
