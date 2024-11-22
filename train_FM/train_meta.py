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

# 换meta-unet的一个版本,updated
config = load_config()


def load_updated_params(network, updated_params):
    for name, param in network.named_parameters():
        param.data = updated_params[name].data.clone()  # Avoid in-place operation


def compute_loss(model, images, labels, mode):
    # 处理输入
    print(f"images:{images.shape}")

    if mode == 'meta_train':
        outputs = model(images)
        # print(outputs.shape)
        # print(label_adjusted.shape)
        loss = dice_loss(outputs, labels)
    elif mode == 'meta_test':
        # 在元训练后参数上计算损失
        seg_out, rec_out = model(images)
        # 分割损失
        seg_loss = dice_loss(seg_out, labels)
        # 重建损失
        reconstruction_loss = ReconstructionLoss()
        rec_loss = reconstruction_loss(images, rec_out)
        loss = seg_loss + rec_loss

    return loss


# 每个batch里面划分元测试集和元训练集
def split_batch(images, labels, meta_train_radio=0.8):
    batch_size = images.size(0)
    meta_train_size = int(batch_size * meta_train_radio)
    indices = torch.randperm(batch_size)

    meta_train_indices = indices[:meta_train_size]
    meta_test_indices = indices[meta_train_size:]

    meta_train_images = images[meta_train_indices]
    meta_train_labels = labels[meta_train_indices]

    meta_test_images = images[meta_test_indices]
    meta_test_labels = labels[meta_test_indices]

    return (meta_train_images, meta_train_labels), (meta_test_images, meta_test_labels)


def train():
    # 源相似域
    train_similar_dir = config['train_FM'][
        'train_source_similar_dir']  # 'dataset/BRATS-2018/train_npz_data/t2_ss_train'
    test_similar_dir = config['train_FM']['test_source_similar_dir']  # 'dataset/BRATS-2018/train_npz_data/t2_ss_test'
    # 源非相似域
    train_dissimilar_dir = config['train_FM'][
        'train_source_dissimilar_dir']  # 'dataset/BRATS-2018/train_npz_data/t2_sd_train'
    test_dissimilar_dir = config['train_FM'][
        'test_source_dissimilar_dir']  # 'dataset/BRATS-2018/train_npz_data/t2_sd_test'

    batch_size = config['train_FM']['batch_size']
    max_epoch = config['train_FM']['max_epoch']
    train_lr = config['train_FM']['lr']
    save_interval = config['train_FM']['save_interval']
    rec_dir = config['train_FM']['rec_dir']
    # 日志记录
    # 创建一个 SummaryWriter 实例
    writer = SummaryWriter(log_dir='logs')
    # 加载训练数据集
    # 这里源相似域和源非相似域分开，作为训练和测试
    ss_meta_train_dataset = BraTSDataset([train_similar_dir], 'train_FM')
    ss_meta_test_dataset = BraTSDataset([test_similar_dir], 'train_FM')
    sd_meta_train_dataset = BraTSDataset([train_dissimilar_dir], 'train_FM')
    sd_meta_test_dataset = BraTSDataset([test_dissimilar_dir], 'train_FM')

    ss_meta_train_dataloader = DataLoader(ss_meta_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                          num_workers=16)
    ss_meta_test_dataloader = DataLoader(ss_meta_test_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                         num_workers=16)
    sd_meta_train_dataloader = DataLoader(sd_meta_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                          num_workers=16)
    sd_meta_test_dataloader = DataLoader(sd_meta_test_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                         num_workers=16)
    ss_batch_num = min(len(ss_meta_train_dataloader), len(ss_meta_test_dataloader))  # 获取最小的批次数
    sd_batch_num = min(len(sd_meta_train_dataloader), len(sd_meta_test_dataloader))
    model = UNet()
    model = model.to('cuda:0')
    optimizer = torch.optim.Adam(model.parameters(), lr=train_lr)
    # # 计算参数量
    # total_params = sum(p.numel() for p in model.parameters())
    # model_size_bytes = total_params * 8  # 每个参数8字节
    # model_size_mb = model_size_bytes / (1024 * 1024)
    # print(f"模型大小: {model_size_mb:.2f} MB")
    # exit(0)

    for epoch in range(max_epoch):
        print(f'Epoch {epoch + 1}/{max_epoch}')
        # total_meta_train_loss = 0.0
        # total_meta_test_loss = 0.0
        model.train()
        # with tqdm(meta_train_dataloader, desc=f'Epoch{epoch+1}/{max_epoch}', leave=True) as pbar:
        #     for batch_idx, (batch_images, batch_labels) in enumerate(pbar):
        for batch_idx, (meta_train_data, meta_test_data) in enumerate(
                tqdm(zip(ss_meta_train_dataloader, ss_meta_test_dataloader), total=ss_batch_num,
                     desc=f'Epoch{epoch + 1}/{max_epoch}')):

            # (train_images, train_labels), (test_images, test_labels) = split_batch(batch_images, batch_labels)
            train_images = meta_train_data[0]
            train_labels = meta_train_data[1]
            # meta-train部分
            train_images = train_images.float().to('cuda:0')
            train_labels = train_labels.float().to('cuda:0')
            outputs = model(train_images, mode='meta-train_FM')
            print(f"outputs:{outputs.shape}")
            print(f"train_labels:{train_labels.shape}")
            meta_train_loss = dice_loss(outputs, train_labels)

            # meta-test部分
            model.mode = 'meta-test'
            test_images = meta_test_data[0]
            test_labels = meta_test_data[1]
            test_images = test_images.float().to('cuda:0')
            test_labels = test_labels.float().to('cuda:0')

            # 在元训练后参数上计算损失
            seg_out, rec_out = model(test_images, mode='meta-test', meta_loss=meta_train_loss)
            # print(f"seg_out:{seg_out.shape}")
            # exit(0)
            # 分割损失
            seg_loss = dice_loss(seg_out, test_labels)
            # 重建损失
            reconstruction_loss = ReconstructionLoss()
            rec_loss = reconstruction_loss(test_images, rec_out)
            meta_test_loss = seg_loss + rec_loss
            # print(f"meta_test_loss:{meta_test_loss}")
            # total_meta_test_loss = total_meta_test_loss + meta_test_loss
            # 总损失
            total_loss = meta_test_loss + meta_train_loss
            # print(f"total_loss:{total_loss}")
            # model.train_FM()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # 将损失写入 TensorBoard
            writer.add_scalar('Loss/Meta_Train_Loss', meta_train_loss.item(), epoch * ss_batch_num + batch_idx)
            writer.add_scalar('Loss/Meta_Test_Loss', meta_test_loss.item(), epoch * ss_batch_num + batch_idx)
            writer.add_scalar('Loss/Total_Loss', total_loss.item(), epoch * ss_batch_num + batch_idx)
            # original_images = vutils.make_grid(test_images, normalize=True, scale_each=True)
            # reconstructed_images = vutils.make_grid(rec_out, normalize=True, scale_each=True)
            # writer.add_image('Original_Images', original_images, epoch * batch_num + batch_idx)
            # writer.add_image('Reconstructed_Images', reconstructed_images, epoch * batch_num + batch_idx)
            # if not os.path.exists(rec_dir):
            #     os.makedirs(rec_dir)
            # # 保存重建图像到文件夹
            # for idx in range(rec_out.size(0)):
            #     save_image(rec_out[idx], os.path.join(rec_dir,
            #                                           f'reconstructed_epoch{epoch + 1}_batch{batch_idx + 1}_img{idx + 1}.png'))
            # 保存模型
            if (epoch + 1) % save_interval == 0:
                save_model(model, optimizer, epoch + 1, f'ss_model_epoch_{epoch + 1}.pth')

    for epoch in range(max_epoch):
        print(f'Epoch {epoch + 1}/{max_epoch}')
        # total_meta_train_loss = 0.0
        # total_meta_test_loss = 0.0
        model.train()
        # with tqdm(meta_train_dataloader, desc=f'Epoch{epoch+1}/{max_epoch}', leave=True) as pbar:
        #     for batch_idx, (batch_images, batch_labels) in enumerate(pbar):
        for batch_idx, (meta_train_data, meta_test_data) in enumerate(
                tqdm(zip(sd_meta_train_dataloader, sd_meta_test_dataloader),
                     total=sd_batch_num, desc=f'Epoch{epoch + 1}/{max_epoch}')):

            # (train_images, train_labels), (test_images, test_labels) = split_batch(batch_images, batch_labels)
            train_images = meta_train_data[0]
            train_labels = meta_train_data[1]
            # meta-train部分
            train_images = train_images.float().to('cuda:0')
            train_labels = train_labels.float().to('cuda:0')
            outputs = model(train_images, mode='meta-train_FM')
            meta_train_loss = dice_loss(outputs, train_labels)

            # meta-test部分
            model.mode = 'meta-test'
            test_images = meta_test_data[0]
            test_labels = meta_test_data[1]
            test_images = test_images.float().to('cuda:0')
            test_labels = test_labels.float().to('cuda:0')
            # 在元训练后参数上计算损失
            seg_out, rec_out = model(test_images, mode='meta-test', meta_loss=meta_train_loss)
            # print(f"seg_out:{seg_out.shape}")
            # exit(0)
            # 分割损失
            seg_loss = dice_loss(seg_out, test_labels)
            # 重建损失
            reconstruction_loss = ReconstructionLoss()
            rec_loss = reconstruction_loss(test_images, rec_out)
            meta_test_loss = seg_loss + rec_loss
            # print(f"meta_test_loss:{meta_test_loss}")
            # total_meta_test_loss = total_meta_test_loss + meta_test_loss
            # 总损失
            total_loss = meta_test_loss + meta_train_loss
            # print(f"total_loss:{total_loss}")
            # model.train_FM()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # 将损失写入 TensorBoard
            writer.add_scalar('Loss/Meta_Train_Loss', meta_train_loss.item(), epoch * sd_batch_num + batch_idx)
            writer.add_scalar('Loss/Meta_Test_Loss', meta_test_loss.item(), epoch * sd_batch_num + batch_idx)
            writer.add_scalar('Loss/Total_Loss', total_loss.item(), epoch * sd_batch_num + batch_idx)
            # original_images = vutils.make_grid(test_images, normalize=True, scale_each=True)
            # reconstructed_images = vutils.make_grid(rec_out, normalize=True, scale_each=True)
            # writer.add_image('Original_Images', original_images, epoch * batch_num + batch_idx)
            # writer.add_image('Reconstructed_Images', reconstructed_images, epoch * batch_num + batch_idx)
            # if not os.path.exists(rec_dir):
            #     os.makedirs(rec_dir)
            # # 保存重建图像到文件夹
            # for idx in range(rec_out.size(0)):
            #     save_image(rec_out[idx], os.path.join(rec_dir,
            #                                           f'reconstructed_epoch{epoch + 1}_batch{batch_idx + 1}_img{idx + 1}.png'))
            # 保存模型
            if (epoch + 1) % save_interval == 0:
                save_model(model, optimizer, epoch + 1, f'sd_model_epoch_{epoch + 1}.pth')

    # 所有 epoch 结束后保存最终模型
    torch.save({
        'epoch': max_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'final_model.pth')

    # 关闭 SummaryWriter
    writer.close()


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


if __name__ == '__main__':
    # train_FM()

    batch_size = 16
    domain = 't2'
    train_path = f'F:\MetaStyleDate\BRATS-2018\processed_2d_train_bezier\\{domain}'

    # train_case是一个列表，0到13291
    train_case = [i for i in range(13291)]
    print(f"train_case:{train_case[-1]}")
    # train_case_path = r'G:\Pycharm_project\MetaStyle\dataset\BRATS-2018\processed_2d_train_bezier\train.txt'
    # with open(train_case_path, 'r') as f:
    #     for line in f:
    #         train_case.append(line.strip())
    dataset = MyBraTSDataset(train_path, train_case)
    traindataloder = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 初始化模型、优化器和损失函数
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 开始训练
    model.train()
    for epoch in range(50):  # 训练 10 个 epoch
        loss_list = []
        dice_list = []
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
            segmentation_loss = dice_loss(segmentation_output, label)

            # 总损失
            total_loss = segmentation_loss + total_style_loss

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
            segmentation_loss = dice_loss(segmentation_output, label)

            # 总损失
            total_loss = segmentation_loss + total_style_loss
            loss_list.append(total_loss.item())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        loss = np.mean(loss_list)
        if (epoch + 1) % 10 == 0:
            # 保存模型
            torch.save(model.state_dict(), f'./meta_style_unet_{epoch + 1}.pth')
        print(f"Epoch {epoch + 1}, Total Loss: {loss:.4f}")


