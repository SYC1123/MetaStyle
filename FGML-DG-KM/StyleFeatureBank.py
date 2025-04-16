
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ops import *
# from code.util.losses import dice_loss
# from meta_style import StyleFeatureBank


# 定义或扩展 StyleFeatureBank 类
class StyleFeatureBank(nn.Module):
    def __init__(self):
        super(StyleFeatureBank, self).__init__()
        self.feature_stats = {}  # 存储层名称到统计列表的映射，键为层名称，值为 {'mean': [], 'std': []}
        self.layer_names = []  # 跟踪层名称列表

    def compute_statistics(self, features):
        """计算给定特征每个通道的均值和标准差"""
        mean = features.mean(dim=[2, 3], keepdim=True)  # [batch_size, channels, 1, 1]
        std = features.std(dim=[2, 3], keepdim=True, unbiased=False)  # [batch_size, channels, 1, 1]
        return mean, std

    def add_statistics(self, layer_name, mean, std):
        """添加指定层的统计信息，按样本级别存储"""
        if layer_name not in self.feature_stats:
            self.feature_stats[layer_name] = {'mean': [], 'std': []}
            self.layer_names.append(layer_name)  # 跟踪层名称

        # 遍历批次中的每个样本，单独存储每个样本的统计
        B = mean.size(0)  # batch_size
        for i in range(B):
            mean_i = mean[i].detach().cpu()  # 形状 [channels, 1, 1]
            std_i = std[i].detach().cpu()    # 形状 [channels, 1, 1]
            self.feature_stats[layer_name]['mean'].append(mean_i)
            self.feature_stats[layer_name]['std'].append(std_i)


    def get_vector(self, layer_name):
        """获取指定层的统计向量"""
        if layer_name in self.feature_stats:
            mean_list = torch.stack(self.feature_stats[layer_name]['mean'])  # [total_num_samples, channels, 1, 1]
            std_list = torch.stack(self.feature_stats[layer_name]['std'])  # [total_num_samples, channels, 1, 1]
            return mean_list, std_list
        else:
            raise ValueError(f"No stats found for layer: {layer_name}")


    def mix_statistics(self, target_mean, target_std, source_mean_list, source_std_list):
        """混合目标统计和源统计（随机选择源样本）"""
        # source_mean_list 和 source_std_list 形状为 [num_source_samples, channels, 1, 1]
        # target_mean 和 target_std 形状为 [batch_size, channels, 1, 1]
        batch_size = target_mean.size(0) # 4
        num_source_samples = source_mean_list.size(0) # 24

        # 随机选择源样本（与您的代码类似，随机 permute 并取前 batch_size 个）
        idx = torch.randperm(num_source_samples)[:batch_size]
        selected_source_mean = source_mean_list[idx].to(target_mean.device)  # 移到相同设备
        selected_source_std = source_std_list[idx].to(target_std.device)

        # 混合：简单平均或加权平均（这里使用等权重平均作为示例，您可以自定义）
        mixed_mean = (target_mean + selected_source_mean) / 2.0
        mixed_std = (target_std + selected_source_std) / 2.0  # 或使用更复杂的混合策略

        return mixed_mean, mixed_std

    def apply_meta_style(self, content_feat, style_mean, style_std):
        """应用自适应实例归一化 (AdaIN) 来调整特征"""
        # content_feat: 当前特征 [batch_size, channels, H, W]
        # style_mean 和 style_std: 混合统计 [batch_size, channels, 1, 1]
        content_mean = content_feat.mean(dim=[2, 3], keepdim=True)
        content_std = content_feat.std(dim=[2, 3], keepdim=True, unbiased=False)
        sigma = content_std.sqrt() + 1e-5  # 避免除以零
        normalized_feat = (content_feat - content_mean) / sigma
        styled_feat = normalized_feat * style_std.sqrt() + style_mean  # 应用风格统计
        return styled_feat

    def get_avg_stats(self, layer_name):
        """获取指定层的平均统计（用于 eval 模式）"""
        if layer_name in self.feature_stats:
            mean_list = torch.stack(self.feature_stats[layer_name]['mean']).mean(dim=0)
            std_list = torch.stack(self.feature_stats[layer_name]['std']).mean(dim=0)
            return mean_list, std_list
        else:
            raise ValueError(f"No stats found for layer: {layer_name}")

    def clear_stats(self):
        """清空所有统计信息"""
        self.feature_stats.clear()
        self.layer_names.clear()

    def save_style_bank(self, path):
        """保存统计信息到文件"""
        torch.save(self.feature_stats, path)

    def load_style_bank(self, path):
        """加载统计信息从文件"""
        self.feature_stats = torch.load(path)
        # 重新构建 layer_names（如果需要）
        self.layer_names = list(self.feature_stats.keys())
