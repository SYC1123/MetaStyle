import torch


# 元训练时的特征统计量bank
class StyleFeatureBank:
    def __init__(self):
        self.mean_bank = []
        self.std_bank = []

    def add_statistics(self, mean, std):
        # print('mean:', mean.shape)
        # print('std:', std.shape)
        self.mean_bank.append(mean)
        self.std_bank.append(std)

    def get_statistics(self):
        return torch.stack(self.mean_bank), torch.stack(self.std_bank)

    def get_vector(self):
        # print(len(self.mean_bank), len(self.std_bank))
        mean_vector = torch.mean(torch.stack(self.mean_bank), dim=0)
        std_vector = torch.mean(torch.stack(self.std_bank), dim=0)
        return mean_vector, std_vector


    # 计算均值和标准差
    def compute_statistics(self,x, eps=1e-6):
        mean = torch.mean(x, dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + eps).sqrt()
        mean, sig = mean.detach(), sig.detach()

        return mean, sig


    # 将元训练阶段保存的风格均值和方差与当前测试特征图混合
    def mix_statistics(self,current_mean, current_std, style_mean, style_std, lambda_=0.5):
        mixed_mean = lambda_ * current_mean + (1 - lambda_) * style_mean
        mixed_std = lambda_ * current_std + (1 - lambda_) * style_std
        return mixed_mean, mixed_std


    # 使用混合后的均值和方差对当前特征图进行标准化
    def apply_meta_style(self,features, mixed_mean, mixed_std):
        # 标准化
        normalized_features = (features - torch.mean(features, dim=[0, 2, 3], keepdim=True)) / (
                    torch.std(features, dim=[0, 2, 3], keepdim=True) + 1e-6)
        # print(f"normalized_features{normalized_features.shape}")
        # 应用混合后的均值和方差
        return normalized_features * mixed_std + mixed_mean

    def clean_style_feature_bank(self):
        self.mean_bank.clear()
        self.std_bank.clear()