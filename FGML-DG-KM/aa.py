import torch
import itertools
from torch.utils.data import DataLoader, Dataset

# --- 1. 准备工作：创建示例数据集和 DataLoader ---
# (在你的实际代码中，替换成你自己的 Dataset 和 DataLoader)

# 假设你的数据是图像数据 (C, W, H)
C, W, H = 3, 32, 32 # 示例通道、宽度、高度

# 定义一个通用的示例数据集
class DomainDataset(Dataset):
    def __init__(self, num_samples, domain_id):
        # 创建一些虚拟图像数据
        self.data = torch.randn(num_samples, C, W, H)
        # 创建一些虚拟标签（例如，域标签或者可以是类别标签）
        self.labels = torch.full((num_samples,), fill_value=domain_id, dtype=torch.long)
        self.domain_id = domain_id
        print(f"Created Dataset for Domain {domain_id} with {num_samples} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 返回单个数据样本和对应的标签
        return self.data[idx], self.labels[idx]

# 创建6个不同域的数据集 (可以有不同的大小)
num_domains = 6
datasets = [
    DomainDataset(num_samples=100, domain_id=0),
    DomainDataset(num_samples=100, domain_id=1),
    DomainDataset(num_samples=100, domain_id=2),
    DomainDataset(num_samples=100, domain_id=3),
    DomainDataset(num_samples=100, domain_id=4),
    DomainDataset(num_samples=100, domain_id=5),
]

# --- 关键：为每个 DataLoader 设置 batch_size=1 ---
batch_size_per_loader = 1

# 创建6个 DataLoader
dataloaders = [
    DataLoader(ds, batch_size=batch_size_per_loader, shuffle=True, num_workers=0)
    for ds in datasets
]
print(f"\nCreated {len(dataloaders)} DataLoaders, each with batch_size={batch_size_per_loader}.")

# --- 2. 创建循环迭代器 ---

# 使用 itertools.cycle 来确保即使某个dataloader耗尽，也能继续从头开始
# 使用 iter() 获取每个dataloader的迭代器
cycled_loaders = [itertools.cycle(iter(dl)) for dl in dataloaders]

# 使用 zip 将这些循环迭代器聚合起来
# 每次迭代 zip 会从每个迭代器中取一项 (一个 batch_size=1 的批次)
combined_loader_iter = zip(*cycled_loaders)

# --- 3. 训练循环示例 ---

num_training_steps = 50 # 你希望进行多少次训练迭代

print(f"\nStarting training loop for {num_training_steps} steps.")
print(f"Each step will sample 1 item (batch_size=1) from each of the {num_domains} dataloaders.")
print(f"These {num_domains} items will be combined into a single batch.")
print("-" * 30)

for step in range(num_training_steps):
    # --- 4. 从每个 DataLoader 获取一个样本 (大小为1的批次) ---
    try:
        # next(combined_loader_iter) 返回一个元组，包含来自每个 loader 的批次
        # 例如: batch_tuple = (batch_loader0, batch_loader1, ..., batch_loader5)
        # 其中 batch_loaderX 是一个 (data, label) 元组
        # 因为 batch_size=1, data 的形状是 [1, C, W, H], label 的形状是 [1]
        batch_tuple = next(combined_loader_iter)

    except StopIteration:
        # 理论上，因为使用了 itertools.cycle，这里不应该触发 StopIteration
        # 除非某个 DataLoader 本身就是空的
        print("Warning: A DataLoader might be empty initially.")
        break # 退出循环或采取其他措施

    # --- 5. 提取并组合数据 ---
    # batch_tuple 结构: ((data0, label0), (data1, label1), ..., (data5, label5))

    # 提取所有数据张量 (每个形状为 [1, C, W, H])
    # data_tensors 是一个包含 6 个张量的列表
    data_tensors = [batch[0] for batch in batch_tuple]

    # 提取所有标签张量 (每个形状为 [1])
    # label_tensors 是一个包含 6 个张量的列表
    label_tensors = [batch[1] for batch in batch_tuple]

    # 使用 torch.cat 将数据张量列表沿第一个维度 (batch 维度) 拼接起来
    # 输入: 6 个 [1, C, W, H] 张量
    # 输出: 1 个 [6, C, W, H] 张量
    final_batch_data = torch.cat(data_tensors, dim=0)

    # 对标签也进行同样的操作
    # 输入: 6 个 [1] 张量
    # 输出: 1 个 [6] 张量
    final_batch_labels = torch.cat(label_tensors, dim=0)

    # --- 6. 模型训练步骤 ---
    # 现在 final_batch_data 的形状是 [6, C, W, H]
    # final_batch_labels 的形状是 [6]
    # 你可以将它们用于你的模型训练

    # print(f"Training Step {step + 1}/{num_training_steps}")
    # print(f"  Final Data Batch Shape: {final_batch_data.shape}")
    # print(f"  Final Labels Batch Shape: {final_batch_labels.shape}")
    # print(f"  Labels in batch: {final_batch_labels.tolist()}") # 显示标签，检查是否来自不同域

    # 在这里添加你的模型训练代码:
    # model.train()
    # optimizer.zero_grad()
    # outputs = model(final_batch_data)
    # loss = criterion(outputs, final_batch_labels) # 或者根据你的需要处理标签
    # loss.backward()
    # optimizer.step()

    if (step + 1) % 10 == 0: # 每10步打印一次信息
       print(f"Step {step + 1}/{num_training_steps} processed. Batch shape: {final_batch_data.shape}")


print("\nTraining loop finished.")

# --- 注意事项 ---
# 1.  **Dataset `__getitem__`**: 确保你的 `Dataset` 的 `__getitem__` 方法返回的是单个样本的数据和标签，而不是一个批次。DataLoader 会负责将它们（在这个案例中是单个）组成批次。
# 2.  **标签处理**: 代码中假设每个 `Dataset` 返回 `(data, label)`。`final_batch_labels` 将包含来自 6 个不同域的 6 个标签。你需要根据你的任务目标决定如何使用这些标签（例如，是否需要域分类，或者标签本身是类别等）。
# 3.  **内存**: 这种方法每次只加载 6 个样本到内存中进行拼接，内存效率较高。
# 4.  **性能**: 如果数据加载（从磁盘读取、预处理等）是瓶颈，使用 `num_workers > 0` 在 `DataLoader` 中可以提高性能。
