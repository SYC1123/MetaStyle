import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms import functional as F # 导入 functional 以便单独使用转换

# --- 设置自定义模型缓存目录 (如果需要) ---
current_working_directory = os.getcwd()
custom_cache_dir = os.path.join(current_working_directory, 'pretrained_models_cache')
os.makedirs(custom_cache_dir, exist_ok=True)
os.environ['TORCH_HOME'] = custom_cache_dir
print(f"模型缓存目录 (TORCH_HOME) 已设置为: {custom_cache_dir}")

if 'torch' in sys.modules or 'torchvision' in sys.modules:
    print("\n警告：torch 或 torchvision 模块已在设置 TORCH_HOME 之前被导入。")
    print("请确保此脚本顶部设置 TORCH_HOME 的代码在任何 'import torch' 或 'import torchvision' 之前执行。")

import torch # 现在可以安全导入
import torchvision.models as models
import torchvision.transforms as transforms

# --- 创建一个示例 .npz 文件 (如果不存在) ---
dummy_npz_path = os.path.join(current_working_directory, 'dummy_medical_image.npz')
if not os.path.exists(dummy_npz_path):
    print(f"创建示例 .npz 文件: {dummy_npz_path}")
    # 创建一个 128x128 的单通道图像，值在 0 到 1000 之间
    dummy_image = np.random.randint(0, 1001, size=(128, 128), dtype=np.uint16)
    dummy_label = np.array([1]) # 示例标签
    np.savez(dummy_npz_path, image=dummy_image, label=dummy_label)
else:
    print(f"使用已存在的示例 .npz 文件: {dummy_npz_path}")

# --- 加载预训练的 ResNet 模型 ---
print("正在加载 ResNet50 模型...")
try:
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights)
    print("模型加载成功。")
    model_file_name = os.path.basename(weights.url)
    expected_path = os.path.join(custom_cache_dir, 'hub', 'checkpoints', model_file_name)
    print(f"预期本地缓存文件路径: {expected_path}")
except Exception as e:
    print(f"加载模型时出错: {e}")
    exit()

model.eval()

# --- 定义医学影像预处理流程 ---

# 1. 获取标准 ResNet 预处理步骤 (Resize, CenterCrop, Normalize)
#    我们需要手动应用它们，因为输入不是 PIL Image
std_transforms = weights.transforms()
# 从Compose中提取各个转换，通常顺序是 Resize, CenterCrop, ToTensor, Normalize
# 我们需要手动执行类似的操作，并跳过 ToTensor
img_size = 224 # ResNet50 v1 通常需要 224x224 输入
# 找到 Resize 和 CenterCrop (注意：具体实现可能因 torchvision 版本而异)
# 通常是 transforms.Resize(256), transforms.CenterCrop(224)
# 我们将直接使用目标尺寸来应用
resize_size = 256 # 通常先 resize 到稍大尺寸
crop_size = img_size

# 获取 ImageNet 的均值和标准差用于归一化
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# --- 加载并预处理单张医学影像 ---
file_path = r'G:\VS_project\Brats-Demo\processed_2d_train_bezier\t2\sample_Brats18_2013_0_1_0_0.npz' # 使用我们创建的或已有的示例文件

try:
    data = np.load(file_path, allow_pickle=True)
    image = data['image'] # 加载单通道图像 NumPy 数组
    label = data['label'] # 加载标签 (此处不用于特征提取)
    print(f"成功从 '{file_path}' 加载图像，原始形状: {image.shape}, 数据类型: {image.dtype}, 标签: {label}")
    # 可视化图像
    import matplotlib.pyplot as plt

    plt.imshow(image, cmap='gray')
    plt.title(f"原图")
    plt.axis('off')
    plt.show()

    # 1. 转换为浮点类型
    image = image.astype(np.float32)

    # 2. 归一化像素值到 [0.0, 1.0] 范围
    #    注意：这是常见的做法，但最佳策略取决于具体医学影像数据的特性
    #    例如，CT 可能需要窗口化(windowing)
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val > min_val:
        image = (image - min_val) / (max_val - min_val)
    else:
        image = np.zeros(image.shape, dtype=np.float32) # 如果图像是恒定的，则设为0
    print(f"图像归一化到 [0.0, 1.0] 范围")

    # 3. 转换为 PyTorch Tensor
    #    形状: [H, W]
    image_tensor = torch.from_numpy(image)

    # 4. 添加通道维度
    #    形状: [1, H, W]
    image_tensor = image_tensor.unsqueeze(0)

    # 5. 复制通道维度以匹配 ResNet 输入 (1 -> 3)
    #    形状: [3, H, W]
    image_tensor = image_tensor.repeat(3, 1, 1)
    print(f"图像转换为 Tensor 并复制通道，形状: {image_tensor.shape}")

    # 6. 应用 Resize 和 CenterCrop (使用 functional API)
    #    注意： F.resize 需要 PIL Image 或 Tensor [C, H, W]
    #    先 Resize 到 resize_size (保持长宽比，短边为 resize_size)
    image_tensor = F.resize(image_tensor, resize_size, antialias=True) # 使用 antialias=True 获得更好效果
    # 再 CenterCrop 到 crop_size
    image_tensor = F.center_crop(image_tensor, (crop_size, crop_size))
    print(f"Resize 和 CenterCrop 后形状: {image_tensor.shape}")

    # 7. 应用 ImageNet 归一化
    #    注意：在医学影像上使用 ImageNet 统计数据是否最优是有争议的，
    #    但这是使用预训练模型的一种常见做法。
    image_tensor = normalize(image_tensor)
    print(f"应用 ImageNet 归一化")

    # 8. 添加 Batch 维度
    #    形状: [1, 3, crop_size, crop_size]
    input_batch = image_tensor.unsqueeze(0)
    print(f"最终输入批次形状: {input_batch.shape}")

except FileNotFoundError:
    print(f"错误：文件未找到 {file_path}")
    exit()
except KeyError as e:
    print(f"错误：.npz 文件中缺少键 '{e}'。请确保文件包含 'image'。")
    exit()
except Exception as e:
    print(f"处理图像时发生错误: {e}")
    exit()

# --- GPU 加速 (如果可用) ---
if torch.cuda.is_available():
    print("CUDA 可用，将模型和数据移至 GPU")
    model = model.to('cuda')
    input_batch = input_batch.to('cuda')
else:
    print("CUDA 不可用，使用 CPU")

# --- 定义 Hook 函数和注册 Hook ---
features = {}
def get_features_hook(module, input, output):
    features['shallow_features'] = output.detach()

# 选择目标浅层 (例如 model.layer1)
target_layer = model.layer1
hook_handle = target_layer.register_forward_hook(get_features_hook)

# --- 执行前向传播 ---
print("执行前向传播以提取特征...")
with torch.no_grad():
    output = model(input_batch) # 使用预处理后的医学影像批次

# --- 移除 Hook ---
hook_handle.remove()

# --- 访问提取到的特征 ---
if 'shallow_features' in features:
    shallow_features_tensor = features['shallow_features']
    print("\n成功提取浅层特征！")
    print(f"特征张量的形状: {shallow_features_tensor.shape}")
    print(f"特征张量所在设备: {shallow_features_tensor.device}")

    # (可选) 将特征移回 CPU
    # shallow_features_cpu = shallow_features_tensor.cpu()
    # print(f"将特征移至 CPU (如果需要): {shallow_features_cpu.device}")

else:
    print("\n未能提取到特征，请检查 Hook 是否正确注册或模型结构。")

print(f"\n脚本执行完毕。")
