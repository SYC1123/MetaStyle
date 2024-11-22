import shutil

from code.util.config import load_config
import os
from sklearn.model_selection import train_test_split

config = load_config()
data_root = config['train_FM']['data_root']
all_files = os.listdir(data_root)

train_dirs, test_dirs = train_test_split(all_files, test_size=0.2, random_state=42)

split_train_dir = config['train_FM']['split_train_dir']
split_test_dir = config['test']['split_test_dir']
os.makedirs(split_train_dir, exist_ok=True)
os.makedirs(split_test_dir, exist_ok=True)

# 复制训练集文件夹
for subdir in train_dirs:
    source_dir = os.path.join(data_root, subdir)
    target_dir = os.path.join(split_train_dir, subdir)
    shutil.copytree(source_dir, target_dir)

# 复制测试集文件夹
for subdir in test_dirs:
    source_dir = os.path.join(data_root, subdir)
    target_dir = os.path.join(split_test_dir, subdir)
    shutil.copytree(source_dir, target_dir)

print(f"训练集文件夹数: {len(train_dirs)}")
print(f"测试集文件夹数: {len(test_dirs)}")
