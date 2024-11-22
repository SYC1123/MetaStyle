import yaml
import os

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构建 config.yaml 文件的绝对路径
config_path = os.path.join(current_dir, '..', 'config.yaml')

def load_config():
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config