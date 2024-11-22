# 定义保存模型的函数
import os

import numpy as np
import torch

outdir = 'outdir/checkpoint'
def save_model(model, optimizer, epoch, file_name):
    # 确保目录存在
    os.makedirs(outdir, exist_ok=True)
    file_path = os.path.join(outdir, file_name)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, file_path)


def color_map(n_color=256, normalized=False):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((n_color, 3), dtype=dtype)
    for i in range(n_color):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap