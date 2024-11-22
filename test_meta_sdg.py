import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

import logging
import medpy.metric.binary as mmb

from code.dataloader.dataloader import BraTSDataset
from code.model.test.meta_unet import UNet
from code.util.config import load_config
from code.util.util import color_map


# 初始化日志
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("test_result.log"),
                        logging.StreamHandler()
                    ])

def evaluate():
    config = load_config()
    # os.environ['CUDA_VISIBLE_DEVICES'] = config['test']['gpu_ids']
    # print(config['test']['gpu_ids'])
    model_dir = config['test']['model_dir']
    n_classes = config['test']['n_classes']
    test_root = config['test']['test_dir']
    test_domain_list = config['test']['test_domain_list']
    save_label = config['test']['save_label']
    label_dir = config['test']['label_dir']
    real_mask_dir = config['test']['real_mask_dir']
    num_domain = len(test_domain_list)
    # 颜色映射
    cmap = color_map(n_color=256, normalized=False).reshape(-1)
    #
    # 创建保存标签的目录
    if save_label and not os.path.exists(label_dir):
        os.makedirs(label_dir)

    for test_idx in range(num_domain):
        # 初始化模型
        model = UNet()
        model.load_state_dict(torch.load(os.path.join(model_dir, 'model_epoch_50.pth'))['model_state_dict'])
        model = model.to('cuda:2')

        # 加载测试数据
        
        test_dir = os.path.join(test_root, test_domain_list[test_idx])
        # t1_test_dir = os.path.join(test_dir, test_domain_list[1])
        # t1ce_test_dir = os.path.join(test_dir, test_domain_list[2])
        # print(f"flair_test_dir,t1_test_dir,t1ce_test_dir:{flair_test_dir, t1_test_dir, t1ce_test_dir}")
        dataset = BraTSDataset([test_dir], 'test')
        dataloader = DataLoader(dataset, batch_size=config['test']['batch_size'], shuffle=False, num_workers=8, pin_memory=True)

        # 测试模型
        model.eval()
        tbar = tqdm(dataloader, ncols=150)
        total_dice = 0
        total_hd = 0
        total_asd = 0
        for idx, batch in enumerate(tbar):
            sample_data = batch[0].to('cuda:2')
            mask = batch[1].detach().numpy()
            # 确保 mask 是单通道
            mask = mask.squeeze(1)  # 转换形状为 (batch_size, height, width)
            # print(f"mask_shape:{mask.shape}")
        
            sample_data = sample_data.float()
            # print(f'sample_data: {sample_data}')

            output = model(sample_data, mode='eval')
        
            pred_y = torch.sigmoid(output)  # 如果使用 sigmoid，保留原始输出
            pred_y = torch.argmax(pred_y, dim=1) 
            pred_y = pred_y.cpu().detach().numpy()
            
            # 打印预测和真实标签的形状
            # print(f'pred_y : {pred_y[0][0][120]}')
            # print(f'mask: {mask[0][120]}')

            if pred_y.sum() == 0 or mask.sum() == 0:
                total_dice += 0
                total_hd += 100
                total_asd += 100
            else:
                total_dice += mmb.dc(pred_y, mask)
                total_hd += mmb.hd95(pred_y, mask)
                total_asd += mmb.asd(pred_y, mask)

            # 记录日志
            logging.info('Domain: {}, Dice: {}, HD: {}, ASD: {}'.format(
                test_domain_list[test_idx],
                round(100 * total_dice / (idx + 1), 2),
                round(total_hd / (idx + 1), 2),
                round(total_asd / (idx + 1), 2)
            ))

            # 保存标签图像
            if save_label:
                if not os.path.exists(os.path.join(label_dir, test_domain_list[test_idx])):
                    os.makedirs(os.path.join(label_dir, test_domain_list[test_idx]))
                    
                if not os.path.exists(os.path.join(real_mask_dir, test_domain_list[test_idx])):
                    os.makedirs(os.path.join(real_mask_dir, test_domain_list[test_idx]))
                
                for i, (pred_mask, real_mask) in enumerate(zip(pred_y, mask)):
                    # print(pred_mask[120])
                    pred_mask = Image.fromarray(np.uint8(pred_mask.T))
                    pred_mask = pred_mask.convert('P')
                    pred_mask.putpalette(cmap)
                    pred_mask.save(os.path.join(label_dir, test_domain_list[test_idx], str(i) + '.png'))
                    
                    real_mask = Image.fromarray(np.uint8(real_mask.T))
                    real_mask = real_mask.convert('P')
                    real_mask.putpalette(cmap)
                    real_mask.save(os.path.join(real_mask_dir, test_domain_list[test_idx], str(i) + '.png'))

if __name__ == '__main__':
    evaluate()
