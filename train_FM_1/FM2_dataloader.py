import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import shutil
import os


class BraTSDataset(Dataset):
    def __init__(self, data_dir_list, mode=None):
        self.mode = mode
        self.file_paths = []
        # 源相似域和源非相似域两个文件夹全部读取到一个dataset里面
        # print(f"list:{data_dir_list}")
        for data_path in data_dir_list:
            file_list = os.listdir(data_path)
            self.file_paths.extend([os.path.join(data_path, file) for file in file_list])
        # print(f"file_path:{self.file_paths}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        _, image_name = os.path.split(file_path)
        data = np.load(file_path, allow_pickle=True)
        image = data['image']
        label = data['label']
        # 转换为tensor
        image_tensor = torch.tensor(image)
        label_tensor = torch.tensor(label)
        image_tensor = image_tensor.unsqueeze(0)
        label_tensor = label_tensor.unsqueeze(0)

        if self.mode == 'test':
            return image_tensor, label_tensor, image_name.replace('.npz', '')
        else:
            return image_tensor, label_tensor


class MyBraTSDataset1(Dataset):
    def __init__(self, path, cases, domainid, mode='train'):
        self.mode = mode
        self.path = path
        self.cases = cases
        self.domainid = domainid

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]

        def mask_to_tensor(mask, size=(128, 128)):
            mask = np.array(mask)  # 转换为numpy数组
            mask = torch.tensor(mask, dtype=torch.float32)  # 转换为PyTorch张量
            mask = mask.unsqueeze(0)  # 增加通道维度
            return mask

        if self.mode == 'train':

            path = os.path.join(self.path, '{}_{}.npz'.format(case, self.domainid))

            data0 = np.load(path)
            image0 = data0['image']
            image0 = np.array(image0)

            mask = data0['label']
            # print(type(mask))
            mask = np.array(mask)

            # print(np.unique(image0))

            image0 = Image.fromarray(image0)
            mask = Image.fromarray(mask)

            # print(type(img),type(mask))  # <class 'PIL.Image.Image'> <class 'PIL.Image.Image'>
            transform = transforms.Compose([
                transforms.Resize((128, 128))
            ])
            transform1 = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            img0 = transform1(image0)

            mask = transform(mask)
            mask = mask_to_tensor(mask)

            return img0, mask
        else:
            image_path = os.path.join(self.path, case)
            data = np.load(image_path)
            image = data['image']
            mask = data['label']
            image = np.array(image)
            mask = np.array(mask)

            image = Image.fromarray(image)
            mask = Image.fromarray(mask)
            transform = transforms.Compose([
                transforms.Resize((128, 128))
            ])
            transform1 = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            image = transform1(image)
            mask = transform(mask)
            mask = mask_to_tensor(mask)
            return image, mask


if __name__ == '__main__':
    domain = 't2'
    train_path = f'G:\VS_project\Brats-Demo\processed_2d_train_num_bezier\{domain}'
    # train_case是一个列表，0到13291
    train_case = [i for i in range(len(os.listdir(train_path)) // 6)]
    print(f"train_case:{train_case[-1]}")
    train_case = train_case[:10]
    print(train_case)
    batch_size = 1
    tasks = []
    for domainid in range(6):
        traindataset = MyBraTSDataset1(train_path, train_case, domainid, mode='train')
        traindataloder=DataLoader(traindataset, batch_size=batch_size, shuffle=True, drop_last=True)
        print(len(traindataloder))
        tasks.append(traindataloder)
    # print(tasks)
    for data1,data2 in zip(tasks[0],tasks[1]):
        img1, mask1 = data1
        img2, mask2 = data2
        print(img1.shape)
        print(img2.shape)
        print(mask1.shape)
        print(mask2.shape)
        break
