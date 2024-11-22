import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


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


class MyBraTSDataset(Dataset):
    def __init__(self, path, cases, mode='train'):
        self.mode = mode
        self.path = path
        self.cases = cases

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
            s0_path = os.path.join(self.path, '{}_0.npz'.format(case))
            s1_path = os.path.join(self.path, '{}_1.npz'.format(case))
            s2_path = os.path.join(self.path, '{}_2.npz'.format(case))
            s3_path = os.path.join(self.path, '{}_3.npz'.format(case))
            s4_path = os.path.join(self.path, '{}_4.npz'.format(case))
            s5_path = os.path.join(self.path, '{}_5.npz'.format(case))

            data0 = np.load(s0_path)
            image0 = data0['image']
            image0 = np.array(image0)

            data1 = np.load(s1_path)
            image1 = data1['image']
            image1 = np.array(image1)

            data2 = np.load(s2_path)
            image2 = data2['image']
            image2 = np.array(image2)

            data3 = np.load(s3_path)
            image3 = data3['image']
            image3 = np.array(image3)

            data4 = np.load(s4_path)
            image4 = data4['image']
            image4 = np.array(image4)

            data5 = np.load(s5_path)
            image5 = data5['image']
            image5 = np.array(image5)

            mask = data0['label']
            # print(type(mask))
            mask = np.array(mask)

            # print(np.unique(image0))

            image0 = Image.fromarray(image0)
            image1 = Image.fromarray(image1)
            image2 = Image.fromarray(image2)
            image3 = Image.fromarray(image3)
            image4 = Image.fromarray(image4)
            image5 = Image.fromarray(image5)
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
            img1 = transform1(image1)
            img2 = transform1(image2)
            img3 = transform1(image3)
            img4 = transform1(image4)
            img5 = transform1(image5)

            mask = transform(mask)
            mask = mask_to_tensor(mask)

            return img0, img1, img2, img3, img4, img5, mask
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
    train_path = f'F:\MetaStyleDate\BRATS-2018\processed_2d_train_bezier\\{domain}'
    train_case = []
    train_case_path = r'F:\MetaStyleDate\BRATS-2018\processed_2d_train_bezier\train.txt'
    with open(train_case_path, 'r') as f:
        for line in f:
            train_case.append(line.strip())
    # 得到每一个case的切片的数量 sample_Brats18_2013_0_1_0_0.npz---sample_Brats18_2013_0_1_？？_0.npz

    dataset = MyBraTSDataset(train_path, train_case)
    # # 查看数据集的第一个数据
    # traindataloder = DataLoader(dataset, batch_size=8, shuffle=True)
    # for i, (img0, img1, img2, img3, img4, img5, mask) in enumerate(traindataloder):
    #     print(img0.shape, img1.shape, img2.shape, img3.shape, img4.shape, img5.shape, mask.shape)
    #     break
    id = 0
    for case in train_case:
        slice_files = [f for f in os.listdir(train_path) if f.startswith(f'sample_{case}_') and f.endswith('.npz')]
        # print(slice_files)
        # print(len(slice_files))
        for i in range(len(slice_files) // 6):
            replace_str = f'sample_{case}_' + str(i) + '_'
            for filename in slice_files:
                if filename.startswith(replace_str):
                    # 将slice_files中的replace_str开头的所有文件的名字中的replace_str替换为id
                    old_name = os.path.join(train_path, filename)
                    new_name = os.path.join(train_path, filename.replace(replace_str, str(id) + '_'))
                    # print(new_name)
                    os.rename(old_name, new_name)
            id += 1
    print(id)
