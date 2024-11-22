import os
import numpy as np
from bezier_curve import bezier_curve
import nibabel as nib
import numpy as np


def nonlinear_transformation(slices):
    points_1 = [[-1, -1], [-1, -1], [1, 1], [1, 1]]
    xvals_1, yvals_1 = bezier_curve(points_1, nTimes=100000)
    xvals_1 = np.sort(xvals_1)

    points_2 = [[-1, -1], [-0.5, 0.5], [0.5, -0.5], [1, 1]]
    xvals_2, yvals_2 = bezier_curve(points_2, nTimes=100000)
    xvals_2 = np.sort(xvals_2)
    yvals_2 = np.sort(yvals_2)

    points_3 = [[-1, -1], [-0.5, 0.5], [0.5, -0.5], [1, 1]]
    xvals_3, yvals_3 = bezier_curve(points_3, nTimes=100000)
    xvals_3 = np.sort(xvals_3)

    points_4 = [[-1, -1], [-0.75, 0.75], [0.75, -0.75], [1, 1]]
    xvals_4, yvals_4 = bezier_curve(points_4, nTimes=100000)
    xvals_4 = np.sort(xvals_4)
    yvals_4 = np.sort(yvals_4)

    points_5 = [[-1, -1], [-0.75, 0.75], [0.75, -0.75], [1, 1]]
    xvals_5, yvals_5 = bezier_curve(points_5, nTimes=100000)
    xvals_5 = np.sort(xvals_5)

    """
    slices, nonlinear_slices_2, nonlinear_slices_4 are source-similar images
    nonlinear_slices_1, nonlinear_slices_3, nonlinear_slices_5 are source-dissimilar images
    """
    nonlinear_slices_1 = np.interp(slices, xvals_1, yvals_1)
    nonlinear_slices_1[nonlinear_slices_1 == 1] = -1

    nonlinear_slices_2 = np.interp(slices, xvals_2, yvals_2)

    nonlinear_slices_3 = np.interp(slices, xvals_3, yvals_3)
    nonlinear_slices_3[nonlinear_slices_3 == 1] = -1

    nonlinear_slices_4 = np.interp(slices, xvals_4, yvals_4)

    nonlinear_slices_5 = np.interp(slices, xvals_5, yvals_5)
    nonlinear_slices_5[nonlinear_slices_5 == 1] = -1

    return slices, nonlinear_slices_1, nonlinear_slices_2, \
        nonlinear_slices_3, nonlinear_slices_4, nonlinear_slices_5


domin = 'flair'

train_path_save = f'./processed_2d_train_bezier/{domin}'
val_path_save = f'./processed_2d_val_bezier/{domin}'
test_path_save = f'./processed_2d_test_bezier/{domin}'


def save_img(slice, label, dir):
    np.savez_compressed(dir, image=slice, label=label)


def data_process(img_path, seg_path, case, sava_path, modes):
    img = nib.load(img_path)
    seg = nib.load(seg_path)
    img_data = img.get_fdata()
    seg_data = seg.get_fdata()
    count = 0
    for i in range(seg_data.shape[2]):
        mask1 = seg_data[:, :, i]
        img1 = img_data[:, :, i]
        # 归一化到-1到1
        img1 = (img1 - np.min(img1)) / ((np.max(img1) - np.min(img1)) + 1e-7) * 2 - 1
        processed_mask = np.where(mask1 != 0, 1, mask1)
        if np.sum(processed_mask) == 0:
            continue

        if modes == 'train':
            slices, nonlinear_slices_1, nonlinear_slices_2, onlinear_slices_3, nonlinear_slices_4, nonlinear_slices_5 = nonlinear_transformation(
                img1)
            # 可视化
            # plt.imshow(slices, cmap='gray')
            # plt.show()
            # plt.imshow(nonlinear_slices_1, cmap='gray')
            # plt.show()
            # plt.imshow(nonlinear_slices_2, cmap='gray')
            # plt.show()
            # plt.imshow(onlinear_slices_3, cmap='gray')
            # plt.show()
            # plt.imshow(nonlinear_slices_4, cmap='gray')
            # plt.show()
            # plt.imshow(nonlinear_slices_5, cmap='gray')
            # plt.show()
            # # break

            save_img(slices, processed_mask, os.path.join(sava_path, 'sample_{}_{}_{}.npz'.format(case, count, 0)))
            save_img(nonlinear_slices_1, processed_mask,
                     os.path.join(sava_path, 'sample_{}_{}_{}.npz'.format(case, count, 1)))
            save_img(nonlinear_slices_2, processed_mask,
                     os.path.join(sava_path, 'sample_{}_{}_{}.npz'.format(case, count, 2)))
            save_img(onlinear_slices_3, processed_mask,
                     os.path.join(sava_path, 'sample_{}_{}_{}.npz'.format(case, count, 3)))
            save_img(nonlinear_slices_4, processed_mask,
                     os.path.join(sava_path, 'sample_{}_{}_{}.npz'.format(case, count, 4)))
            save_img(nonlinear_slices_5, processed_mask,
                     os.path.join(sava_path, 'sample_{}_{}_{}.npz'.format(case, count, 5)))
            count += 1
        else:
            save_img(img1, processed_mask, os.path.join(sava_path, 'sample_{}_{}.npz'.format(case, count)))
            count += 1
        # plt.imshow(img1, cmap='gray')
        # plt.imshow(processed_mask, cmap='jet', alpha=0.5)
        # plt.title(f'Slice {i}')
        # plt.show()
        # break
    return img_data, seg_data


train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2

train_path = r'G:\VS_project\Brats-Demo\datasets\BraTS_2018\train'
val_path = r'G:\VS_project\Brats-Demo\datasets\BraTS_2018\val'
test_path = r'G:\VS_project\Brats-Demo\datasets\BraTS_2018\test'
data_train = os.listdir(train_path)
data_val = os.listdir(val_path)
data_test = os.listdir(test_path)
# print(data)

# # 打乱数据
# np.random.seed(0)
# np.random.shuffle(data)
# print(data)

# train=data[:int(len(data)*train_ratio)]
# val=data[int(len(data)*train_ratio):int(len(data)*(train_ratio+val_ratio))]
# test=data[int(len(data)*(train_ratio+val_ratio)):]

# 训练集
for case in data_train:
    img_path = os.path.join(train_path, case, case + '_' + domin + '.nii.gz')
    seg_path = os.path.join(train_path, case, case + '_seg.nii.gz')
    print(img_path)
    print(seg_path)
    img_data, seg_data = data_process(img_path, seg_path, case, train_path_save, modes='train')

# 验证集
# for case in data_val:
#     img_path=os.path.join(val_path,case,case+'_'+domin+'.nii.gz')
#     seg_path=os.path.join(val_path,case,case+'_seg.nii.gz')
#     print(img_path)
#     print(seg_path)
#     img_data,seg_data=data_process(img_path,seg_path,case,val_path_save,modes='val')

# 测试集
for case in data_test:
    img_path = os.path.join(test_path, case, case + '_' + domin + '.nii.gz')
    seg_path = os.path.join(test_path, case, case + '_seg.nii.gz')
    print(img_path)
    print(seg_path)
    img_data, seg_data = data_process(img_path, seg_path, case, test_path_save, modes='test')

# train_ori_path_save = f'./processed_2d_train_ori/{domin}'
# # 不过贝塞尔的训练集
# for case in data_train:
#     img_path=os.path.join(train_path,case,case+'_'+domin+'.nii.gz')
#     seg_path=os.path.join(train_path,case,case+'_seg.nii.gz')
#     print(img_path)
#     print(seg_path)
#     img_data,seg_data=data_process(img_path,seg_path,case,train_ori_path_save,modes='val')
