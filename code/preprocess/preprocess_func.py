import os

import SimpleITK as sitk
import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm

from bezier_curve import bezier_curve
from code.util.config import load_config
from code.util.util import color_map

# modality_name_list = {'t1': '_t1.nii.gz',
#                       't1ce': '_t1ce.nii.gz',
#                       't2': '_t2.nii.gz',
#                       'flair': '_flair.nii.gz'}
modality_name_list = {'t1': '_t1.nii',
                      't1ce': '_t1ce.nii',
                      't2': '_t2.nii',
                      'flair': '_flair.nii'}

def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int)
    resampler.SetReferenceImage(itkimage)
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    return itkimgResampled


def save_img(slice, label, dir):
    np.savez_compressed(dir, image=slice, label=label)


def norm(slices):
    max = np.max(slices)
    min = np.min(slices)
    slices = 2 * (slices - min) / (max - min) - 1
    return slices


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


def main(data_root, modality, target_root):
    # 颜色映射
    cmap = color_map(n_color=256, normalized=False).reshape(-1)
    list_dir = os.listdir(data_root)
    tbar = tqdm(list_dir, ncols=70)
    count = 14079
    for name in tbar:
        # 原始图片和标签， 加载NIfTI 文件
        nib_img = nib.load(os.path.join(data_root, name, name + modality_name_list[modality]))
        # nib_mask = nib.load(os.path.join(data_root, name, name + '_seg.nii.gz'))
        nib_mask = nib.load(os.path.join(data_root, name, name + '_seg.nii'))

        affine = nib_img.affine.copy()
        # 获取三维MRI图片的numpy数组
        slices = nib_img.get_fdata()
        masks = nib_mask.get_fdata()
        # 将多个mask合成为一个mask,便于处理
        masks[masks != 0] = 1

        # 贝塞尔曲线非线性变换
        slices = norm(slices)
        slices, nonlinear_slices_1, nonlinear_slices_2, \
            nonlinear_slices_3, nonlinear_slices_4, nonlinear_slices_5 = nonlinear_transformation(slices)

        label_dir = "../../results/original"
        # 源域相似域和源域非相似域
        if not os.path.exists(os.path.join(target_root, modality + '_ss_train')):
            os.makedirs(os.path.join(target_root, modality + '_ss_train'))
        if not os.path.exists(os.path.join(target_root, modality + '_ss_test')):
            os.makedirs(os.path.join(target_root, modality + '_ss_test'))
        if not os.path.exists(os.path.join(target_root, modality + '_sd_train')):
            os.makedirs(os.path.join(target_root, modality + '_sd_train'))
        if not os.path.exists(os.path.join(target_root, modality + '_sd_test')):
            os.makedirs(os.path.join(target_root, modality + '_sd_test'))

        # if not os.path.exists(os.path.join(label_dir, modality)):
        #     os.makedirs(os.path.join(label_dir, modality))


        for i in range(slices.shape[2]):
            """
            Source-Similar
            """
            """
            这里是用第三维度的数据进行循环，读取体积数据切片后的第i张图片
            
            在这段代码中，slices和masks是3D数据，表示的是体积数据，其中每个体素（voxel）代表一个三维空间的点。
            因此，slices[:, :, i]和masks[:, :, i]表示体积数据中的第i层，即将该体积数据切片后的第i张图片。
            在这个实现中，对于每个切片，都会将其与对应的标签一起保存为一个.npz文件。这样实现的目的可能是方便之后的读取和处理。
            
            在3D体积数据中，第三维通常表示切片的数量或深度。在医学图像中，第三维通常表示扫描的层数，也就是病人的解剖结构在z轴方向的分割层数。
            每个切片都是二维图像，具有相同的高度和宽度，但每个切片之间可能具有不同的解剖特征。
            通过查看3D数据，可以获得一个完整的3D解剖结构的表示，可以用于医学诊断、科学研究等领域。
            """

            if np.sum(masks[:, :, i]) != 0:
                # print(f"mask:{masks[:,:,i]}")
                # print(f"mask.shape:{masks[:, :, i].shape}")
                # print(f"mask.sum:{np.sum(masks[:, :, i])}")
                save_img(slices[:, :, i], masks[:, :, i],
                         os.path.join(target_root, modality + '_ss_train', 'sample{}_0.npz'.format(count)))
                save_img(nonlinear_slices_2[:, :, i], masks[:, :, i],
                         os.path.join(target_root, modality + '_ss_train', 'sample{}_1.npz'.format(count)))
                save_img(nonlinear_slices_4[:, :, i], masks[:, :, i],
                         os.path.join(target_root, modality + '_ss_test', 'sample{}_2.npz'.format(count)))
                """
                Source-Dissimilar
                """
                save_img(nonlinear_slices_1[:, :, i], masks[:, :, i],
                         os.path.join(target_root, modality + '_sd_train', 'sample{}_0.npz'.format(count)))
                save_img(nonlinear_slices_3[:, :, i], masks[:, :, i],
                         os.path.join(target_root, modality + '_sd_train', 'sample{}_1.npz'.format(count)))
                save_img(nonlinear_slices_5[:, :, i], masks[:, :, i],
                         os.path.join(target_root, modality + '_sd_test', 'sample{}_2.npz'.format(count)))
                count += 1

                # 画图检查
                # pred_mask = Image.fromarray(np.uint8(masks[:, :, i].T))
                # pred_mask = pred_mask.convert('P')
                # pred_mask.putpalette(cmap)
                # pred_mask.save(os.path.join(label_dir, modality, 'mask{}.png'.format(count)))



def save_test_npz(data_root, modality, target_root):
    list_dir = os.listdir(data_root)
    tbar = tqdm(list_dir, ncols=70)
    count = 0

    for name in tbar:
        nib_img = nib.load(os.path.join(data_root, name, name + modality_name_list[modality]))
        nib_mask = nib.load(os.path.join(data_root, name, name + '_seg.nii'))

        affine = nib_img.affine.copy()

        slices = nib_img.get_fdata()
        masks = nib_mask.get_fdata()
        masks[masks != 0] = 1

        slices = norm(slices)

        if not os.path.exists(os.path.join(target_root, modality)):
            os.makedirs(os.path.join(target_root, modality))

        for i in range(slices.shape[2]):
            if np.sum(masks[:, :, i]) > 150:
                save_img(slices[:, :, i], masks[:, :, i],
                         os.path.join(target_root, modality, 'test_sample{}.npz'.format(count)))
                count += 1


if __name__ == '__main__':
    # 加载 YAML 文件
    config = load_config()
    # # 训练
    train_data_root = config['train_FM']['split_train_dir']
    train_target_root = config['train_FM']['target_train_dir']
    modality = config['train_FM']['modality']
    main(train_data_root, modality, train_target_root)
    # 测试
    # test_data_root = config['test']['split_test_dir']
    # test_target_root = config['test']['target_test_dir']
    # modality_list = ['flair', 't1', 't1ce']
    # for modality in modality_list:
    #     save_test_npz(test_data_root, modality, test_target_root)

