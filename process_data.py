import shutil

import SimpleITK as sitk
from PIL import Image
import numpy as np
import os

def tif_to_nii(input_folder: str, output_nii_file: str):
    # 获取文件夹中的所有.tif文件
    tif_files = [f for f in os.listdir(input_folder) if f.endswith('dataset.tif')]

    # 读取所有.tif图像并将它们叠加成3D体积
    image_stack = []
    for tif_file in sorted(tif_files):
        tif_image = Image.open(os.path.join(input_folder, tif_file))
        num_images = tif_image.n_frames
        for i in range(num_images):
            tif_image.seek(i)  # 跳转到第i帧
            current_image = tif_image.copy()  # 复制当前帧
            image_stack.append(current_image)

    # 创建SimpleITK图像
    nii_image = sitk.GetImageFromArray(np.array(image_stack))

    # 如果需要，设置图像的原点、方向和间隔等元数据信息
    nii_image.SetOrigin((0.0, 0.0, 0.0))
    nii_image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    nii_image.SetSpacing((1.0, 1.0, 1.0))

    # 保存为.nii.gz文件
    sitk.WriteImage(nii_image, f"{input_folder}/{output_nii_file}")

def jpg_to_nii(input_folder: str, output_nii_file: str):
    # 获取文件夹中的所有.jpg文件
    jpg_files = [f for f in os.listdir(f"{input_folder}/ground_truth") if f.endswith('.jpg')]

    # 读取所有.jpg图像并将它们叠加成3D体积
    image_stack = []
    for jpg_file in sorted(jpg_files):
        jpg_image = Image.open(os.path.join(f"{input_folder}/ground_truth", jpg_file)).convert('L')
        jpg_array = np.array(jpg_image)
        # 阈值处理
        threshold = jpg_array.mean() + jpg_array.std()
        jpg_array[jpg_array < threshold] = 0
        jpg_array[jpg_array >= threshold] = 1
        image_stack.append(jpg_array)

    # 创建SimpleITK图像
    nii_image = sitk.GetImageFromArray(np.array(image_stack))

    # 如果需要，设置图像的原点、方向和间隔等元数据信息
    nii_image.SetOrigin((0.0, 0.0, 0.0))
    nii_image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    nii_image.SetSpacing((1.0, 1.0, 1.0))

    # 保存为.nii.gz文件
    sitk.WriteImage(nii_image, f"{input_folder}/{output_nii_file}")


path = ['0.1', '0.2', '0.05', '0.05,1600', '0.15',
            '2,1000', '2,1000,500', 'cell', 'epl', 'gfp']

def convert():
    for idx, pt in enumerate(path):
        print(f'now process {idx + 1}st folder: {pt}')
        tif_to_nii(pt, "SMLM_%03d.nii.gz" % (idx + 1))
        jpg_to_nii(pt, "SMLM_label_%03d.nii.gz" % (idx + 1))
        print(f'{idx + 1}st folder finished!')

def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)

def move():
    imageTr_dir = r"E:\pyCharm\Project\nnUNet-master\nnUNet_raw\Dataset002_SMLM\imagesTr"
    labelsTr_dir = r"E:\pyCharm\Project\nnUNet-master\nnUNet_raw\Dataset002_SMLM\labelsTr"
    del_files(imageTr_dir)
    del_files(labelsTr_dir)
    for idx, pt in enumerate(path):
        idx = idx + 1
        name = "SMLM_%03d.nii.gz" % idx
        label = "SMLM_label_%03d.nii.gz" % idx
        shutil.copy(os.path.join(pt, name), imageTr_dir + "/SMLM_%03d_0000.nii.gz" % idx)
        shutil.copy(os.path.join(pt, label), f"{labelsTr_dir}/{name}")
        print(f"idx {idx} has moved")

if __name__ == '__main__':
    convert()
    move()

