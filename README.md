# nnunet-for-SMLM
## 数据预处理

独立出来的 `process_data.py` 为处理数据的脚本文件。其中应用到的数据有：

```txt
path = ['0.1', '0.2', '0.05', '0.05,1600', '0.15',
            '2,1000', '2,1000,500', 'cell', 'epl', 'gfp']
```

其中每个数据文件夹的结构为：

```txt
0.1
 |
 | --- ground_truth
 |          | --- frame_xxx.jpg
 | --- Artifical dataset.tif
```

如果想用该代码生成其他数据集的话，只需修改相应的绝对路径为本地路径即可。



## 模型训练

代码位于 `nnunet_SMLM` 中。

### 环境配置

进入后打开文件 `pyproject.toml` 中含有需要的依赖项，将其全部安装即可。其中最后两个 `hiddenlayer` 和 `ipython` 为可视化，不安装也可以运行。



### 数据集储存路径

`nnunet` 对数据集储存有严格的要求，不同数据储存在不同的位置，分为三个文件夹：

* `nnUNet_raw`: 存放原始数据集
* `nnUNet_preprocessed`：存放预处理的数据集，该神经网络会根据这些预处理的内容进行合理调整网络结构
* `nnUNet-results`：存放结果数据

已经将结果数据集放在[云盘](https://bhpan.buaa.edu.cn/link/AA19B283D74EFD46E59D6C452B1F91BA38),下载后直接替换即可。`https://bhpan.buaa.edu.cn/link/AA19B283D74EFD46E59D6C452B1F91BA38`
文件夹名：SMLM_data
有效期限：2023-12-07 06:08





对原始数据集中文件结构进行说明：

```
- **imagesTr** contains the images belonging to the training cases. nnU-Net will perform pipeline configuration, training with 
cross-validation, as well as finding postprocessing and the best ensemble using this data. 
- **imagesTs** (optional) contains the images that belong to the test cases. nnU-Net does not use them! This could just 
be a convenient location for you to store these images. Remnant of the Medical Segmentation Decathlon folder structure.
- **labelsTr** contains the images with the ground truth segmentation maps for the training cases. 
- **dataset.json** contains metadata of the dataset.
```

### 路径变更

打开 `nnunetv2\paths.py` 中找到代码：

```python
nnUNet_raw = r"E:\pyCharm\Project\nnUNet-master\nnUNet_raw"
nnUNet_preprocessed = r"E:\pyCharm\Project\nnUNet-master\nnUNet_preprocessed"
nnUNet_results = r"E:\pyCharm\Project\nnUNet-master\nnUNet_results"
```

将其替换为本地实际路径。



另外需要替换路径的文件还有：



在 `nnunetv2\experiment_planning\plan_and_process_entrypoints.py` 中修改：

```python
import sys
sys.path.append(r'E:\pyCharm\Project\nnUNet-master')
```

为本地实际路径。



在 `nnunet2\run\run_training.py` 中修改：

```python
import sys
sys.path.append(r'E:\pyCharm\Project\nnUNet-master')
```

为本地实际路径。



### 代码运行

进入对应目录下命令行执行：

```bash
python .\run_training.py Dataset002_SMLM 3d_lowres 4
```

即可训练该神经网络。



其中第一个参数 `Dataset002_SMLM` 为数据集名称

第二个参数 `3d_lowres` 为三个可选模型之一，另外两个为 `3d_fullres` 和 `2d`

第三个参数 `4` 意为采取四折交叉验证



成功运行输出：

```txt
Using device: cuda:0

#######################################################################
Please cite the following paper when using nnU-Net:
Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.
#######################################################################


This is the configuration used by this training:
Configuration name: 3d_lowres
 {'data_identifier': 'nnUNetPlans_3d_lowres', 'preprocessor_name': 'DefaultPreprocessor', 'batch_size': 2, 'patch_size': [256, 96, 96], 'median_image_size_in_voxels': [813, 104, 104], 'spacing': [1.2298738654248702, 1.2298738654248702, 1.2298738654248702], 'normalization_schemes': ['Resc
aleTo01Normalization'], 'use_mask_for_norm': [False], 'UNet_class_name': 'PlainConvUNet', 'UNet_base_num_features': 32, 'n_conv_per_stage_encode
r': [2, 2, 2, 2, 2, 2], 'n_conv_per_stage_decoder': [2, 2, 2, 2, 2], 'num_pool_per_axis': [5, 4, 4], 'pool_op_kernel_sizes': [[1, 1, 1], [2, 2, 
2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 1, 1]], 'conv_kernel_sizes': [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]], 'unet_
max_num_features': 320, 'resampling_fn_data': 'resample_data_or_seg_to_shape', 'resampling_fn_seg': 'resample_data_or_seg_to_shape', 'resampling
_fn_data_kwargs': {'is_seg': False, 'order': 3, 'order_z': 0, 'force_separate_z': None}, 'resampling_fn_seg_kwargs': {'is_seg': True, 'order': 1
, 'order_z': 0, 'force_separate_z': None}, 'resampling_fn_probabilities': 'resample_data_or_seg_to_shape', 'resampling_fn_probabilities_kwargs': {'is_seg': False, 'order': 1, 'order_z': 0, 'force_separate_z': None}, 'batch_dice': False, 'next_stage': '3d_cascade_fullres'}

These are the global plan.json settings:
 {'dataset_name': 'Dataset002_SMLM', 'plans_name': 'nnUNetPlans', 'original_median_spacing_after_transp': [1.0, 1.0, 1.0], 'original_median_shap
e_after_transp': [1000, 128, 128], 'image_reader_writer': 'SimpleITKIO', 'transpose_forward': [0, 1, 2], 'transpose_backward': [0, 1, 2], 'exper
iment_planner_used': 'ExperimentPlanner', 'label_manager': 'LabelManager', 'foreground_intensity_properties_per_channel': {'0': {'max': 65535.0, 'mean': 1198.0870361328125, 'median': 424.0, 'min': 217.0, 'percentile_00_5': 419.0, 'percentile_99_5': 15514.0, 'std': 2377.73486328125}}}    

2023-11-07 05:22:42.470831: unpacking dataset...
2023-11-07 05:22:48.317097: unpacking done...
2023-11-07 05:22:48.319865: do_dummy_2d_data_aug: False
2023-11-07 05:22:48.320946: Using splits from existing split file: E:\pyCharm\Project\nnUNet-master\nnUNet_preprocessed\Dataset002_SMLM\splits_final.json
2023-11-07 05:22:48.683050: The split file contains 5 splits.
2023-11-07 05:22:48.684048: Desired fold for training: 4
2023-11-07 05:22:48.685048: This split has 8 training and 2 validation cases.

.....
```



同时也可以在 `nnUNet_results` 中找到对应的日志文件。



### 数据集更改

目前十个数据集全部为训练集，如果想额外加数据的话可以使用之前那个数据转换的脚本生成后添加进来，另外记得修改 `dataet.json` 文件。



如果想知道自己的数据集是否正确时，可以运行 `nnunet2\experiment_planning\verify_dataset_integrity.py` 进行验证。



如果想重新生成预处理的结果数据(也即 `nnUNet_preprocessed` 文件夹里的) 可以运行同级目录下的文件 `plan_and_preprocess_entrypoints.py` 有：

```bash
python plan_and_preprocess_entrypoints.py -d 2
```

其中的 `2` 表示对第二个数据集 `Dataset002` 进行生成。
