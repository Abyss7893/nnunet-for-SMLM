# nnunet-for-SMLM
### 数据预处理

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

### 模型训练

代码位于 `nnunet_SMLM` 中。
