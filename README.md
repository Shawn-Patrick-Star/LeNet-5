# LeNet-5加入不同归一化方法的模型训练

数据集: CIFAR10(dataset/)

平台: Windows

环境: python3.10.14 pytorch


## 目录结构
```shell
.
├── dataset
│   └── cifar-10-batches-py
├── main.py         # 主程序入口
├── model.py        # LeNet-5模型(加入不同归一化方法)
├── plot.py         # 绘制训练过程中的loss和accuracy
├── stats.py        # 用于打印输出的相关函数
├── useModel.py     # 使用模型进行预测
├── myDataset.py    # 自定义数据集,但是没有使用
└── README.md
```