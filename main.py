import sys
import time

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from model import *
from plot import plot
from stats import *

"""
准备数据集，dataloader加载数据集，搭建网络模型，创建网络模型实例，
定义损失函数，定义优化器，设置网络训练的参数，
开始训练，验证模型，最后保存模型。可以将训练结果展示
"""

epoch = 30
batch_size = 64
learning_rate = 1e-2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 创建网络模型
models = [LeNet5, LeNet5_BN, LeNet5_LN, LeNet5_IN, LeNet5_GN]

RES = {
    "LeNet5": {
        "test_loss": [],
        "accuracy": []
    },
    "LeNet5_BN": {
        "test_loss": [],
        "accuracy": []
    },
    "LeNet5_LN": {
        "test_loss": [],
        "accuracy": []
    },
    "LeNet5_IN": {
        "test_loss": [],
        "accuracy": []
    },
    "LeNet5_GN": {
        "test_loss": [],
        "accuracy": []
    }
}


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

train_data = torchvision.datasets.CIFAR10('./dataset', True,
                                          transform=transforms.ToTensor(), download=False)
test_data = torchvision.datasets.CIFAR10('./dataset', False,
                                         transform=transforms.ToTensor(), download=False)

# length 长度
train_data_len = len(train_data)
test_data_len = len(test_data)

# DataLoader 加载数据
train_dataloader = DataLoader(train_data, batch_size)
test_dataloader = DataLoader(test_data, batch_size)


def init(net):
    net.to(device)
    # 损失函数
    loss_func = nn.CrossEntropyLoss()
    loss_func.to(device)
    # 优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    return loss_func, optimizer


def train(net, loss_func, optimizer):
    net.train()     # 对于特定的层会有作用，如果没有不写也行
    total_train_loss = 0
    for data in train_dataloader:
        img, target = data
        img = img.cuda()
        target = target.cuda()

        output = net(img)

        loss = loss_func(output, target)

        # 优化器优化模型
        optimizer.zero_grad()       # 梯度清零
        loss.backward()             # 反向传播求解梯度
        optimizer.step()            # 更新权重参数
        
        total_train_loss += loss.item()
    
    return total_train_loss

def test(net, loss_func):
    net.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            img, target = data

            img = img.cuda()
            target = target.cuda()
            
            output = net(img)

            loss = loss_func(output, target)
            total_test_loss += loss.item()

            accuracy = (output.argmax(1) == target).sum().item()    # 1表示横向取最大值
            total_accuracy += accuracy
    
    return total_test_loss, total_accuracy / test_data_len

def train_and_test(net, loss_func, optimizer):
    init_stats()
    start_time = time.time()
    for _ in range(epoch):  
        train_loss = train(net, loss_func, optimizer) # 开始[训练]
        test_loss, accuracy = test(net, loss_func) # 开始[验证]

        # 更新统计数据
        RES[net.__class__.__name__]["test_loss"].append(test_loss)
        RES[net.__class__.__name__]["accuracy"].append(accuracy)
        update_stats(time.time()-start_time, train_loss, test_loss, accuracy)

    # torch.save(net, f"save_model_path/net_{i+1}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.pth")



def main():

    print("===================Start===================")
    print(f"TrainData_len:  \t{train_data_len}")
    print(f"TestData_len:   \t{test_data_len}")
    print(f"Device:         \t{device}")
    print(f"Epoch:          \t{epoch}")
    print(f"Batch_size:     \t{batch_size}")
    print(f"Learning_rate:  \t{learning_rate}")

    print("===========================================")

    for model in models:

        net = model()
        loss_func, optimizer = init(net)
        print(f"\n-----------------{net.__class__.__name__}-----------------")
        print(f"Model:          \t{net.__class__.__name__}")
        print(f"Loss Function:  \t{loss_func.__class__.__name__}")
        print(f"Optimizer:      \t{optimizer.__class__.__name__}")
        
        train_and_test(net, loss_func, optimizer)


    print("\n===================End===================")

if __name__ == '__main__':
    main()
    plot(RES)