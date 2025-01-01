import torch
from torch import nn


class LeNet5(nn.Module):
    """
    Input - 3x32x32
    Output - 10
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(6, 16, 5, 1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(16, 120, 5, 1),
            nn.ReLU(),
            
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

class LeNet5_BN(nn.Module):
    """
    Input - 3x32x32
    Output - 10
    """
    def __init__(self):
        super(LeNet5_BN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.BatchNorm2d(6),  # 批量归一化
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, 5, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 120, 5, 1),
            nn.BatchNorm2d(120),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

class LeNet5_LN(nn.Module):
    """
    Input - 3x32x32
    Output - 10
    """
    def __init__(self):
        super(LeNet5_LN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.LayerNorm([6, 28, 28]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(6, 16, 5, 1),
            nn.LayerNorm([16, 10, 10]),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 120, 5, 1),
            nn.LayerNorm([120, 1, 1]),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(120, 84),
            nn.LayerNorm(84),
            nn.ReLU(),

            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

class LeNet5_IN(nn.Module):
    """
    Input - 3x32x32
    Output - 10
    """
    def __init__(self):
        super(LeNet5_IN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.InstanceNorm2d(6),  # 实例归一化
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(6, 16, 5, 1),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 120, 5, 1),
            nn.ReLU(),
            
            nn.Flatten(),
            
            nn.Linear(120, 84),
            # nn.InstanceNorm1d(84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

class LeNet5_GN(nn.Module):
    """
    Input - 3x32x32
    Output - 10
    """
    def __init__(self):
        super(LeNet5_GN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.GroupNorm(3, 6),  # 组归一化
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, 5, 1),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 120, 5, 1),
            nn.GroupNorm(30, 120),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.GroupNorm(6, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    cnn = LeNet5()
    input = torch.ones((64, 3, 32, 32))
    output = cnn(input)
    print(output.shape)
