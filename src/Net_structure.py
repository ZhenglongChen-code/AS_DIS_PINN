# this file contain several NN structures we use.
import torch
from torch import nn
import torch.nn.functional as F


def check_devices():
    gpu_count = torch.cuda.device_count()
    devices = [torch.device(f'cuda:{i}') for i in range(gpu_count)]
    print('gpu num is: {}'.format(gpu_count))
    return devices


# fcn模型
class FNN(nn.Module):
    def __init__(self, layer_size, activation=nn.Tanh):
        super().__init__()
        self.layer = nn.ModuleList()
        self.activation = activation

        # 数组长n，则网络的全连接层有n-1层

        for i in range(1, len(layer_size) - 1):
            self.layer.append(nn.Linear(layer_size[i - 1], layer_size[i]))
            self.layer.append(self.activation())

        # 这是最后一层输出层
        self.layer.append(nn.Linear(layer_size[-2], layer_size[-1]))

    def forward(self, inputs):
        X = inputs
        for layer in self.layer:
            X = layer(X)

        return X


class sub_res_block(torch.nn.Module):
    def __init__(self, input_channels, output_channels, use_1x1conv=False, strides=1):  # strides=1 means add original x
        super().__init__()
        self.conv1 = torch.nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = torch.nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = torch.nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = torch.nn.BatchNorm2d(output_channels)
        self.bn2 = torch.nn.BatchNorm2d(output_channels)

    def forward(self, X):
        # Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.tanh(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        # return F.relu(Y)
        return F.tanh(Y)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(sub_res_block(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(sub_res_block(num_channels, num_channels))

    return blk  # 返回的是一个列表，；里面每个元素才是一层模块。所以后面传入Sequential的时候需要用*解压一下


class LinearRestLayer(nn.Module):
    def __init__(self, layer_size, activation=nn.Tanh):
        """Make sure that layer_size[0] == layer_size[-1]"""
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(1, len(layer_size)):
            self.layers.append(nn.Linear(layer_size[i - 1], layer_size[i]))
            self.layers.append(activation())

    def forward(self, inputs):
        layer1 = self.layers[0]  # 原来写法Y = input, 然后套循环会导致Y 和 input 指向同一个tensor,最后计算梯度报错
        Y = layer1(inputs)
        for layer in self.layers[1:]:
            Y = layer(Y)

        Y = Y + inputs
        return Y


class GAM_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4):
        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out

# b1 = nn.Sequential(nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1),
#                    nn.BatchNorm2d(20), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))  # 1x20x20 --> 20x10x10
# b2 = nn.Sequential(*resnet_block(20, 20, 2, first_block=True))
# b3 = nn.Sequential(*resnet_block(20, 20, 2, first_block=True))
# b4 = nn.Sequential(*resnet_block(20, 40, 2, first_block=False))
# # b5 = nn.Sequential(*resnet_block(40,80,2,first_block=False))
#
# input_size = 400  # 输入为单元渗透率或trans，后者更好
# # 实例化模型
# model = nn.Sequential(b1, b2, b3, b4,
#                       nn.Flatten(), nn.Linear(40 * 5 * 5, input_size))
# linear rest net
# model = nn.Sequential(FNN([5,10,10]), nn.Tanh(), LinearRestLayer([10,20,20,10]), nn.Linear(10,5))
# x = torch.rand((4,5))
# y = model(x)
# print(y)

class CNN(torch.nn.Module):
    def __init__(self, output_size):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(  # 100*3*20*20 -> 100*25*18*18: (100) 对每一个batch即time step (2) channel
            torch.nn.Conv2d(4, 25, kernel_size=(3,3), stride=(1,1), padding=1),
            torch.nn.BatchNorm2d(25),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)  # # ->100*25*9*9
        )

        self.conv2 = torch.nn.Sequential( # ->100*50*7*7
            torch.nn.Conv2d(25, 50, kernel_size=(3,3), stride=(1,1), padding=1),
            torch.nn.BatchNorm2d(50),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)  # ->100*50*3*3
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(50 * 5 * 5, output_size),
            # torch.nn.ReLU(),
            # torch.nn.Linear(600, output_size)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.size(0), -1)   # x: tensor (time steps (100), 50*3*3)
        x = self.fc(x) # x: tensor (time steps (100), output size)
        return x
