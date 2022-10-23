import torch
import torch.nn as nn
import torch.nn.functional as F


# block块   basic_block,  每个block串联两个卷积层
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):  # stride决定网络是否下采样
        super(ResBlock, self).__init__()
        # 主干串联分支
        self.layer = nn.Sequential(
            # 卷积核使用3*3大小的
            # 如果stride=1,卷积后输出大小不变,stride>1,下采样到stride倍
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            # 通常只在这一层下采样
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1,
                      bias=False),
            # 通常只在第一层下采样，故stride=1
            nn.BatchNorm2d(out_channel),
        )
        # 跳连分支(将输入 连接到 layer输出)
        # 为了保证输入和 layer层的输出能够连接(直接相加),需要保证通道一致(不一致通过1*1卷积实现通道一致)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:  # 如果通道不相等
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channel,
                          out_channels=out_channel,
                          kernel_size=1,                 # 通过1*1卷积通道数升维
                          stride=stride,
                          padding=1),  # 将输入大小下采样到layer输出一样大小
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out_1 = self.layer(x)  # 主干结构
        out_2 = self.shortcut(x)  # 分支结构
        out = out_1 + out_2  # 因为要相加，所以两者下采样的大小要相同。 两者相加
        out = F.relu(out)
        return out


#
class ResNet(nn.Module):
    # 生成layers,  相当于论文中的一个state(多个同颜色的block组成)
    def make_layer(self, block, out_channel, stride, num_block):
        layers_list = []
        for i in range(num_block):
            if i == 0:  # 第一个block进行下采样
                in_stride = stride
            else:
                in_stride = 1
            layers_list.append(block(self.in_channel, out_channel, in_stride))
            self.in_channel = out_channel  # 修改in_channel
            return nn.Sequential(*layers_list)  # *layers_list:获取list的每个元素(在这里是每一个block)

    def __init__(self, ResBlock):
        super(ResNet, self).__init__()

        self.in_channel = 32  # ResBlock输入通数

        # 第一个卷积层,通常第一层都是普通的卷积(和VGGNet一样的结构)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        # 第二层
        self.layer1 = self.make_layer(block=ResBlock, out_channel=64, stride=2, num_block=2)
        self.layer2 = self.make_layer(block=ResBlock, out_channel=128, stride=2, num_block=2)
        self.layer3 = self.make_layer(block=ResBlock, out_channel=256, stride=2, num_block=2)
        self.layer4 = self.make_layer(block=ResBlock, out_channel=512, stride=2, num_block=2)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out = self.conv1(x)          # torch.Size([128, 32, 28, 28])
        # print("1", out.shape)
        out = self.layer1(out)       # torch.Size([128, 64, 14, 14])
        # print("2", out.shape)
        out = self.layer2(out)       # torch.Size([128, 128, 7, 7])
        # print("3", out.shape)
        out = self.layer3(out)       # torch.Size([128, 256, 4, 4])
        # print("4", out.shape)
        out = self.layer4(out)       # torch.Size([128, 512, 2, 2])
        # print("before pool", out.shape)
        out = F.avg_pool2d(out, 2)   # 全局平均池化
        # print("after pool", out.shape)  #torch.Size([128, 512, 1, 1])
        out = out.view(out.size(0), -1)  # 将(batch_size,channel,h,w)  ===> (batch_size,channel*h*w)
        out = self.fc(out)
        # out = F.softmax(out, dim=1)  # 可以将概率分布结果经过softmax映射到[0,1]直接。分类问题采用交叉熵的损失函数,交叉熵的损失函数里面会计算softmax,故在这里不用softmax计算
        return out


# 定义函数返回ResNet网络
def resnet():
    print("ResNet")
    return ResNet(ResBlock)
