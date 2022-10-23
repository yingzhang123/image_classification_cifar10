import torch
import torch.nn as nn
import torch.nn.functional as F


# inception模块:应用到需要加宽网络的宽度,进而提高模型的性能

def ConvBNRelu(in_channel, out_channel, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel,
                  out_channels=out_channel,
                  kernel_size=kernel_size,
                  stride=1,
                  padding=kernel_size // 2),
        nn.BatchNorm2d(out_channel),
        nn.ReLU()
    )


# inception的基本block块
class BaseInception(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel_list,
                 reduce_channel_list):
        super(BaseInception, self).__init__()

        self.branch1_conv = ConvBNRelu(in_channel,
                                       out_channel_list[0],
                                       1)

        self.branch2_conv1 = ConvBNRelu(in_channel,
                                        reduce_channel_list[0],
                                        1)
        self.branch2_conv2 = ConvBNRelu(reduce_channel_list[0],
                                        out_channel_list[1],
                                        3)

        self.branch3_conv1 = ConvBNRelu(in_channel,
                                        reduce_channel_list[1],
                                        1)
        self.branch3_conv2 = ConvBNRelu(reduce_channel_list[1],
                                        out_channel_list[2],
                                        5)

        self.branch4_pool = nn.MaxPool2d(kernel_size=3,
                                         stride=1,
                                         padding=1)
        self.branch4_conv = ConvBNRelu(in_channel,
                                       out_channel_list[3],
                                       3)

    def forward(self, x):
        out1 = self.branch1_conv(x)

        out2 = self.branch2_conv1(x)
        out2 = self.branch2_conv2(out2)

        out3 = self.branch3_conv1(x)
        out3 = self.branch3_conv2(out3)

        out4 = self.branch4_pool(x)
        out4 = self.branch4_conv(out4)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out



# class InceptionNet(nn.Module):
#     def __init__(self):
#         super(InceptionNet, self).__init__()
#         self.block1 = nn.Sequential(
#             nn.Conv2d(3, 64,
#                       kernel_size=7,
#                       stride=2,
#                       padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
#
#         self.block2 = nn.Sequential(
#             nn.Conv2d(64, 128,
#                       kernel_size=3,
#                       stride=2,
#                       padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU()
#         )
#
#         self.block3 = nn.Sequential(
#             BaseInception(in_channel=128,
#                              out_channel_list=[62, 64, 64, 64],
#                              reduce_channel_list=[16, 16]),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)    # 使用最大池化下采样(stride=2进行下采样)
#         )
#
#         self.block4 = nn.Sequential(
#             BaseInception(in_channel=256,
#                           out_channel_list=[96, 96, 96, 96],
#                           reduce_channel_list=[32, 32]),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         )
#
#         self.fc = nn.Linear(384, 10)
#
#     def forward(self, x):
#         out = self.block1(x)
#         out = self.block2(out)
#         out = self.block3(out)
#         out = self.block4(out)
#         out = torch.nn.functional.avg_pool2d(out, 2)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         return out
#
#
# def InceptionNetSmall():
#     return InceptionNet()

class InceptionNet(nn.Module):
    def __init__(self):
        super(InceptionNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64,
                      kernel_size=7,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.block3 = nn.Sequential(
            BaseInception(in_channel=128,
                          out_channel_list=[64, 64,
                                            64, 64],
                          reduce_channel_list=[16, 16]),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.block4 = nn.Sequential(
            BaseInception(in_channel=256,
                          out_channel_list=[96, 96,
                                            96, 96],
                          reduce_channel_list=[32, 32]),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.fc = nn.Linear(384, 10)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = torch.nn.functional.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def InceptionNetSmall():
    return InceptionNet()