import torch
import torch.nn as nn
import torch.nn.functional as F


class mobilenet(nn.Module):

    # 深度可分离卷积（depthwise separable convolution） =  depthwise + pointwise
    def conv_dw(self, in_channel, out_channel, stride):     # block块
        return nn.Sequential(
            # Depthwise卷积, 分组卷积的特例(group = in_channel)
            nn.Conv2d(in_channels=in_channel,
                      out_channels=in_channel,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=False,
                      groups=in_channel),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            # Pointwise卷积(普通卷积的特例,1*1的卷积不改变图片的height和width,但可以改变通道来实现升维或降维)
            nn.Conv2d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def __init__(self):
        super(mobilenet, self).__init__()
        # 第一层都是标准普通的卷积
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv_dw2 = self.conv_dw(in_channel=32, out_channel=32, stride=1)
        self.conv_dw3 = self.conv_dw(in_channel=32, out_channel=64, stride=2)    # stride=2,进行下采样

        self.conv_dw4 = self.conv_dw(in_channel=64, out_channel=64, stride=1)
        self.conv_dw5 = self.conv_dw(in_channel=64, out_channel=128, stride=2)

        self.conv_dw6 = self.conv_dw(in_channel=128, out_channel=128, stride=1)
        self.conv_dw7 = self.conv_dw(in_channel=128, out_channel=256, stride=2)

        self.conv_dw8 = self.conv_dw(in_channel=256, out_channel=256, stride=1)
        self.conv_dw9 = self.conv_dw(in_channel=256, out_channel=512, stride=2)

        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_dw2(out)
        out = self.conv_dw3(out)
        out = self.conv_dw4(out)
        out = self.conv_dw5(out)
        out = self.conv_dw6(out)
        out = self.conv_dw7(out)
        out = self.conv_dw8(out)
        out = self.conv_dw9(out)

        out = F.avg_pool2d(out, 2)
        out = out.view(-1, 512)
        out = self.fc(out)

        return out


def mobilenetv1_small():
    return mobilenet()
