import torch
import torch.nn as nn
import torch.nn.functional as F

class VGGbase(nn.Module):
    def __init__(self):
        super(VGGbase, self).__init__()
        # 经过数据增强后，照片大小为3 * 28 * 28
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # (input_w - kernel_size + 2p)/stride + 1
        self.max_pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 经过Maxpooling后,照片尺寸变为14 * 14
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            # 在VGGnet中每次经过下采样，channel数量增加一倍
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.max_pooling2 = nn.MaxPool2d(kernel_size=2, stride=2)

        #  经过Maxpooling后,照片尺寸变为 7 * 7
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv3_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.max_pooling3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        # 因为7*7不padding的话，大小为3*3(会损失边缘信息), padding后，输出大小为4*4

        # 4*4
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.conv4_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.max_pooling4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2*2
        # batch_size * 512 * 2 * 2    -->   batchsize * (512*4)
        self.fc = nn.Linear(512 * 4, 10)

    def forward(self, x):
        batchszie = x.size(0)
        out = self.conv1(x)
        out = self.max_pooling1(out)

        out = self.conv2_1(out)
        out = self.conv2_2(out)
        out = self.max_pooling2(out)

        out = self.conv3_1(out)
        out = self.conv3_2(out)
        out = self.max_pooling3(out)

        out = self.conv4_1(out)
        out = self.conv4_2(out)
        out = self.max_pooling4(out)

        # (batchsize,c,h, w) --->  (batchsize, c*h*w)
        out = out.view(batchszie, -1)

        out = self.fc(out)     # (batchsize, 10)
        out = F.log_softmax(out, dim=1)

        return out

def VGGNet():
    return VGGbase()