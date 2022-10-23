import torch
import torch.nn as nn
import torchvision
from vggnet import VGGNet
from resnet import resnet
from mobilenetv1 import mobilenetv1_small
from inceptionMolule import InceptionNetSmall
from load_cifar10 import train_data_loader, test_data_loader
import os

import tensorboardX  # 用于记录训练过程的情况
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epoch_num = 200
# 学习率
lr = 0.01
batch_size = 128

#VGG
# net = VGGNet().to(device=device)  # VGGNet
# net = resnet().to(device=device)   # 采用resnet
# net = mobilenetv1_small().to(device=device)   # 使用mobileNet
net = InceptionNetSmall().to(device=device)   # 使用inceptionNet

# loss 损失函数，分类问题使用交叉熵
loss_func = nn.CrossEntropyLoss()

# 优化器
optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # 采用Adam
# optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)   # 采用随机梯度下降

# 调整学习率
# step_size=5:每进行5个epoch调整一次学习率，  gamma=0.9,每次调整为上一次的0.9倍
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=5,
                                            gamma=0.9)

# 定义tensorboardX
if not os.path.exists("log"):
    os.mkdir("log")

writer = tensorboardX.SummaryWriter("log")
step_n = 0

# 定义训练的过程
for epoch in range(epoch_num):
    print("epoch is", epoch)  # 第几个epoch
    net.train()  # 训练的过程   BN, dropout 会选择相应的参数并更新，  但在推理的时候是不进行更新的

    for i, data in enumerate(train_data_loader):
        # print("step ", i)   # 一个epoch 第几个迭代次数(第几个batch)
        # print(len(data))
        inputs, labels = data  # 输入数据和标签
        inputs, labels = inputs.to(device=device), labels.to(device=device)

        outputs = net(inputs)  # 将输入放进网络，得到预测输出结果
        loss = loss_func(outputs, labels)  # 预测输出和真实的labels,计算损失

        optimizer.zero_grad()  # 反向传播前，将优化器的梯度置为0，否则梯度会累加
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        _, pred = torch.max(outputs.data, dim=1)  #
        correct = pred.eq(labels.data).cpu().sum()
        #print("epoch is ", epoch)
        # print("train lr is ", optimizer.state_dict()["param_groups"][0]["lr"])
        # print("train step", i, "loss is:", loss.item(),
        #       "mini-batch correct is:", 100.0 * correct / batch_size)

        writer.add_scalar("train loss", loss.item(), global_step=step_n)  # 记录训练集的数据
        writer.add_scalar("train correct",
                          100.0 * correct.item() / batch_size, global_step=step_n)

        im = torchvision.utils.make_grid(inputs)    # 将一个batch_size的图片拼接成一张大图片
        writer.add_image("train image", im, global_step=step_n)

        step_n += 1
    if not os.path.exists("models"):
        os.mkdir("models")

    torch.save(net.state_dict(), "models/{}.pth".format(epoch + 1))  # 保存模型
    scheduler.step()  # 每个epoch对学习率进行更新

    # print("lr is", optimizer.state_dict()["param_groups"][0]["lr"])

    # 每训练完一个epoch,测试一下
    sum_loss = 0
    sum_correct = 0
    for i, data in enumerate(test_data_loader):
        net.eval()  # 推理模式
        inputs, labels = data  # 输入数据和标签
        inputs, labels = inputs.to(device=device), labels.to(device=device)

        outputs = net(inputs)  # 将输入放进网络，得到预测输出结果
        loss = loss_func(outputs, labels)  # 预测输出和真实的labels,计算损失

        # optimizer.zero_grad()  # 反向传播前，将优化器的梯度置为0，否则梯度会累加
        # loss.backward()  # 反向传播
        # optimizer.step()  # 更新参数

        _, pred = torch.max(outputs.data, dim=1)  #
        correct = pred.eq(labels.data).cpu().sum()

        sum_loss += loss.item()
        sum_correct += correct.item()

        # writer.add_scalar("test loss", loss.item(), global_step=step_n)
        # writer.add_scalar("test correct", 100.0 * correct.item() / batch_size, global_step=step_n)

        im = torchvision.utils.make_grid(inputs)
        writer.add_image("test image", im, global_step=step_n)

    test_loss = sum_loss * 1.0 / len(test_data_loader)  # 测试机平均的loss
    test_correct = sum_correct * 100.0 / len(test_data_loader) / batch_size

    writer.add_scalar("test loss", test_loss, global_step=epoch + 1)
    writer.add_scalar("test correct", test_correct, global_step=epoch + 1)

    print("epoch", epoch + 1,
          "loss is", test_loss,
          "test_correct is:", test_correct)  # 打印出当前batch的整体loss    # i:一个epoch 第几个迭代次数(第几个batch)

writer.close()
