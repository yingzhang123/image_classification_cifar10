# 使用PyTorch自定义数据加载，加载Cifar10数据集
import glob

from torchvision import transforms  # 数据增强
from torch.utils.data import DataLoader, Dataset
import torch
from PIL import Image  # 读入图片
import numpy as np

label_name = ["airplane",
              "automobile",
              "bird",
              "cat",
              "deer",
              "dog",
              "frog",
              "horse",
              "ship",
              "truck"]
# 将label转化为字典
lable_dict = {}
for idx, name in enumerate(label_name):
    lable_dict[name] = idx  # 标签所对应的数字


#
def default_loader(path):
    return Image.open(path)


# torchvision.transforms是pytorch中的图像预处理包。一般用Compose把多个步骤整合到一起
train_transform = transforms.Compose([
    transforms.RandomResizedCrop((28, 28)),  # 将PIL图像裁剪成任意大小和纵横比
    transforms.RandomHorizontalFlip(),  # 水平反转   默认概率为0.5   以0.5的概率水平翻转给定的PIL图像
    transforms.RandomVerticalFlip(),  # 垂直方向   默认概率为0.5   以0.5的概率竖直翻转给定的PIL图像
    transforms.RandomRotation(90),  # 旋转
    transforms.RandomGrayscale(0.1),  # 将图像以一定的概率转换为灰度图像
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
    # transforms.ColorJitter 改变图像的属性：亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue)
    transforms.ToTensor()
    # convert a PIL image to tensor (H*W*C) in range [0,255] to a torch.Tensor(C*H*W) in the range [0.0,1.0]
])


class MyDataset(Dataset):
    def __init__(self, im_list, transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        imgs = []

        for im_item in im_list:  # im_list:一个类别的文件夹图片路径的列表
            "E:\pycharm_file\CV_project\image_classification_cifar10\data\CIFAR10\TRAIN\airplane\twinjet_s_000994.png"
            # print(im_item)
            im_label_name = im_item.split("\\")[-2]  # 照片的标签
            imgs.append([im_item, lable_dict[im_label_name]])

        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        im_path, im_label = self.imgs[index]  # 拿到图片的路径、标签
        im_data = self.loader(im_path)  # 读取图片数据

        # 数据增强
        if self.transform is not None:
            im_data = self.transform(im_data)

        return im_data, im_label

    def __len__(self):
        """返回样本总数"""
        return len(self.imgs)


# 拿到所有训练图片的的路径列表列表
im_train_list = glob.glob("E:\pycharm_file\CV_project\image_classification_cifar10\data\CIFAR10\TRAIN\*\*.png")
# 测试集图片的路径
im_test_list = glob.glob("E:\pycharm_file\CV_project\image_classification_cifar10\data\CIFAR10\TEST\*\*.png")

# 实例化Dataset
train_dataset = MyDataset(im_train_list, transform=train_transform)  # 训练集的数据需要数据增强
test_dataset = MyDataset(im_test_list, transform=transforms.ToTensor())  # 测试集不需要数据增强

train_data_loader = DataLoader(dataset=train_dataset,
                               batch_size=128,
                               shuffle=True,
                               num_workers=0)

test_data_loader = DataLoader(dataset=test_dataset,
                              batch_size=128,
                              shuffle=False,  # 测试集数据不需要shuffle
                              num_workers=0)

print("num_of_train", len(train_dataset))
print("num_of_test", len(test_dataset))
