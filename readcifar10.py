import pickle
import os
import cv2
import numpy as np
import glob  # 用于匹配文件

"""
下载cifar10数据集,并解析数据集，保存为图片
"""

# 使用cifar10官网提供的函数解析
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# 10个标签的名字
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

import glob

#train_list = glob.glob("./data/CIFAR10/data_batch_*")  # 返回：所有匹配到的文件路径列表list（一次性生成）
# print(train_list)
test_list = glob.glob("./data/CIFAR10/test_batch*")  # 返回：所有匹配到的文件路径列表list（一次性生成）
# 训练集图片
# save_path = "E:/pycharm_file/CV_project/image_classification_cifar10/data/CIFAR10/TRAIN"
# 测试机图片
save_path = "E:/pycharm_file/CV_project/image_classification_cifar10/data/CIFAR10/TEST"

# 遍历这个列表
#for l in train_list:
for l in test_list:
    print(l)
    l_dict = unpickle(l)
    # print(type(l_dict))
    # print(len(l_dict))
    # print(l_dict.keys())
    # print((l_dict[b'batch_label']))
    # print(len(l_dict[b'labels']))
    # print(l_dict[b'data'].shape)   # (10000,3072)     有5个训练集的文件，每个文件都是10000张3*32*32=3072的图片

    for im_idx, im_data in enumerate(l_dict[b'data']):  # 每个文件里的图片
        # print(im_idx)
        # print(im_data.shape)     # (3072,)

        im_label = l_dict[b'labels'][im_idx]
        im_name = l_dict[b'filenames'][im_idx]  # 图片的文件名
        print(im_label, im_name, im_data)  # 打印照片的 标签、名字、数据

        im_label_name = label_name[im_label]
        im_data = np.reshape(im_data, [3, 32, 32])
        im_data = np.transpose(im_data, (1, 2, 0))

        # cv2.imshow("im_data", cv2.resize(im_data, (200, 200)))
        # cv2.waitKey(0)

        if not os.path.exists("{}/{}".format(save_path,
                                             im_label_name)):       # 不存在文件夹
            os.mkdir("{}/{}".format(save_path, im_label_name))    # 则创建文件夹

        cv2.imwrite("{}/{}/{}".format(save_path,
                                      im_label_name,
                                      im_name.decode("utf-8")),
                    im_data)