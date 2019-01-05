# coding=utf-8
# SY1806718 胡继文
# by eng
import torch
import time
import csv
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from BasicModule import BasicModule
'''
import torch.nn.functional as F
from torchvision import models
'''

# 配置参数
random_state = 1
torch.manual_seed(random_state)  # 设置随机数种子，确保结果可重复
torch.cuda.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)
np.random.seed(random_state)
# random.seed(random_state)
# SUPER parameters
epochs = 1      # 训练次数
batch_size = 1  # 批处理大小
num_workers = 0  # 多线程的数目
ModelSavePath = 'cat_vs_dog.model'
csv_out_file_path = 'submission.csv'
Input_Test_Path = 'test/'


# 对加载的图像作归一化处理， 并裁剪为[224x224x3]大小的图像
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 数据的批处理，尺寸大小为batch_size,
# 在训练集中，shuffle 必须设置为False, 表示按顺序加载
test_dataset = datasets.ImageFolder(root=Input_Test_Path, transform=data_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# 搭建网络
class AlexNet(BasicModule):
    """
    code from torchvision/models/alexnet.py
    结构参考 <https://arxiv.org/abs/1404.5997>
    """
    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()

        self.model_name = 'alexnet'

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


# 加载模型
# cnn = AlexNet()
cnn = torch.load(ModelSavePath)
print(cnn)


# 定义loss和optimizer
optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.Adam(cnn.parameters(), lr=0.001, weight_decay=1e-4)

# 开始训练
cnn.train()
file = open('cat_vs_dog.txt', 'w+')

# added by eng for drawing the order of data_loader
list_images = []
for i, image in enumerate(test_dataset.imgs, 0):
    image_split1 = image[0].split('.')
    list_images.append(((image_split1[-2]).split('\\'))[-1])
    list_images[i] = int(eval(list_images[i]))
    # print(i, image_split[-2])


# 测试准备
running_loss = 0.0
train_correct = 0
train_total = 0
final_list = []


# 开始预测
cnn.eval()
scale = len(test_dataset)
factor = scale / 100
print("start predict".center(112, "-"))
start = time.perf_counter()
for i, data in enumerate(test_loader, 0):
    # Core
    inputs, train_labels = data
    inputs, labels = Variable(inputs), Variable(train_labels)
    optimizer.zero_grad()
    outputs = cnn(inputs)
    final_predict = torch.max(outputs, 1)[1].data.squeeze().cpu().numpy()
    if final_predict == 0:
        final = 'Cat'
    else:
        final = 'Dog'
    final_list.append(final)

    # print processing  bar
    scale_bar = int(scale / factor)
    a = '#' * int((i + 1) / factor)
    b = '.' * (scale_bar - int((i + 1) / factor))
    c = float(i / scale) * 100
    dur = time.perf_counter() - start
    print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(c, a, b, dur), end='')
print("\n")
print("end predict".center(112, "-"))
end = time.perf_counter()
print("Total predict time: {:5.2f} minutes {:5.2f} seconds".format((end - start) / 60, (end - start) % 60))
print("typing into csv...\nplease waite for a moment")


# csv write
csv_out_file = open(csv_out_file_path, 'w+', newline='')
csv_write = csv.writer(csv_out_file, dialect='excel')
csv_write.writerow(['id', 'label'])
csv_data = [(a, b) for a, b in zip(list_images, final_list)]
csv_data = sorted(csv_data, key=lambda x: x[0])

for i in range(len(csv_data)):
    print("{:4d}, {}".format(csv_data[i][0], csv_data[i][1]))
    csv_write.writerow([csv_data[i][0], csv_data[i][1]])
    csv_out_file.flush()
print("typed csv successfully !")
csv_out_file.close()
