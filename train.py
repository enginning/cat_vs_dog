# coding=utf-8
# SY1806718
# eng
import numpy as np
import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from BasicModule import BasicModule
'''
import torch.nn.functional as F
'''


# 配置参数
random_state = 1
torch.manual_seed(random_state)  # 设置随机数种子，确保结果可重复
torch.cuda.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)
np.random.seed(random_state)
# random.seed(random_state)

epochs = 100      # 训练次数
batch_size = 100  # 批处理大小
num_workers = 0  # 多线程的数目
ModelSavePath = 'cat_vs_dog.model'

# 对加载的图像作归一化处理， 并裁剪为[224x224x3]大小的图像
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 数据的批处理，尺寸大小为batch_size,
# 在训练集中，shuffle 必须设置为True, 表示次序是随机的
train_dataset = datasets.ImageFolder(root='cats_and_dogs_small/train/', transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

test_dataset = datasets.ImageFolder(root='cats_and_dogs_small/test/', transform=data_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


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


# cnn = AlexNet()
cnn = torch.load(ModelSavePath)
print(cnn)

# 定义loss和optimizer
cirterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.0005, momentum=0.9)
# optimizer = optim.Adam(cnn.parameters(), lr=0.001, weight_decay=1e-4)

# 开始训练
cnn.train()
file = open('cat_vs_dog.txt', 'w+')

for epoch in range(epochs):
    # 模型训练
    start = time.time()
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    for i, data in enumerate(train_loader, 0):
        inputs, train_labels = data
        inputs, labels = Variable(inputs), Variable(train_labels)
        optimizer.zero_grad()
        outputs = cnn(inputs)
        _, train_predicted = torch.max(outputs.data, 1)
        train_correct += (train_predicted == labels.data).sum()
        loss = cirterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_total += train_labels.size(0)
        # torch.save(cnn, ModelSavePath)
        print("train {:d}-{:4d} epoch loss: {:.5f}  acc: {:7.3f} ".format(epoch + 1, i+1, running_loss / train_total,
                                                                          100 * train_correct / train_total))
        file.write("train {:d}-{:4d} epoch loss: {:.5f}  acc: {:7.3f}\n".
                   format(epoch + 1, i + 1, running_loss / train_total, 100 * train_correct / train_total))
        file.flush()
    torch.save(cnn, ModelSavePath)
    end = time.time()
    print("per epoch time:" + str(end - start) + "s")

    # 模型测试
    correct = 0
    test_loss = 0.0
    test_total = 0
    cnn.eval()
    for data in test_loader:
        images, labels = data
        images, labels = Variable(images), Variable(labels)
        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        loss = cirterion(outputs, labels)
        test_loss += loss.item()
        test_total += labels.size(0)
        correct += (predicted == labels.data).sum()
        print('test  %d epoch loss: %.5f  acc: %.3f ' % (epoch + 1, test_loss / test_total, 100 * correct / test_total))
        file.write("test {:d} epoch loss: {:.5f}  acc: {:7.3f}\n".
                   format(epoch + 1, test_loss / test_total, 100 * correct / test_total))

file.close()
