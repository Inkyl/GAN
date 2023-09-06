import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
tans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
batch_size = 256

train_iter = DataLoader(datasets.MNIST(
    '/mnt/data0/public_datasets/torch_build_in_datas'), batch_size=batch_size, shuffle=True, download=True)

test_iter = DataLoader(datasets.MNIST(
    '/mnt/data0/public_datasets/torch_build_in_datas'), batch_size=batch_size, download=True)


class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(50),
            nn.ReLU(True)),
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(25),
            nn.ReLU(True))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 16, 3, 1, 1),
            nn.BatchNorm2d(),
            nn.ReLU(True))
        self.layer4 = nn.Sequential(
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.BatchNorm2d(),
            nn.ReLU(True))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),  # 16 * 28 * 28
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)  # 16 * 14 * 14
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),  # 32 * 14 * 14
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)  # 32 * 7 * 7
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc(x)
        return x


def train():
    loss = nn.BCELoss()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    G = Generator().to(device)
    D = Discriminator().to(device)
    learn_rate = 0.01
    epochs = 100
    for epoch in epochs:
        print("====EPOCH {}=====".format(epoch))
        for i, (img, _) in enumerate(train_iter):
            img = img.to(device)
            real_label = torch.ones(batch_size).to(device)
            fake_label = torch.ones(batch_size).to(device)

            


if __name__ == '__main__':
    train()
