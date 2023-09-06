import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.utils import make_grid
from torchvision import datasets, transforms

tans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


def save_batch(img, path):
    img = make_grid(img, padding=1)
    img = img / 2 + 0.5
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    print(path)
    plt.savefig(path)
    return img

# 参数太少会炸
# class Generator(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         
#         self.fc = nn.Sequential(nn.Flatten(),
#                                 nn.Linear(1*28*28, 16 * 28 * 28))

#         self.layer1 = nn.Sequential(
#             nn.Conv2d(16, 32, 3, 1, 1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(True))
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(32, 64, 3, 1, 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True))
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(64, 16, 3, 1, 1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(True))
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(16, 1, 3, 1, 1),
#             nn.Sigmoid())

#     def forward(self, x):
#         x = self.fc(x)
#         x = x.view(x.shape[0], 16, 28, 28)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         return x

class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(100, 3136)
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 50, 3, 1, 1),
            nn.BatchNorm2d(50),
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(50, 25, 3, 1, 1),
            nn.BatchNorm2d(25),
            nn.ReLU(True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(25, 1, 2, 2),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor):
        x = self.fc(x)
        x = x.view(x.shape[0], 1, 56, 56)
        x = self.br(x)
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
            nn.Linear(32 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc(x)
        x = x.squeeze(1)
        return x

# class Discriminator(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 32, 5, padding=2),  # batch, 32, 28, 28
#             nn.LeakyReLU(0.2, True),
#             nn.AvgPool2d(2, stride=2),  # batch, 32, 14, 14
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(32, 64, 5, padding=2),  # batch, 64, 14, 14
#             nn.LeakyReLU(0.2, True),
#             nn.AvgPool2d(2, stride=2)  # batch, 64, 7, 7
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(64*7*7, 1024),
#             nn.LeakyReLU(0.2, True),
#             nn.Linear(1024, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x: torch.Tensor):
#         x = x.view(x.shape[0], 1, 28, 28)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         x = x.squeeze(1)
#         return x


def train():
    D_lr = 0.0003
    G_lr = 0.0003
    batch_size = 512
    loss_fn = nn.BCELoss()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    G = Generator().to(device)
    D = Discriminator().to(device)
    D_optimizer = Adam(D.parameters(), lr=D_lr)
    G_optimizer = Adam(G.parameters(), lr=G_lr)
    print(device)
    epochs = 100
    train_datasets = datasets.MNIST(
        '/mnt/data0/public_datasets/torch_build_in_datas', download=True, train=True, transform=tans)
    train_iter = DataLoader(
        train_datasets, batch_size=batch_size, shuffle=True)
    test_datasets = datasets.MNIST(
        '/mnt/data0/public_datasets/torch_build_in_datas', download=True, train=False, transform=tans)
    test_iter = DataLoader(test_datasets, batch_size=batch_size)
    total_size = len(train_iter.dataset)
    mean = 0
    std = 1
    writer = SummaryWriter(
        log_dir='/mnt/data0/xuekang/workspace/models/gan/runs')
    step = 0
    for epoch in range(epochs):
        print("====EPOCH {}=====".format(epoch + 1))
        for i, (img, _) in enumerate(train_iter):

            num_img = img.shape[0]
            img = img.to(device)
            real_label = torch.ones(num_img).to(device)
            fake_label = torch.zeros(num_img).to(device)

            z = torch.normal(mean=mean, std=std, size=(
                num_img, 100)).to(device)
            fake_img = G(z)

            real_out = D(img)
            d_loss_real = loss_fn(real_out, real_label)
            fake_out = D(fake_img)
            d_loss_fake = loss_fn(fake_out, fake_label)

            d_loss = d_loss_real+d_loss_fake
            D_optimizer.zero_grad()
            d_loss.backward()
            D_optimizer.step()

            z = torch.normal(mean=mean, std=std, size=(
                num_img, 100)).to(device)
            fake_img = G(z)
            out = D(fake_img)

            g_loss = loss_fn(out, real_label)
            G_optimizer.zero_grad()
            g_loss.backward()
            G_optimizer.step()
            if (i + 1) % 10 == 0:
                step += 1
                writer.add_scalar('d_loss', d_loss.item(),
                                  step)
                writer.add_scalar('g_loss', g_loss.item(),
                                  step)
                writer.add_scalar('real_scores', real_out.data.mean(),
                                  step)
                writer.add_scalar('fake_score', fake_out.data.mean(),
                                  step)
                print(f'{epoch+1} : {num_img * (i+1)} / {total_size} d_loss:{d_loss}, g_loss:{g_loss} \
real_scores:{real_out.data.mean()} fake_score:{fake_out.data.mean()}')
        noise = torch.normal(mean=mean, std=std, size=(64, 100)).to(device)
        img = save_batch(G(noise).cpu(
        ), "/mnt/data0/xuekang/workspace/models/gan/data/res{}.jpg".format(epoch + 1))
        writer.add_image('images', img, epoch)
        print("--img saved--")


if __name__ == '__main__':
    train()
    # noise = torch.randn(64, 1, 28, 28)
    # save_batch(nn.Sigmoid()(noise), "/mnt/data0/xuekang/workspace/models/gan/data/aa.jpg")
