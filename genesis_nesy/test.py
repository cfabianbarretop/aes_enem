import os
import random
from typing import *

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser
from tqdm import tqdm

import scallopy

mnist_img_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    ]
)


class MNISTSum2Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):

        self.mnist_dataset = torchvision.datasets.MNIST(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        self.index_map = list(range(len(self.mnist_dataset)))
        random.shuffle(self.index_map)

    def __len__(self):
        return int(len(self.mnist_dataset) / 2)

    def __getitem__(self, idx):

        a_img, a_digit = self.mnist_dataset[self.index_map[idx * 2]]
        b_img, b_digit = self.mnist_dataset[self.index_map[idx * 2 + 1]]

        return (a_img, b_img, a_digit, b_digit, a_digit + b_digit)

    @staticmethod
    def collate_fn(batch):

        a_imgs = torch.stack([item[0] for item in batch])

        b_imgs = torch.stack([item[1] for item in batch])

        a_digits = torch.tensor([item[2] for item in batch]).long()

        b_digits = torch.tensor([item[3] for item in batch]).long()

        sums = torch.tensor([item[4] for item in batch]).long()

        return ((a_imgs, b_imgs), (a_digits, b_digits, sums))


def mnist_sum_2_loader(data_dir, batch_size_train, batch_size_test):

    train_loader = torch.utils.data.DataLoader(
        MNISTSum2Dataset(
            data_dir,
            train=True,
            download=True,
            transform=mnist_img_transform,
        ),
        collate_fn=MNISTSum2Dataset.collate_fn,
        batch_size=batch_size_train,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        MNISTSum2Dataset(
            data_dir,
            train=False,
            download=True,
            transform=mnist_img_transform,
        ),
        collate_fn=MNISTSum2Dataset.collate_fn,
        batch_size=batch_size_test,
        shuffle=True,
    )

    return train_loader, test_loader


class MNISTNet(nn.Module):

    def __init__(self):

        super(MNISTNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)

        self.fc1 = nn.Linear(1024, 1024)

        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):

        x = F.max_pool2d(self.conv1(x), 2)

        x = F.max_pool2d(self.conv2(x), 2)

        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))

        x = F.dropout(x, p=0.5, training=self.training)

        x = self.fc2(x)

        return F.softmax(x, dim=1)


class MNISTSum2Net(nn.Module):

    def __init__(self, provenance, k):

        super(MNISTSum2Net, self).__init__()

        self.mnist_net = MNISTNet()

        self.scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)

        self.scl_ctx.add_relation("digit_1", int, input_mapping=list(range(10)))

        self.scl_ctx.add_relation("digit_2", int, input_mapping=list(range(10)))

        self.scl_ctx.add_rule("sum_2(a + b) :- digit_1(a), digit_2(b)")

        self.sum_2 = self.scl_ctx.forward_function(
            "sum_2", output_mapping=[(i,) for i in range(19)]
        )

    def forward(self, x):

        a_imgs, b_imgs = x

        a_distrs = self.mnist_net(a_imgs)

        b_distrs = self.mnist_net(b_imgs)

        sum_distrs = self.sum_2(digit_1=a_distrs, digit_2=b_distrs)

        return (sum_distrs, a_distrs, b_distrs)


class Trainer:

    def __init__(self, train_loader, test_loader, learning_rate, loss, k, provenance):

        self.network = MNISTSum2Net(provenance, k)

        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        self.train_loader = train_loader
        self.test_loader = test_loader

        if loss == "nll":
            self.loss = self.nll_loss

        elif loss == "bce":
            self.loss = self.bce_loss

        else:
            raise Exception(f"Unknown loss function `{loss}`")

    def bce_loss(self, output, ground_truth):

        _, dim = output.shape

        gt = torch.stack(
            [
                torch.tensor([1.0 if i == t else 0.0 for i in range(dim)])
                for t in ground_truth
            ]
        )

        return F.binary_cross_entropy(output, gt)

    def nll_loss(self, output, ground_truth):

        return F.nll_loss(torch.log(output), ground_truth)

    def train_epoch(self, epoch):

        self.network.train()

        iterator = tqdm(self.train_loader, total=len(self.train_loader))

        for data, target in iterator:

            self.optimizer.zero_grad()

            # target contiene:
            # target[0] = digit 1 real
            # target[1] = digit 2 real
            # target[2] = suma real

            digit1_target = target[0]
            digit2_target = target[1]
            sum_target = target[2]

            # Forward
            sum_pred, digit1_pred, digit2_pred = self.network(data)

            # Loss de Scallop
            loss_sum = self.loss(sum_pred, sum_target)

            # Loss directo de la red
            loss_digit1 = self.nll_loss(digit1_pred, digit1_target)

            loss_digit2 = self.nll_loss(digit2_pred, digit2_target)

            # Loss total
            loss = loss_sum + loss_digit1 + loss_digit2

            # Un solo backward
            loss.backward()

            self.optimizer.step()

            iterator.set_description(
                f"[Train Epoch {epoch}] "
                f"Loss: {loss.item():.4f} "
                f"(sum={loss_sum.item():.4f}, "
                f"d1={loss_digit1.item():.4f}, "
                f"d2={loss_digit2.item():.4f})"
            )

    def test(self, epoch):

        self.network.eval()

        num_items = len(self.test_loader.dataset)

        correct_sum = 0
        correct_digit1 = 0
        correct_digit2 = 0

        with torch.no_grad():

            iterator = tqdm(self.test_loader, total=len(self.test_loader))

            for data, target in iterator:

                digit1_target = target[0]
                digit2_target = target[1]
                sum_target = target[2]

                sum_pred, digit1_pred, digit2_pred = self.network(data)

                pred_sum = sum_pred.argmax(dim=1)

                pred_digit1 = digit1_pred.argmax(dim=1)

                pred_digit2 = digit2_pred.argmax(dim=1)

                correct_sum += (pred_sum == sum_target).sum()

                correct_digit1 += (pred_digit1 == digit1_target).sum()

                correct_digit2 += (pred_digit2 == digit2_target).sum()

        print(f"""
[Test Epoch {epoch}]

Sum accuracy:
{100. * correct_sum / num_items:.2f} %

Digit 1 accuracy:
{100. * correct_digit1 / num_items:.2f} %

Digit 2 accuracy:
{100. * correct_digit2 / num_items:.2f} %
""")

    def train(self, n_epochs):

        self.test(0)

        for epoch in range(1, n_epochs + 1):

            self.train_epoch(epoch)

            self.test(epoch)


if __name__ == "__main__":

    parser = ArgumentParser("mnist_sum_2_multiloss")

    parser.add_argument("--n-epochs", type=int, default=10)

    parser.add_argument("--batch-size-train", type=int, default=64)

    parser.add_argument("--batch-size-test", type=int, default=64)

    parser.add_argument("--learning-rate", type=float, default=0.001)

    parser.add_argument("--loss-fn", type=str, default="nll")

    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--provenance", type=str, default="difftopkproofs")

    parser.add_argument("--top-k", type=int, default=3)

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    random.seed(args.seed)

    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))

    train_loader, test_loader = mnist_sum_2_loader(
        data_dir, args.batch_size_train, args.batch_size_test
    )

    trainer = Trainer(
        train_loader,
        test_loader,
        args.learning_rate,
        args.loss_fn,
        args.top_k,
        args.provenance,
    )

    trainer.train(args.n_epochs)
