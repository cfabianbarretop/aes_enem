import os
import random
import torch
import torchvision

from typing import *
from argparse import ArgumentParser
from distribution import main_distribution

# ==============================================
# CONFIG
# ==============================================
DATA_MNIST_FASHION_PATH = "data"
# ==============================================
# Dataset
# ==============================================
mnist_img_transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()]
)


class MNISTFashionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        # Contains a MNIST dataset
        self.mnist_dataset = torchvision.datasets.FashionMNIST(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        self.index_map = list(range(len(self.mnist_dataset)))
        random.shuffle(self.index_map)

    def __len__(self):
        return int(len(self.mnist_dataset))

    def __getitem__(self, idx):
        # Get two data points
        img, digit = self.mnist_dataset[self.index_map[idx]]

        # Each data has two images and the GT is the sum of two digits
        return (img, digit)

    @staticmethod
    def collate_fn(batch):
        img = torch.stack([item[0] for item in batch])
        digit = torch.stack([torch.tensor(item[1]).long() for item in batch])
        return (img, digit)


def mnist_fashion_loader(data_dir, batch_size_train, batch_size_test):
    train_loader = torch.utils.data.DataLoader(
        MNISTFashionDataset(
            data_dir,
            train=True,
            download=True,
            transform=mnist_img_transform,
        ),
        collate_fn=MNISTFashionDataset.collate_fn,
        batch_size=batch_size_train,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        MNISTFashionDataset(
            data_dir,
            train=False,
            download=True,
            transform=mnist_img_transform,
        ),
        collate_fn=MNISTFashionDataset.collate_fn,
        batch_size=batch_size_test,
        shuffle=True,
    )

    return train_loader, test_loader


# ==============================================
# MAIN
# ==============================================
if __name__ == "__main__":
    # Argument parser
    parser = ArgumentParser("mnist_fashion")
    parser.add_argument("--batch-size-train", type=int, default=1)
    parser.add_argument("--batch-size-test", type=int, default=1)
    args = parser.parse_args()

    # Parameters
    batch_size_train = args.batch_size_train
    batch_size_test = args.batch_size_test

    # Obtiene el directorio donde está este archivo.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Une el directorio de base_dir con la carpeta "data"
    data_dir = os.path.abspath(os.path.join(base_dir, "..", DATA_MNIST_FASHION_PATH))
    print("PATH data -> ", data_dir)
    train_loader, test_loader = mnist_fashion_loader(data_dir, batch_size_train, batch_size_test)
    main_distribution(train_loader, test_loader)
