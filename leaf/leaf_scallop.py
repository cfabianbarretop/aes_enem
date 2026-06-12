import os
import random
from typing import *
import csv
import math
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split

from argparse import ArgumentParser
from tqdm import tqdm

import scallopy

# ==============================================
# CONFIG
# ==============================================
DATA_LEAF_PATH = "data/leaf_11"          # Original dataset path
IMG_SIZE = (224, 224)               # Standar size for ResNet/CNN type network
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device)

# ==============================================
# Dataset
# ==============================================

# Transformation to imagens
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

class LeafDatset(torch.utils.data.Dataset):
  def __init__(self, root: str, train: str):
    dataset = datasets.ImageFolder(
        root=root,
        transform=transform
    )
    total = len(dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )
    if train == 'train':
      self.leaf_dataset = train_dataset
    elif train == 'test':
      self.leaf_dataset = test_dataset
    else:
      self.leaf_dataset = val_dataset

  def __len__(self):
    return int(len(self.leaf_dataset))
  
  def __getitem__(self, idx):
    return self.leaf_dataset[idx]

def leaf_loader(data_dir, batch_size_train, batch_size_test):
  train_loader = torch.utils.data.DataLoader(
    LeafDatset(data_dir, "train"),
    batch_size=batch_size_train,
    shuffle=True
  )
  test_loader = torch.utils.data.DataLoader(
    LeafDatset(data_dir, "test"),
    batch_size=batch_size_test,
    shuffle=False
  )
  return train_loader, test_loader

# ==============================================
# Modelo Neural
# ==============================================

class LeafClassifierNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*56*56,128),
            nn.ReLU(),
            nn.Linear(128,num_classes), # 11 species
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        x = self.conv(x)
        x = self.fc(x)
        return x 
    
# ==============================================
# Modelo Lógico
# ==============================================

class LeafClassifierLog(nn.Module):
   def __init__(self, provenance, k):
      super(LeafClassifierLog, self).__init__()
      # Neural network classification
      self.leaf_net = LeafClassifierNet(num_classes=11)

      # Scallop Context
      self.scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
      self.scl_ctx.add_relation("leaf", int, input_mapping=list(range(11)))
      self.scl_ctx.add_rule("species(a) :- leaf(a)")

      self.specie = self.scl_ctx.forward_function("species", output_mapping=[(i,) for i in range(11)])
    
   def forward(self, x: Tuple[torch.Tensor]):
      img = x
      img_distrs = self.leaf_net(img)
      return img_distrs, self.specie(leaf=img_distrs)

# ==============================================
# Calculo de error
# ==============================================

def bce_loss(output, ground_truth):
  (_, dim) = output.shape
  gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth])
  return F.binary_cross_entropy(output, gt)


def nll_loss(output, ground_truth):
  return F.nll_loss(output, ground_truth)

# ==============================================
# Entrenamiento y Test
# ==============================================
class Trainer():
  def __init__(self, train_loader, test_loader, learning_rate, loss, k, provenance):
     self.network = LeafClassifierLog(provenance, k).to(device)
     self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
     self.train_loader = train_loader
     self.test_loader = test_loader
     if loss == "nll":
         self.loss = nll_loss
     elif loss == "bce":
         self.loss = bce_loss
     else:
         raise Exception(f"Unknown loss function `{loss}`")
    
  def train_epoch(self, epoch):
     self.network.train()
     correct = 0
     num_items = len(self.train_loader.dataset)
     iter = tqdm(self.train_loader, total=len(self.train_loader))
     for (img, target) in iter:
        img = img.to(device)
        self.optimizer.zero_grad()
        img_distrs, output = self.network(img)
        output = output.cpu()
        loss = self.loss(output, target)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        perc = 100. * correct / num_items
        loss.backward()
        self.optimizer.step()
        iter.set_description(f"[Train Epoch {epoch}] Loss: {loss.item():.4f}, Accuracy: {correct}/{num_items} ({perc:.2f}%)")

  def test(self, epoch):
     self.network.eval()
     num_items = len(self.test_loader.dataset)
     correct = 0
     test_loss = 0
     with torch.no_grad():
        iter = tqdm(self.test_loader, total=len(self.test_loader))
        for (img, target) in iter:
           img = img.to(device)
           img_distrs, output = self.network(img)
           output = output.cpu()
           test_loss += self.loss(output, target).item()
           pred = output.data.max(1, keepdim=True)[1]
           correct += pred.eq(target.data.view_as(pred)).sum()
           perc = 100. * correct / num_items
           iter.set_description(f"[Test Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy: {correct}/{num_items} ({perc:.2f}%)")

  def train(self, n_epochs):
    self.test(0)
    for epoch in range(1, n_epochs + 1):
      print("-----------> EPOCH: ",epoch)
      self.train_epoch(epoch)
      self.test(epoch)

# ==============================================
# Main
# ==============================================

if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("leaf")
  parser.add_argument("--n-epochs", type=int, default=5)
  parser.add_argument("--batch-size-train", type=int, default=64)
  parser.add_argument("--batch-size-test", type=int, default=64)
  parser.add_argument("--learning-rate", type=float, default=0.001)
  parser.add_argument("--loss-fn", type=str, default="bce")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--provenance", type=str, default="difftopkproofs")
  parser.add_argument("--top-k", type=int, default=3)
  args = parser.parse_args()

  # Parameters
  n_epochs = args.n_epochs
  batch_size_train = args.batch_size_train
  batch_size_test = args.batch_size_test
  learning_rate = args.learning_rate
  loss_fn = args.loss_fn
  k = args.top_k
  provenance = args.provenance
  torch.manual_seed(args.seed)
  random.seed(args.seed)

  # Obtiene el directorio donde está este archivo.py
  base_dir = os.path.dirname(os.path.abspath(__file__))
  # Une el directorio de base_dir con la carpeta "data"
  data_dir = os.path.join(base_dir, DATA_LEAF_PATH)
  print("PATH data -> ", data_dir)

  # Dataloaders
  train_loader, test_loader = leaf_loader(data_dir, batch_size_train, batch_size_test)
  print("Dataset mnist")
  print("Train -> ", len(train_loader))
  print("Test -> ", len(test_loader))

  # Create trainer and train
  trainer = Trainer(train_loader, test_loader, learning_rate, loss_fn, k, provenance)
  trainer.train(n_epochs)