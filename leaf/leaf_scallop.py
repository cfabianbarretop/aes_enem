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
from datasets import load_dataset
from argparse import ArgumentParser
from tqdm import tqdm

import scallopy

# ==============================================
# CONFIG
# ==============================================
DATA_LEAF_PATH = "data/leaf_11"     # Original dataset path
DATA_RESULT_PATH = "result"         # Result data path
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
    
    if train == 'train':
      self.leaf_dataset = load_dataset("dayagd/leaf-dataset", cache_dir="tmp/leaf", trust_remote_code=True)['train']
    elif train == 'test':
      self.leaf_dataset = load_dataset("dayagd/leaf-dataset", cache_dir="tmp/leaf", trust_remote_code=True)['test']

  def __len__(self):
    return int(len(self.leaf_dataset))
  
  def __getitem__(self, idx):
    img = self.leaf_dataset[idx]['image']
    if transform is not None: img = transform(img)
    label = self.leaf_dataset[idx]['idx_species']
    margin = self.leaf_dataset[idx]['idx_margin']
    shape = self.leaf_dataset[idx]['idx_shape']
    texture = self.leaf_dataset[idx]['idx_texture']
    return (img, label, margin, shape, texture)
  
  @staticmethod
  def collate_fn(batch):
    img = torch.stack([torch.tensor(item[0]).float() for item in batch])
    label = torch.stack([torch.tensor(item[1]).long() for item in batch])
    margin = torch.stack([torch.tensor(item[2]).long() for item in batch])
    shape = torch.stack([torch.tensor(item[3]).long() for item in batch])
    texture = torch.stack([torch.tensor(item[4]).long() for item in batch])
    return ((img), (label, margin, shape, texture))

def leaf_loader(data_dir, batch_size_train, batch_size_test):
  train_loader = torch.utils.data.DataLoader(
    LeafDatset(data_dir, "train"),
    collate_fn = LeafDatset.collate_fn,
    batch_size=batch_size_train,
    shuffle=True
  )
  test_loader = torch.utils.data.DataLoader(
    LeafDatset(data_dir, "test"),
    collate_fn = LeafDatset.collate_fn,
    batch_size=batch_size_test,
    shuffle=False
  )
  return train_loader, test_loader

# ==============================================
# Modelo Neural
# ==============================================

class LeafClassifierNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*56*56,128),
            nn.ReLU()
        )

        self.c1_head = nn.Linear(128, 6)
        self.c2_head = nn.Linear(128, 5)
        self.c3_head = nn.Linear(128, 4)

    def forward(self, x):

        x = self.conv(x)
        h = self.features(x)

        margin = F.softmax(self.c1_head(h), dim=1)
        shape = F.softmax(self.c2_head(h), dim=1)
        texture = F.softmax(self.c3_head(h), dim=1)

        return margin, shape, texture 
    
# ==============================================
# Modelo Lógico
# ==============================================

class LeafClassifierLog(nn.Module):
   def __init__(self, provenance, k):
      super(LeafClassifierLog, self).__init__()
      # Neural network classification
      self.leaf_net = LeafClassifierNet()

      # Scallop Context
      self.scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
      self.scl_ctx.add_relation("margin", int, input_mapping=list(range(6)))
      self.scl_ctx.add_relation("shape", int, input_mapping=list(range(6)))
      self.scl_ctx.add_relation("texture", int, input_mapping=list(range(6)))
      self.scl_ctx.add_rule("species(0) :- margin(0), shape(3), texture(1)")
      self.scl_ctx.add_rule("species(1) :- margin(4), shape(0), texture(0)")
      self.scl_ctx.add_rule("species(2) :- margin(2), shape(4), texture(2)")
      self.scl_ctx.add_rule("species(3) :- margin(0), shape(1), texture(0)")
      self.scl_ctx.add_rule("species(4) :- margin(3), shape(4), texture(2)")
      self.scl_ctx.add_rule("species(5) :- margin(2), shape(4), texture(3)")
      self.scl_ctx.add_rule("species(6) :- margin(0), shape(4), texture(0)")
      self.scl_ctx.add_rule("species(7) :- margin(0), shape(0), texture(3)")
      self.scl_ctx.add_rule("species(8) :- margin(0), shape(2), texture(0)")
      self.scl_ctx.add_rule("species(9) :- margin(0), shape(2), texture(1)")
      self.scl_ctx.add_rule("species(10) :- margin(0), shape(0), texture(2)")

      self.specie = self.scl_ctx.forward_function("species", output_mapping=[(i,) for i in range(11)])
    
   def forward(self, x: Tuple[torch.Tensor]):
      img = x
      margin, shape, texture = self.leaf_net(img)
      return margin, self.specie(margin=margin, shape=shape, texture=texture)

# ==============================================
# Guardar resultados
# ==============================================

def save__metrics(file_path, file_name, metric):
    name_file = f"{file_path}/result_metric_{file_name}.csv"
    with open(name_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch","loss","acc"])
        for row in metric:
            writer.writerow([
                row["epoch"],
                row["loss"],
                row["acc"],
            ])

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
  def __init__(self, result_dir, train_loader, test_loader, learning_rate, loss, k, provenance):
     self.network = LeafClassifierLog(provenance, k).to(device)
     self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
     self.train_loader = train_loader
     self.test_loader = test_loader
     self.result_dir = result_dir
     self.result_metrics_train = []
     self.result_metrics_test = []
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
     for (img, data_des) in iter:
        img = img.to(device)
        (target, margin, shape, texture) = data_des
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
     self.result_metrics_train.append({
        "epoch": epoch,
        "loss": loss.item(),
        "acc": perc.item(),
      })

  def test(self, epoch):
     self.network.eval()
     num_items = len(self.test_loader.dataset)
     correct = 0
     test_loss = 0
     with torch.no_grad():
        iter = tqdm(self.test_loader, total=len(self.test_loader))
        for (img, data_des) in iter:
           img = img.to(device)
           (target, margin, shape, texture) = data_des
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
    save__metrics(self.result_dir, "train",self.result_metrics_train)

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
  # Une el directorio de base_dir con las carpetas "data" y "result"
  data_dir = os.path.join(base_dir, DATA_LEAF_PATH)
  result_dir = os.path.join(base_dir, DATA_RESULT_PATH)

  # Dataloaders
  train_loader, test_loader = leaf_loader(data_dir, batch_size_train, batch_size_test)
  print("Dataset mnist")
  print("Train -> ", len(train_loader))
  print("Test -> ", len(test_loader))

  # Create trainer and train
  trainer = Trainer(result_dir, train_loader, test_loader, learning_rate, loss_fn, k, provenance)
  trainer.train(n_epochs)