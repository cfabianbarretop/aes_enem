import os
import random
from typing import *
import csv
import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser
from tqdm import tqdm

from graphs import  main_graph
from distribution import main_distribution

import scallopy

# ==============================================
# CONFIG
# ==============================================
DATA_PATH = "data"                      # Original dataset path
DATA_LEVEL_PATH = "levels"              # Original dataset levels path
DATA_RESULT_PATH = "result"             # Result data path
FILE_RESUL_METRIC = "result_metric"     # Name file result
device = "cuda" if torch.cuda.is_available() else "cpu"
cy = {0: 1,  1: 3,  2: 6,  3: 10,  4: 15,  5: 21,  6: 28,  7: 36,  8: 45,  9: 55,  10: 63,  11: 69,  12: 73,  13: 75,  14: 75,  15: 73,  16: 69,  17: 63,  18: 55,  19: 45,  20: 36,  21: 28,  22: 21,  23: 15,  24: 10,  25: 6,  26: 3,  27: 1}
print("Device: ", device)

# ==============================================
# Dataset
# ==============================================

fashion_img_transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    (0.2860,), (0.3530,)   # stats Fashion-MNIST
  )
])

class FashionSum3Dataset(torch.utils.data.Dataset):
  def __init__(
    self,
    root: str,
    train: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
  ):
    self.dataset = torchvision.datasets.FashionMNIST(
      root,
      train=train,
      transform=transform,
      target_transform=target_transform,
      download=download,
    )

    self.index_map = list(range(len(self.dataset)))
    random.shuffle(self.index_map)

  def __len__(self):
    return int(len(self.dataset) / 3)

  def __getitem__(self, idx):
    (img_a, digit_a) = self.dataset[self.index_map[idx * 3]]
    (img_b, digit_b) = self.dataset[self.index_map[idx * 3 + 1]]
    (img_c, digit_c) = self.dataset[self.index_map[idx * 3 + 2]]

    return (
      img_a, img_b, img_c,
      digit_a, digit_b, digit_c,
      digit_a + digit_b + digit_c
    )

  @staticmethod
  def collate_fn(batch):
    imgs_a = torch.stack([item[0] for item in batch])
    imgs_b = torch.stack([item[1] for item in batch])
    imgs_c = torch.stack([item[2] for item in batch])

    digits_a = torch.stack([torch.tensor(item[3]).long() for item in batch])
    digits_b = torch.stack([torch.tensor(item[4]).long() for item in batch])
    digits_c = torch.stack([torch.tensor(item[5]).long() for item in batch])

    sums = torch.stack([torch.tensor(item[6]).long() for item in batch])

    return (
      (imgs_a, imgs_b, imgs_c),
      (digits_a, digits_b, digits_c, sums)
    )
  
def fashion_sum_3_loader(train_file, data_dir, batch_size_train, batch_size_test):

  train_loader = torch.utils.data.DataLoader(
    FashionSum3Dataset(
      data_dir,
      train=True,
      download=True,
      transform=fashion_img_transform,
    ),
    collate_fn=FashionSum3Dataset.collate_fn,
    batch_size=batch_size_train,
    shuffle=True
  )

  test_loader = torch.utils.data.DataLoader(
    FashionSum3Dataset(
      data_dir,
      train=False,
      download=True,
      transform=fashion_img_transform,
    ),
    collate_fn=FashionSum3Dataset.collate_fn,
    batch_size=batch_size_test,
    shuffle=True
  )

  print(f"Train loader: {len(train_loader.dataset)} samples")
  print(f"Test loader: {len(test_loader.dataset)} samples")

  return train_loader, test_loader

# ==============================================
# Neural Model
# ==============================================
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
    x = F.dropout(x, p = 0.5, training=self.training)
    x = self.fc2(x)
    return F.softmax(x, dim=1)


# ==============================================
# Logic Model
# ==============================================
class MNISTSum3Net(nn.Module):
  def __init__(self, provenance, k):
    super(MNISTSum3Net, self).__init__()

    # MNIST Digit Recognition Network
    self.mnist_net_d1 = MNISTNet()
    self.mnist_net_d2 = MNISTNet()
    self.mnist_net_d3 = MNISTNet()

    # Scallop Context
    self.scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
    self.scl_ctx.add_relation("digit_1", int, input_mapping=list(range(10)))
    self.scl_ctx.add_relation("digit_2", int, input_mapping=list(range(10)))
    self.scl_ctx.add_relation("digit_3", int, input_mapping=list(range(10)))
    self.scl_ctx.add_rule("sum_3(a + b + c) :- digit_1(a), digit_2(b), digit_3(c)")

    # The `sum_3` logical reasoning module
    # La salida es un tensor de tamaño 64 x 28 (porque la suma de tres dígitos entre 0 y 9 puede dar valores de 0 a 27).
    self.sum_3 = self.scl_ctx.forward_function("sum_3", output_mapping=[(i,) for i in range(28)])

  def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
    (imgs_a, imgs_b, imgs_c) = x

    # First recognize the three digits
    distrs_a = self.mnist_net_d1(imgs_a) # Tensor 64 x 10
    distrs_b = self.mnist_net_d2(imgs_b) # Tensor 64 x 10
    distrs_c = self.mnist_net_d3(imgs_c) # Tensor 64 x 10

    # Then execute the reasoning module; the result is a size 28 tensor
    return distrs_a, distrs_b, distrs_c, self.sum_3(digit_1=distrs_a, digit_2=distrs_b, digit_3=distrs_c) # Tensor 64 x 28

# ==============================================
# Save predictions
# ==============================================
def save_predictions(file_path, file_name, data):

    os.makedirs(file_path, exist_ok=True)
    name_file = f"{file_path}/{file_name}.csv"

    with open(name_file, "w", newline="") as file:
        writer = csv.writer(file)

        # Header correcto
        writer.writerow([
            "g1", "g2", "g3", "y",
            "c1", "pc1",
            "c2", "pc2",
            "c3", "pc3",
            "p"
        ])

        for i in range(len(data["g1"])):
            writer.writerow([
                data["g1"][i],
                data["g2"][i],
                data["g3"][i],
                data["y"][i],
                data["c1"][i],
                data["pc1"][i],
                data["c2"][i],
                data["pc2"][i],
                data["c3"][i],
                data["pc3"][i],
                data["p"][i],
            ])

def save_metrics(file_path, file_name, metric):
    name_file = f"{file_path}/{FILE_RESUL_METRIC}_{file_name}.csv"
    with open(name_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch","loss","accY", "accC", "gt", "rs", "RSR", "RSRw", "prob_model", "prob_mod_no"])
        for row in metric:
            writer.writerow([
                row["epoch"],
                row["loss"],
                row["accY"],
                row["accC"],
                row["gt"],
                row["rs"],
                row["RSR"],
                row["RSRw"],
                row["prob_model"],
                row["prob_mod_no"]
            ])

# ==============================================
# Metrics
# ==============================================
def metrics(g1, g2, g3, y, c1, pc1, c2, pc2, c3, pc3, p):
  pred_tuples = list(zip(c1, c2, c3, p))
  gt_tuples   = list(zip(g1, g2, g3, y))

  cont = 0
  cont_gt = 0
  sum_ars = 0
  sum_gt = 0
  sum_model = 0

  for i, (pred, gt) in enumerate(zip(pred_tuples, gt_tuples)):
    # pred = (c1, c2, c3, sum_pred)
    # gt   = (g1, g2, g3, sum_gt)
    if pred != gt:
        # reasoning shortcut: correct sum but wrong concepts
        if pred[3] == gt[3]:
          #weight = cy.get(pred[3], 0)
          weight = cy.get(pred[3], 1e-8)

          sum_ars += math.log(1 / weight)

          p_c1 = pc1[i]
          p_c2 = pc2[i]
          p_c3 = pc3[i]

          sum_model += (1 - (p_c1 * p_c2 * p_c3)) * math.log(1 / weight)

          cont += 1
    else:
      #weight = cy.get(pred[3], 0)
      weight = cy.get(pred[3], 1e-8)

      sum_gt += math.log(1 / weight)

      p_c1 = pc1[i]
      p_c2 = pc2[i]
      p_c3 = pc3[i]

      sum_model += (1 - (p_c1 * p_c2 * p_c3)) * math.log(1 / weight)

      cont_gt += 1
  
  print(f"Total wrong values ​​(RS): {cont}") 
  print(f"Total correct values ​​(full match): {cont_gt}") 
  print(f"Total values ​​evaluated: {cont + cont_gt}")

  return cont_gt, cont, cont / (cont + cont_gt), sum_ars / (sum_ars + sum_gt), sum_model, sum_model / (sum_ars + sum_gt)

# ==============================================
# Loss functions
# ==============================================
def bce_loss(output, ground_truth):
  (_, dim) = output.shape
  gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth])
  return F.binary_cross_entropy(output, gt)


def nll_loss(output, ground_truth):
  eps = 1e-8
  return F.nll_loss(torch.log(output + eps), ground_truth)

def cal_loss(output, ground_truth, alpha=31):
    batch_size = output.shape[0]
    loss = torch.tensor(0.0, device=output.device)
    for b, i in enumerate(ground_truth):
        y = i.item()
        p = torch.log(output[b, y].clamp(min=1e-8))
        weight = cy.get(y, 1)
        w = torch.log(torch.tensor(1 + (alpha / weight), device=output.device))
        w = w / torch.log(torch.tensor(1 + alpha, device=output.device))
        w = w.detach()
        loss += -p * w 
    return loss / batch_size

def bce_cal_loss(output, ground_truth):
  loss_bce = bce_loss(output, ground_truth)
  loss_cal = cal_loss(output, ground_truth)
  return loss_bce + loss_cal

def cel_loss(output, ground_truth):
  batch_size = output.shape[0]
  loss = torch.tensor(0.0, device=output.device)
  for b, i in enumerate(ground_truth):
      y = i.item()
      p = output[b, y]
      loss += (-torch.log(p))
  return loss / batch_size

def cel_cal(output, ground_truth):
  cel = cel_loss(output, ground_truth)
  cal = cal_loss(output, ground_truth)
  return cel + cal


# ==============================================
# Training and Testing
# ==============================================
class Trainer():
  def __init__(self, result_dir, train_loader, test_loader, learning_rate, loss, k, provenance):
    self.network = MNISTSum3Net_Exp2(provenance, k).to(device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.result_dir = result_dir
    self.metrics_train = []
    self.metrics_test = []
    
    if loss == "nll":
      self.loss = nll_loss
    elif loss == "bce":
      self.loss = bce_loss
    elif loss == "cel":
      self.loss = cel_loss
    elif loss == "cal":
      self.loss = cal_loss
    elif loss == "bce_cal":
      self.loss = bce_cal_loss  
    elif loss == "cel_cal":
      self.loss = cel_cal
    else:
      raise Exception(f"Unknown loss function `{loss}`")

  def train_epoch(self, epoch):
    self.network.train()
    correct = 0
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    num_items = len(self.train_loader.dataset)
    c1 = []
    pc1 = []
    c2 = []
    pc2 = []
    c3 = []
    pc3 = []
    g1 = []
    g2 = []
    g3 = []
    y  = []
    p  = []
    pb  = []
    for (data, data_des) in iter:
      (imgs_a, imgs_b, imgs_c) = data
      imgs_a = imgs_a.to(device)
      imgs_b = imgs_b.to(device)
      imgs_c = imgs_c.to(device)
      data = (imgs_a, imgs_b, imgs_c)
      (digits_a, digits_b, digits_c, target) = data_des
      self.optimizer.zero_grad()
      distrs_a, distrs_b, distrs_c, output = self.network(data)
      output = output.cpu()
      g1.extend(digits_a.tolist())
      g2.extend(digits_b.tolist())
      g3.extend(digits_c.tolist())
      t_pc1, t_c1 = distrs_a.max(dim=1)
      t_pc2, t_c2 = distrs_b.max(dim=1)
      t_pc3, t_c3 = distrs_c.max(dim=1)
      t_pb , t_p = output.max(dim=1)
      c1.extend(t_c1.tolist())
      pc1.extend(t_pc1.tolist())
      c2.extend(t_c2.tolist())
      pc2.extend(t_pc2.tolist())
      c3.extend(t_c3.tolist())
      pc3.extend(t_pc3.tolist())
      p.extend(t_p.tolist())
      pb.extend(t_pb.tolist())
      y.extend(target.tolist())
      loss = self.loss(output, target)
      # loss = self.loss(t_pc1, t_pc2, output, target)
      pred = output.data.max(1, keepdim=True)[1]
      #correct += pred.eq(target.data.view_as(pred)).sum()
      correct += pred.eq(target.view_as(pred)).sum().item()
      perc = 100. * correct / num_items
      loss.backward()
      self.optimizer.step()
      iter.set_description(f"[Train Epoch {epoch}] Loss: {loss.item():.4f} Accuracy: {correct}/{num_items} ({perc:.2f}%)")
    gt, rs, rsr, rsrw, prob_model, prob_mod_no = metrics(g1, g2, g3, y, c1, pc1, c2, pc2, c3, pc3, p)
    
    # correct_concepts = (
    #   sum(gt == pred for gt, pred in zip(g1, c1))
    #   +
    #   sum(gt == pred for gt, pred in zip(g2, c2))
    #   +
    #   sum(gt == pred for gt, pred in zip(g3, c3))
    # )
    # total_concepts = 3 * len(g1)
    # accC = 100.0 * correct_concepts / total_concepts
    correct_triplets = sum(
        (a == b) and (c == d) and (e == f)
        for a, b, c, d, e, f in zip(g1, c1, g2, c2, g3, c3)
      )
    accC = 100.0 * correct_triplets / len(g1)
    
    self.metrics_train.append({
       "epoch": epoch,
       "loss": loss.item(),
       "accY": 100.0 * correct / num_items,
       "accC": accC,
       "gt": gt,
       "rs": rs,
       "RSR": rsr,
       "RSRw": rsrw,
       "prob_model": prob_model,
       "prob_mod_no": prob_mod_no
    })
    # save_predictions(self.result_dir, f"train_epoch_{epoch}", {
    #   "g1": g1,
    #   "g2": g2,
    #   "g3": g3,
    #   "y": y,
    #   "c1": c1,
    #   "pc1": pc1,
    #   "c2": c2,
    #   "pc2": pc2,
    #   "c3": c3,
    #   "pc3": pc3,
    #   "p": p
    # })

  def test(self, epoch):
    self.network.eval()
    num_items = len(self.test_loader.dataset)
    test_loss = 0
    correct = 0
    c1 = []
    pc1 = []
    c2 = []
    pc2 = []
    c3 = []
    pc3 = []
    g1 = []
    g2 = []
    g3 = []
    y  = []
    p  = []
    pb  = []
    with torch.no_grad():
      iter = tqdm(self.test_loader, total=len(self.test_loader))
      for (data, data_des) in iter:
        (imgs_a, imgs_b, imgs_c) = data
        imgs_a = imgs_a.to(device)
        imgs_b = imgs_b.to(device)
        imgs_c = imgs_c.to(device)
        data = (imgs_a, imgs_b, imgs_c)
        (digits_a, digits_b, digits_c, target) = data_des
        distrs_a, distrs_b, distrs_c, output = self.network(data)
        output = output.cpu()
        g1.extend(digits_a.tolist())
        g2.extend(digits_b.tolist())
        g3.extend(digits_c.tolist())
        t_pc1, t_c1 = distrs_a.max(dim=1)
        t_pc2, t_c2 = distrs_b.max(dim=1)
        t_pc3, t_c3 = distrs_c.max(dim=1)
        t_pb , t_p = output.max(dim=1)
        c1.extend(t_c1.tolist())
        pc1.extend(t_pc1.tolist())
        c2.extend(t_c2.tolist())
        pc2.extend(t_pc2.tolist())
        c3.extend(t_c3.tolist())
        pc3.extend(t_pc3.tolist())
        p.extend(t_p.tolist())
        pb.extend(t_pb.tolist())
        y.extend(target.tolist())
        test_loss += self.loss(output, target).item()
        # test_loss += self.loss(t_pc1, t_pc2, output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        #correct += pred.eq(target.data.view_as(pred)).sum()
        correct += pred.eq(target.view_as(pred)).sum().item()
        perc = 100. * correct / num_items
        iter.set_description(f"[Test Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy: {correct}/{num_items} ({perc:.2f}%)")
      gt, rs, rsr, rsrw, prob_model, prob_mod_no = metrics(g1, g2, g3, y, c1, pc1, c2, pc2, c3, pc3, p)
      
      # correct_concepts = (
      #   sum(gt == pred for gt, pred in zip(g1, c1))
      #   +
      #   sum(gt == pred for gt, pred in zip(g2, c2))
      #   +
      #   sum(gt == pred for gt, pred in zip(g3, c3))
      # )
      # total_concepts = 3 * len(g1)
      # accC = 100.0 * correct_concepts / total_concepts

      correct_triplets = sum(
        (a == b) and (c == d) and (e == f)
        for a, b, c, d, e, f in zip(g1, c1, g2, c2, g3, c3)
      )
      accC = 100.0 * correct_triplets / len(g1)

      self.metrics_test.append({
         "epoch": epoch,
         "loss": test_loss,
         "accY": 100.0 * correct / num_items,
         "accC": accC,
         "gt": gt,
         "rs": rs,
         "RSR": rsr,
         "RSRw": rsrw,
         "prob_model": prob_model,
         "prob_mod_no": prob_mod_no
      })
      # save_predictions(self.result_dir, f"test_epoch_{epoch}", {
      #   "g1": g1,
      #   "g2": g2,
      #   "g3": g3,
      #   "y": y,
      #   "c1": c1,
      #   "pc1": pc1,
      #   "c2": c2,
      #   "pc2": pc2,
      #   "c3": c3,
      #   "pc3": pc3,
      #   "p": p
      # })

  def train(self, n_epochs):
    self.test(0)
    for epoch in range(1, n_epochs + 1):
      print("-----------> EPOCH: ",epoch)
      self.train_epoch(epoch)
      self.test(epoch)
    save_metrics(self.result_dir, "train", self.metrics_train)
    save_metrics(self.result_dir, "test", self.metrics_test)


if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("mnist_sum_3")
  parser.add_argument("--n-epochs", type=int, default=20)
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

  # Data
  base_dir = os.path.dirname(os.path.abspath(__file__))
  data_dir = os.path.abspath(os.path.join(base_dir, "..", DATA_PATH))

  result_dir = os.path.join(base_dir, DATA_RESULT_PATH)

  train_file = f"{data_dir}/{DATA_LEVEL_PATH}/mnist_addition_level_VI.pt"
  print("PATH data -> ", data_dir)

  # Dataloaders
  train_loader, test_loader = fashion_sum_3_loader(train_file, data_dir, batch_size_train, batch_size_test)

  # Create trainer and train
  trainer = Trainer(result_dir, train_loader, test_loader, learning_rate, loss_fn, k, provenance)
  trainer.train(n_epochs)
  # main_graph("train")
  # main_distribution(train_loader, test_loader)
