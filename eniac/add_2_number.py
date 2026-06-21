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

from graphs import main_graph
from distribution import main_distribution

import scallopy

# ==============================================
# CONFIG
# ==============================================
DATA_LEAF_PATH = "data"                 # Original dataset path
DATA_LEVEL_PATH = "levels"              # Original dataset levels path
DATA_RESULT_PATH = "result"             # Result data path
FILE_RESUL_METRIC = "result_metric"     # Name file result
device = "cuda" if torch.cuda.is_available() else "cpu"
cy = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 9, 11: 8, 12: 7, 13: 6, 14: 5, 15: 4, 16: 3, 17: 2, 18: 1}
print("Device: ", device)

# ==============================================
# Dataset
# ==============================================
mnist_img_transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    (0.1307,), (0.3081,)
  )
])

class MNISTSum2LevelDataset(torch.utils.data.Dataset):
    def __init__(self, file_name):

        self.data = torch.load(
            file_name,
            weights_only=False
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]

        return (
            sample["x1"],
            sample["x2"],
            sample["z1"],
            sample["z2"],
            sample["y"]
        )

    @staticmethod
    def collate_fn(batch):

        a_imgs = torch.stack(
            [item[0] for item in batch]
        )

        b_imgs = torch.stack(
            [item[1] for item in batch]
        )

        a_digits = torch.tensor(
            [item[2] for item in batch]
        ).long()

        b_digits = torch.tensor(
            [item[3] for item in batch]
        ).long()

        digits = torch.tensor(
            [item[4] for item in batch]
        ).long()

        return (
            (a_imgs, b_imgs),
            (a_digits, b_digits, digits)
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
    # Contains a MNIST dataset
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
    # Get two data points
    (a_img, a_digit) = self.mnist_dataset[self.index_map[idx * 2]]
    (b_img, b_digit) = self.mnist_dataset[self.index_map[idx * 2 + 1]]

    # Each data has two images and the GT is the sum of two digits
    return (a_img, b_img, a_digit, b_digit, a_digit + b_digit)

  @staticmethod
  def collate_fn(batch):
    a_imgs = torch.stack([item[0] for item in batch])
    b_imgs = torch.stack([item[1] for item in batch])
    a_digits = torch.stack([torch.tensor(item[2]).long() for item in batch])
    b_digits = torch.stack([torch.tensor(item[3]).long() for item in batch])
    digits = torch.stack([torch.tensor(item[4]).long() for item in batch])
    return ((a_imgs, b_imgs), (a_digits, b_digits, digits))


def mnist_sum_2_loader(train_file, data_dir, batch_size_train, batch_size_test):
  train_loader = torch.utils.data.DataLoader(
        MNISTSum2LevelDataset(train_file),
        batch_size=batch_size_train,
        shuffle=True,
        collate_fn=MNISTSum2LevelDataset.collate_fn
    )
  # train_loader = torch.utils.data.DataLoader(
  #   MNISTSum2Dataset(
  #     data_dir,
  #     train=True,
  #     download=True,
  #     transform=mnist_img_transform,
  #   ),
  #   collate_fn=MNISTSum2Dataset.collate_fn,
  #   batch_size=batch_size_train,
  #   shuffle=True
  # )

  test_loader = torch.utils.data.DataLoader(
    MNISTSum2Dataset(
      data_dir,
      train=False,
      download=True,
      transform=mnist_img_transform,
    ),
    collate_fn=MNISTSum2Dataset.collate_fn,
    batch_size=batch_size_test,
    shuffle=True
  )

  return train_loader, test_loader

# ==============================================
# Modelo Neural
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
# Modelo Lógico
# ==============================================
class MNISTSum2Net(nn.Module):
  def __init__(self, provenance, k):
    super(MNISTSum2Net, self).__init__()

    # MNIST Digit Recognition Network
    self.mnist_net = MNISTNet()

    # Scallop Context
    self.scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
    self.scl_ctx.add_relation("digit_1", int, input_mapping=list(range(10)))
    self.scl_ctx.add_relation("digit_2", int, input_mapping=list(range(10)))
    self.scl_ctx.add_rule("sum_2(a + b) :- digit_1(a), digit_2(b)")

    # The `sum_2` logical reasoning module
    # La salida es un tensor de tamaño 64 x 19 (porque la suma de dos dígitos entre 0 y 9 puede dar valores de 0 a 18).
    self.sum_2 = self.scl_ctx.forward_function("sum_2", output_mapping=[(i,) for i in range(19)])

  def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
    (a_imgs, b_imgs) = x

    # First recognize the two digits
    a_distrs = self.mnist_net(a_imgs) # Tensor 64 x 10
    b_distrs = self.mnist_net(b_imgs) # Tensor 64 x 10

    # Then execute the reasoning module; the result is a size 19 tensor
    return a_distrs, b_distrs, self.sum_2(digit_1=a_distrs, digit_2=b_distrs) # Tensor 64 x 19 

# ==============================================
# Guardar resultados
# ==============================================
def save_metrics(file_path, file_name, metric):
    name_file = f"{file_path}/{FILE_RESUL_METRIC}_{file_name}.csv"
    with open(name_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch","loss","acc", "GAcc", "gt", "rs", "RSR", "RSRw", "prob_model", "prob_mod_no"])
        for row in metric:
            writer.writerow([
                row["epoch"],
                row["loss"],
                row["acc"],
                row["GAcc"],
                row["gt"],
                row["rs"],
                row["RSR"],
                row["RSRw"],
                row["prob_model"],
                row["prob_mod_no"]
            ])

# ==============================================
# Metricas
# ==============================================
def metrics(g1, g2, y, c1, pc1, c2, pc2, p):
  pred_tuples = list(zip(c1, c2, p))
  gt_tuples   = list(zip(g1, g2, y))
  cont = 0
  cont_gt = 0
  sum_ars = 0
  sum_gt = 0
  sum_model = 0
  for i, (pred, gt) in enumerate(zip(pred_tuples, gt_tuples)):
    if pred != gt:
        if pred[2] == gt[2]:
          peso = cy.get(pred[2], 0)
          sum_ars += math.log(1/peso)
          p_c1 = pc1[i]
          p_c2 = pc2[i]
          sum_model += (1-(p_c1*p_c2))*math.log(1/peso)
          # print(f"Error en índice {i}: pred={pred}, gt={gt}")
          cont += 1
    else:
      peso = cy.get(pred[2], 0)
      sum_gt += math.log(1/peso)
      p_c1 = pc1[i]
      p_c2 = pc2[i]
      sum_model += (1-(p_c1*p_c2))*math.log(1/peso)
      cont_gt += 1
    
  print(f"Total de valores errados: {cont}")
  print(f"Total de valores verdaderos: {cont_gt}")
  print(f"Total de valores acertados: {cont + cont_gt}")
  return cont_gt, cont, cont / (cont + cont_gt), sum_ars / (sum_ars + sum_gt), sum_model, sum_model / (sum_ars + sum_gt)

# ==============================================
# Calculo de error
# ==============================================
def bce_loss(output, ground_truth):
  (_, dim) = output.shape
  gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth])
  return F.binary_cross_entropy(output, gt)


def nll_loss(output, ground_truth):
  return F.nll_loss(output, ground_truth)

def dpm_loss(p_c1, p_c2, output, ground_truth):
  loss = bce_loss(output, ground_truth)
  loss_dpm = torch.tensor(0.0, device=output.device)
  sum_weight = torch.tensor(0.0, device=output.device)
  sum_dpm = torch.tensor(0.0, device=output.device)
  cy_tensor = torch.tensor([cy[i] for i in range(len(cy))], dtype=output.dtype, device=output.device)
  for b, i in enumerate(ground_truth):
    y = i.item()
    weight = -torch.log(cy_tensor[y])
    prob = torch.tensor((1 - (p_c1[b] * p_c2[b])), device=output.device)
    sum_weight += weight
    sum_dpm += (prob * weight)
  loss_dpm = sum_dpm / sum_weight  
  return loss + loss_dpm

def cal_loss(output, ground_truth, alpha=31):
  batch_size = output.shape[0]
  loss = torch.tensor(0.0, device=output.device)
  for b, i in enumerate(ground_truth):
      y = i.item()
      p = torch.log(output[b, y])
      weight = cy.get(y, 0)
      w = torch.tensor(math.log(1 + (alpha / weight)), device=output.device)
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

def dpm_cel_loss(p_c1, p_c2, output, ground_truth):
  loss = cel_loss(output, ground_truth)
  loss_dpm = torch.tensor(0.0, device=output.device)
  sum_weight = torch.tensor(0.0, device=output.device)
  sum_dpm = torch.tensor(0.0, device=output.device)
  cy_tensor = torch.tensor([cy[i] for i in range(len(cy))], dtype=output.dtype, device=output.device)
  for b, i in enumerate(ground_truth):
    y = i.item()
    weight = -torch.log(cy_tensor[y])
    prob = torch.tensor((1 - (p_c1[b] * p_c2[b])), device=output.device)
    sum_weight += weight
    sum_dpm += (prob * weight)
  loss_dpm = sum_dpm / sum_weight  
  return loss + loss_dpm

# ==============================================
# Entrenamiento y Test
# ==============================================
class Trainer():
  def __init__(self, result_dir, train_loader, test_loader, learning_rate, loss, k, provenance):
    self.network = MNISTSum2Net(provenance, k).to(device)
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
    elif loss == "dpm":
      self.loss = dpm_loss
    elif loss == "dpm_cel":
      self.loss = dpm_cel_loss
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
    g1 = []
    g2 = []
    y  = []
    p  = []
    pb  = []
    for (data, data_des) in iter:
      (a_imgs, b_imgs) = data
      a_imgs = a_imgs.to(device)
      b_imgs = b_imgs.to(device)
      data = (a_imgs, b_imgs)
      (a_digits, b_digits, target) = data_des
      self.optimizer.zero_grad()
      a_distrs, b_distrs, output = self.network(data)
      output = output.cpu()
      g1.extend(a_digits.tolist())
      g2.extend(b_digits.tolist())
      t_pc1, t_c1 = a_distrs.max(dim=1)
      t_pc2, t_c2 = b_distrs.max(dim=1)
      t_pb , t_p = output.max(dim=1)
      c1.extend(t_c1.tolist())
      pc1.extend(t_pc1.tolist())
      c2.extend(t_c2.tolist())
      pc2.extend(t_pc2.tolist())
      p.extend(t_p.tolist())
      pb.extend(t_pb.tolist())
      y.extend(target.tolist())
      loss = self.loss(output, target)
      # loss = self.loss(t_pc1, t_pc2, output, target)
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
      perc = 100. * correct / num_items
      loss.backward()
      self.optimizer.step()
      iter.set_description(f"[Train Epoch {epoch}] Loss: {loss.item():.4f} Accuracy: {correct}/{num_items} ({perc:.2f}%)")
    gt, rs, rsr, rsrw, prob_model, prob_mod_no = metrics(g1, g2, y, c1, pc1, c2, pc2, p)
    correct_concepts = (
      sum(gt == pred for gt, pred in zip(g1, c1))
      +
      sum(gt == pred for gt, pred in zip(g2, c2))
    )
    total_concepts = 2 * len(g1)
    gacc = 100.0 * correct_concepts / total_concepts
    self.metrics_train.append({
       "epoch": epoch,
       "loss": loss.item(),
       "acc": perc.item(),
       "GAcc": gacc,
       "gt": gt,
       "rs": rs,
       "RSR": rsr,
       "RSRw": rsrw,
       "prob_model": prob_model,
       "prob_mod_no": prob_mod_no
    })

  def test(self, epoch):
    self.network.eval()
    num_items = len(self.test_loader.dataset)
    test_loss = 0
    correct = 0
    c1 = []
    pc1 = []
    c2 = []
    pc2 = []
    g1 = []
    g2 = []
    y  = []
    p  = []
    pb  = []
    with torch.no_grad():
      iter = tqdm(self.test_loader, total=len(self.test_loader))
      for (data, data_des) in iter:
        (a_imgs, b_imgs) = data
        a_imgs = a_imgs.to(device)
        b_imgs = b_imgs.to(device)
        data = (a_imgs, b_imgs)
        (a_digits, b_digits, target) = data_des
        a_distrs, b_distrs, output = self.network(data)
        output = output.cpu()
        g1.extend(a_digits.tolist())
        g2.extend(b_digits.tolist())
        t_pc1, t_c1 = a_distrs.max(dim=1)
        t_pc2, t_c2 = b_distrs.max(dim=1)
        t_pb , t_p = output.max(dim=1)
        c1.extend(t_c1.tolist())
        pc1.extend(t_pc1.tolist())
        c2.extend(t_c2.tolist())
        pc2.extend(t_pc2.tolist())
        p.extend(t_p.tolist())
        pb.extend(t_pb.tolist())
        y.extend(target.tolist())
        test_loss += self.loss(output, target).item()
        # test_loss += self.loss(t_pc1, t_pc2, output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        perc = 100. * correct / num_items
        iter.set_description(f"[Test Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy: {correct}/{num_items} ({perc:.2f}%)")
      gt, rs, rsr, rsrw, prob_model, prob_mod_no = metrics(g1, g2, y, c1, pc1, c2, pc2, p)
      correct_concepts = (
        sum(gt == pred for gt, pred in zip(g1, c1))
        +
        sum(gt == pred for gt, pred in zip(g2, c2))
      )
      total_concepts = 2 * len(g1)
      gacc = 100.0 * correct_concepts / total_concepts
      self.metrics_test.append({
         "epoch": epoch,
         "loss": test_loss,
         "acc": perc.item(),
         "GAcc": gacc,
         "gt": gt,
         "rs": rs,
         "RSR": rsr,
         "RSRw": rsrw,
         "prob_model": prob_model,
         "prob_mod_no": prob_mod_no
      })

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
  parser = ArgumentParser("mnist_sum_2")
  parser.add_argument("--n-epochs", type=int, default=20)
  parser.add_argument("--batch-size-train", type=int, default=64)
  parser.add_argument("--batch-size-test", type=int, default=64)
  parser.add_argument("--learning-rate", type=float, default=0.001)
  parser.add_argument("--loss-fn", type=str, default="cel")
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

  # Obtiene el directorio donde está este archivo.py
  base_dir = os.path.dirname(os.path.abspath(__file__))
  # Une el directorio de base_dir con la carpeta "data"
  data_dir = os.path.abspath(os.path.join(base_dir, "..", DATA_LEAF_PATH))
  result_dir = os.path.join(base_dir, DATA_RESULT_PATH)
  train_file = f"{data_dir}/{DATA_LEVEL_PATH}/mnist_addition_level_VI.pt"
  print("PATH data -> ", data_dir)

  # Dataloaders
  train_loader, test_loader = mnist_sum_2_loader(train_file, data_dir, batch_size_train, batch_size_test)
  # Create trainer and train
  trainer = Trainer(result_dir, train_loader, test_loader, learning_rate, loss_fn, k, provenance)
  trainer.train(n_epochs)
  main_graph("rs")
  # main_distribution(train_loader, test_loader)
  