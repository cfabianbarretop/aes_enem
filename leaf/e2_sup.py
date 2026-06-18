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
DATA_LEAF_PATH = "data/leaf_11"         # Original dataset path
DATA_RESULT_PATH = "result"             # Result data path
FILE_RESUL_METRIC = "e2_result_metric"  # Name file result
IMG_SIZE = (224, 224)                   # Standar size for ResNet/CNN type network
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
      self.scl_ctx.add_rule("species(0) :- margin(0), shape(0), texture(1)")
      
      self.scl_ctx.add_rule("species(1) :- margin(4), shape(a), texture(b), a>=0, b>=0")
      self.scl_ctx.add_rule("species(1) :- margin(0), shape(0), texture(0)")
      
      self.scl_ctx.add_rule("species(2) :- margin(1), shape(a), texture(b), a>=0, b>=0")
      
      self.scl_ctx.add_rule("species(3) :- margin(0), shape(1), texture(b), b>=0")
      self.scl_ctx.add_rule("species(3) :- margin(5), shape(1), texture(b), b>=0")
      
      self.scl_ctx.add_rule("species(4) :- margin(3), shape(a), texture(b), a>=0, b>=0")

      self.scl_ctx.add_rule("species(5) :- margin(2), shape(a), texture(b), a>=0, b>=0")
      
      self.scl_ctx.add_rule("species(6) :- margin(0), shape(4), texture(b), b>=0")
      
      self.scl_ctx.add_rule("species(7) :- margin(0), shape(3), texture(b), b>=0")
      
      self.scl_ctx.add_rule("species(8) :- margin(0), shape(0), texture(2)")
      
      self.scl_ctx.add_rule("species(9) :- margin(0), shape(2), texture(b), b>=0")
      self.scl_ctx.add_rule("species(9) :- margin(5), shape(a), texture(b), a>=2, b>=0")

      self.scl_ctx.add_rule("species(10) :- margin(0), shape(0), texture(3)")
      self.scl_ctx.add_rule("species(10) :- margin(5), shape(0), texture(b), b>=0")

      self.specie = self.scl_ctx.forward_function("species", output_mapping=[(i,) for i in range(11)])
    
   def forward(self, x: Tuple[torch.Tensor]):
      img = x
      margin, shape, texture = self.leaf_net(img)
      return margin, shape, texture, self.specie(margin=margin, shape=shape, texture=texture)

# ==============================================
# Guardar resultados
# ==============================================

def save__metrics(file_path, file_name, metric):
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
# Calculo de error
# ==============================================

def bce_loss(output, ground_truth):
  (_, dim) = output.shape
  gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth])
  return F.binary_cross_entropy(output, gt)


def nll_loss(output, ground_truth):
  return F.nll_loss(output, ground_truth)

def shortcut(g1, g2, g3, y, c1, pc1, c2, pc2, c3, pc3, p):
  # print("G1 -> ", g1)
  # print("G2 -> ", g2)
  # print("C1 -> ", c1)
  # print("C2 -> ", c2)
  # print("Y -> ", y)
  # print("y -> ", p)
  pred_tuples = list(zip(c1, c2, c3, p))
  gt_tuples   = list(zip(g1, g2, g3, y))
  # print("Predicciones:", pred_tuples)
  # print("Etiquetas reales:", gt_tuples)
  cont = 0
  cont_gt = 0
  sum_ars = 0
  sum_gt = 0
  sum_model = 0
  cy = {
    0: 1,
    1: 21,
    2: 20,
    3: 8,
    4: 20,
    5: 20,
    6: 4,
    7: 4,
    8: 1,
    9: 24,
    10: 5
}
  for i, (pred, gt) in enumerate(zip(pred_tuples, gt_tuples)):
    if pred != gt:
        if pred[3] == gt[3]:
          peso = cy.get(pred[3], 0)
          sum_ars += math.log(1/peso)
          p_c1 = pc1[i]
          p_c2 = pc2[i]
          p_c3 = pc3[i]
          sum_model += (1-(p_c1*p_c2*p_c3))*math.log(1/peso)
          # print(f"Error en índice {i}: pred={pred}, gt={gt}")
          cont += 1
    else:
      peso = cy.get(pred[3], 0)
      sum_gt += math.log(1/peso)
      p_c1 = pc1[i]
      p_c2 = pc2[i]
      p_c3 = pc3[i]
      sum_model += (1-(p_c1*p_c2*p_c3))*math.log(1/peso)
      cont_gt += 1
    
  print("=======================> Total de valores errados <=======================")
  print(f"Total de valores errados: {cont}")
  print(f"Total de valores verdaderos: {cont_gt}")
  print(f"Total de valores acertados: {cont + cont_gt}")
  print("==========================================================================")
  return cont_gt, cont, cont / (cont + cont_gt), sum_ars / (sum_ars + sum_gt), sum_model, sum_model / (sum_ars + sum_gt)

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
     correct_concepts = 0
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
     num_items = len(self.train_loader.dataset)
     iter = tqdm(self.train_loader, total=len(self.train_loader))
     for (img, data_des) in iter:
        img = img.to(device)
        (target, margin, shape, texture) = data_des
        self.optimizer.zero_grad()
        p_margin, p_shape, p_texture, output = self.network(img)
        output = output.cpu()
        g1.extend(margin.tolist())
        g2.extend(shape.tolist())
        g3.extend(texture.tolist())
        t_pc1, t_c1 = p_margin.max(dim=1)
        t_pc2, t_c2 = p_shape.max(dim=1)
        t_pc3, t_c3 = p_texture.max(dim=1)
        _ , t_p = output.max(dim=1)
        c1.extend(t_c1.tolist())
        pc1.extend(t_pc1.tolist())
        c2.extend(t_c2.tolist())
        pc2.extend(t_pc2.tolist())
        c3.extend(t_c3.tolist())
        pc3.extend(t_pc3.tolist())
        p.extend(t_p.tolist())
        y.extend(target.tolist())
        loss = self.loss(output, target)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        perc = 100. * correct / num_items
        loss.backward()
        self.optimizer.step()
        iter.set_description(f"[Train Epoch {epoch}] Loss: {loss.item():.4f}, Accuracy: {correct}/{num_items} ({perc:.2f}%)")
     gt, rs, rsr, rsrw, prob_model, prob_mod_no = shortcut(g1, g2, g3, y, c1, pc1, c2, pc2, c3, pc3, p)
     correct_concepts = (
        sum(gt == pred for gt, pred in zip(g1, c1))
        +
        sum(gt == pred for gt, pred in zip(g2, c2))
        +
        sum(gt == pred for gt, pred in zip(g3, c3))
     )
     total_concepts = 3 * len(g1)
     print(f"{correct_concepts} / {total_concepts}")
     gacc = 100.0 * correct_concepts / total_concepts
     self.result_metrics_train.append({
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
     correct = 0
     test_loss = 0
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
     with torch.no_grad():
        iter = tqdm(self.test_loader, total=len(self.test_loader))
        for (img, data_des) in iter:
           img = img.to(device)
           (target, margin, shape, texture) = data_des
           p_margin, p_shape, p_texture, output = self.network(img)
           output = output.cpu()
           g1.extend(margin.tolist())
           g2.extend(shape.tolist())
           g3.extend(texture.tolist())
           t_pc1, t_c1 = p_margin.max(dim=1)
           t_pc2, t_c2 = p_shape.max(dim=1)
           t_pc3, t_c3 = p_texture.max(dim=1)
           _ , t_p = output.max(dim=1)
           c1.extend(t_c1.tolist())
           pc1.extend(t_pc1.tolist())
           c2.extend(t_c2.tolist())
           pc2.extend(t_pc2.tolist())
           c3.extend(t_c3.tolist())
           pc3.extend(t_pc3.tolist())
           p.extend(t_p.tolist())
           y.extend(target.tolist())
           test_loss += self.loss(output, target).item()
           pred = output.data.max(1, keepdim=True)[1]
           correct += pred.eq(target.data.view_as(pred)).sum()
           perc = 100. * correct / num_items
           iter.set_description(f"[Test Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy: {correct}/{num_items} ({perc:.2f}%)")
        gt, rs, rsr, rsrw, prob_model, prob_mod_no = shortcut(g1, g2, g3, y, c1, pc1, c2, pc2, c3, pc3, p)
        correct_concepts = (
            sum(gt == pred for gt, pred in zip(g1, c1))
            +
            sum(gt == pred for gt, pred in zip(g2, c2))
            +
            sum(gt == pred for gt, pred in zip(g3, c3))
        )
        total_concepts = 3 * len(g1)
        gacc = 100.0 * correct_concepts / total_concepts
        self.result_metrics_test.append({
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
    save__metrics(self.result_dir, "train",self.result_metrics_train)
    save__metrics(self.result_dir, "test",self.result_metrics_test)

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