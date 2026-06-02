import os
import random
from typing import *
import csv
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, accuracy_score
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset, Dataset
from argparse import ArgumentParser
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

import scallopy

TOKENIZER_NAME = f"neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
TOLERANCE = 10
nome_extra = "essay_c1_logic_addmult"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device)
class MNISTSum2Dataset(torch.utils.data.Dataset):
  def __init__(
    self,
    root: str,
    split: str,
    download: bool = False,
  ):
    # Contains a MNIST dataset
    self.split_name = split
    if split == "train":
        self.essays = load_dataset("igorcs/C1-A", cache_dir="tmp/aes_enem", trust_remote_code=True)['train']
    elif split == "test":
        self.essays = load_dataset("igorcs/C1-A", cache_dir="tmp/aes_enem", trust_remote_code=True)['test']

  def __len__(self):
    return int(len(self.essays))  

  def __getitem__(self, idx):
    tokenized_text = tokenizer(
                self.essays[idx]["essay_text"],
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=512
            )
    if isinstance(self.essays[idx]['nota'], str):
        nota = eval(self.essays[idx]['nota'])
    else:
        nota = self.essays[idx]['nota']
    if isinstance(self.essays[idx]['syntax'], str):
        syntax = eval(self.essays[idx]['syntax'])
    else:
        syntax = self.essays[idx]['syntax']
    if isinstance(self.essays[idx]['mistakes'], str):
        mistake = eval(self.essays[idx]['mistakes'])
    else:
        mistake = self.essays[idx]['mistakes']
    return (tokenized_text, syntax, mistake, nota)

  @staticmethod
  def collate_fn(batch):
    input_ids = torch.stack([item[0]['input_ids'][0] for item in batch])
    token_type = torch.stack([item[0]['token_type_ids'][0] for item in batch])
    attention_mask = torch.stack([item[0]['attention_mask'][0] for item in batch])
    syntax = torch.stack([torch.tensor(item[1]).long() for item in batch])
    mistake = torch.stack([torch.tensor(item[2]).long() for item in batch])
    nota = torch.stack([torch.tensor(item[3]).long() for item in batch])
    return ((input_ids, token_type, attention_mask), (syntax, mistake, nota))


def mnist_sum_2_loader(data_dir, batch_size_train, batch_size_test):
  train_loader = torch.utils.data.DataLoader(
    MNISTSum2Dataset(
      data_dir,
      split="train",
      download=True,
    ),
    collate_fn=MNISTSum2Dataset.collate_fn,
    batch_size=batch_size_train,
    shuffle=False
  )

  test_loader = torch.utils.data.DataLoader(
    MNISTSum2Dataset(
      data_dir,
      split="test",
      download=True,
    ),
    collate_fn=MNISTSum2Dataset.collate_fn,
    batch_size=batch_size_test,
    shuffle=False
  )

  return train_loader, test_loader


class MNISTNet(nn.Module):
  def __init__(self):
    super(MNISTNet, self).__init__()
    self.sintaxe = AutoModelForSequenceClassification.from_pretrained(
                "neuralmind/bert-base-portuguese-cased",
                cache_dir="/tmp/aes_enem2",
                num_labels=5,
            )
    
    self.desvios = AutoModelForSequenceClassification.from_pretrained( 
                "neuralmind/bert-base-portuguese-cased",
                cache_dir="/tmp/aes_enem2",
                num_labels=4,
            )

  def forward(self, x):
    output1 = self.sintaxe(input_ids=x[0].to(device), token_type_ids=x[1].to(device), 
                        attention_mask=x[2].to(device))
    output2 = self.desvios(input_ids=x[0].to(device), token_type_ids=x[1].to(device), 
                        attention_mask=x[2].to(device))

    return (F.softmax(output1.logits, dim=1), F.softmax(output2.logits, dim=1))


class MNISTSum2Net(nn.Module):
  def __init__(self, provenance, k):
    super(MNISTSum2Net, self).__init__()

    # MNIST Digit Recognition Network
    self.mnist_net = MNISTNet()

    # Scallop Context
    self.scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
    self.scl_ctx.add_relation("digit_1", int, input_mapping=list(range(5)))
    self.scl_ctx.add_relation("digit_2", int, input_mapping=list(range(4)))
    #self.scl_ctx.add_relation("digit_2", int, input_mapping=list(range(10)))
    self.scl_ctx.add_rule("sum_2(0) :- digit_1(0)")
    self.scl_ctx.add_rule("sum_2(1) :- digit_1(1), digit_2(0)")
    #soma2
    self.scl_ctx.add_rule("sum_2(2) :- digit_1(1), digit_2(b), b>=1")
    self.scl_ctx.add_rule("sum_2(2) :- digit_1(a), digit_2(0), a>=2")
    #self.scl_ctx.add_rule("sum_2(2) :- digit_1(a), digit_2(0), a>=2")
    #soma3
    self.scl_ctx.add_rule("sum_2(3) :- digit_1(2), digit_2(b), b>=1")
    self.scl_ctx.add_rule("sum_2(3) :- digit_1(a), digit_2(1), a>=3")
    #self.scl_ctx.add_rule("sum_2(3) :- digit_1(a), digit_2(1), a>=2")
    #soma4
    self.scl_ctx.add_rule("sum_2(4) :- digit_1(3), digit_2(b), b>=2")
    self.scl_ctx.add_rule("sum_2(4) :- digit_1(a), digit_2(2), a>=4")
    #self.scl_ctx.add_rule("sum_2(4) :- digit_1(a), digit_2(2), a>=3")
    #soma5
    self.scl_ctx.add_rule("sum_2(5) :- digit_1(4), digit_2(3)")
    # The `sum_2` logical reasoning module
    self.sum_2 = self.scl_ctx.forward_function("sum_2", output_mapping=[(i,) for i in range(6)])

  def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
    texto = x
    # First recognize the two digits
    resposta_a, resposta_b = self.mnist_net(texto) # Tensor 64 x 10

    # Then execute the reasoning module; the result is a size 19 tensor
    return resposta_a, resposta_b, self.sum_2(digit_1=resposta_a, digit_2=resposta_b)#, digit_2=b_distrs) # Tensor 64 x 19


def bce_loss(output, ground_truth):
  (_, dim) = output.shape
  gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth])
  return F.binary_cross_entropy(output, gt)


def nll_loss(output, ground_truth):
  return F.nll_loss(output, ground_truth)

def shortcut(g1, g2, y, c1, c2, p):
  # print("G1 -> ", g1)
  # print("G2 -> ", g2)
  # print("C1 -> ", c1)
  # print("C2 -> ", c2)
  # print("Y -> ", y)
  # print("y -> ", p)
  pred_tuples = list(zip(c1, c2, p))
  gt_tuples   = list(zip(g1, g2, y))
  # print("Predicciones:", pred_tuples)
  # print("Etiquetas reales:", gt_tuples)
  cont = 0
  cont_gt = 0
  for i, (pred, gt) in enumerate(zip(pred_tuples, gt_tuples)):
    if pred != gt:
        if pred[2] == gt[2]:
          print(f"Error en índice {i}: pred={pred}, gt={gt}")
          cont += 1
    else:
      cont_gt += 1
  print("Total de valores errados:", cont)
  print("Total de valores verdaderos:", cont_gt)

class Trainer():
  def __init__(self, train_loader, test_loader, learning_rate, loss, k, provenance):
    self.network = MNISTSum2Net(provenance, k).to(device)
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
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    c1 = []
    c2 = []
    g1 = []
    g2 = []
    y  = []
    p  = []
    for (data, data_des) in iter:
      (syntax, mistake, target) = data_des
      self.optimizer.zero_grad()
      p_syntax, p_mistake, output = self.network(data)
      output = output.cpu()
      g1.extend(syntax.tolist())
      g2.extend(mistake.tolist())
      c1.extend(p_syntax.argmax(dim=1).tolist())
      c2.extend(p_mistake.argmax(dim=1).tolist())
      p.extend(output.argmax(dim=1).tolist())
      y.extend(target.tolist())
      loss = self.loss(output, target)
      loss.backward()
      self.optimizer.step()
      iter.set_description(f"[Train Epoch {epoch}] Loss: {loss.item():.4f}")
    shortcut(g1, g2, y, c1, c2, p)

  def test(self, epoch):
    self.network.eval()
    num_items = len(self.test_loader.dataset)
    test_loss = 0
    correct = 0
    c1 = []
    c2 = []
    g1 = []
    g2 = []
    y  = []
    p  = []
    with torch.no_grad():
      iter = tqdm(self.test_loader, total=len(self.test_loader))
      for (data, data_des) in iter:
        (syntax, mistake, target) = data_des
        p_syntax, p_mistake, output = self.network(data)
        output = output.cpu()
        g1.extend(syntax.tolist())
        g2.extend(mistake.tolist())
        c1.extend(p_syntax.argmax(dim=1).tolist())
        c2.extend(p_mistake.argmax(dim=1).tolist())
        p.extend(output.argmax(dim=1).tolist())
        y.extend(target.tolist())
        test_loss += self.loss(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        perc = 100. * correct / num_items
        iter.set_description(f"[Test Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy: {correct}/{num_items} ({perc:.2f}%)")
      shortcut(g1, g2, y, c1, c2, p)


  def train(self, n_epochs):
    self.test(0)
    for epoch in range(1, n_epochs + 1):
      print("-----------> EPOCH: ",epoch)
      self.train_epoch(epoch)
      self.test(epoch)

if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("mnist_sum_2")
  parser.add_argument("--n-epochs", type=int, default=20)
  parser.add_argument("--batch-size-train", type=int, default=1)
  parser.add_argument("--batch-size-test", type=int, default=64)
  parser.add_argument("--learning-rate", type=float, default=0.000001)
  parser.add_argument("--loss-fn", type=str, default="bce")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--provenance", type=str, default="diffaddmultprob")
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
  data_dir = os.path.join(base_dir, "data_enem")
  # print("PATH data -> ", data_dir)

  # Dataloaders
  train_loader, test_loaders = mnist_sum_2_loader(data_dir, batch_size_train, batch_size_test)
  # Create trainer and train
  #print(validation_loader.dataset.essays)
  trainer = Trainer(train_loader, test_loaders , learning_rate, loss_fn, k, provenance)
  trainer.train(n_epochs)


