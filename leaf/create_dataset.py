import os
from argparse import ArgumentParser
from datasets import Dataset
from huggingface_hub import login, upload_folder
from collections import defaultdict
import json
import random
from PIL import Image
from tqdm import tqdm
import numpy as np
import h5py

# ==============================================
# CONFIG
# ==============================================
DATA_LEAF_PATH = "data/leaf_11"                         # Original dataset path
DATA_RESULT_PATH = "../results"                         # Result data path
LEAF_LABEL = "leaf_labels_vlm.json"                     # Leaf labels data
DATASET_H5PY_TRAIN = "leaf_h5py_dataset_train.h5"       # H5py data
DATASET_H5PY_TEST = "leaf_h5py_dataset_test.h5"         # H5py data
DATASET_HUGGING_FACE = "leaf_hf_dataset"                # Hugging face data
DATASET_HUGGING_FACE_TRAIN = "train"                    # Hugging face data
DATASET_HUGGING_FACE_TEST = "test"                      # Hugging face data
HUGGING_FACE_HUB = "dayagd/leaf-dataset"                # Hub dataset
API_KEY_SECRET = "api_key.json"                         # Api key data path

# Concept's map
MARGIN_MAP = {
    "entire": 0,
    "indented": 1,
    "lobed": 2,
    "serrate": 3,
    "serrulate": 4,
    "undulate": 5
}

MARGIN_MAP_INV = {
    v: k for k, v in MARGIN_MAP.items()
}

SPECIES_MAP = {
    "Alstonia Scholaris": 0,
    "Citrus limon": 1,
    "Jatropha curcas": 2,
    "Mangifera indica": 3,
    "Ocimum basilicum": 4,
    "Platanus orientalis": 5,
    "Pongamia Pinnata": 6,
    "Psidium guajava": 7,
    "Punica granatum": 8,
    "Syzygium cumini": 9,
    "Terminalia Arjuna": 10
}

SPECIES_MAP_INV = {
    v: k for k, v in SPECIES_MAP.items()
}

SHAPE_MAP = {
    "elliptical": 0,
    "lanceolate": 1,
    "oblong": 2,
    "obovate": 3,
    "ovate": 4
}

SHAPE_MAP_INV = {
    v: k for k, v in SHAPE_MAP.items()
}

TEXTURE_MAP = {
    "glossy": 0,
    "leathery": 1,
    "smooth": 2,
    "rough": 3
}

TEXTURE_MAP_INV = {
    v: k for k, v in TEXTURE_MAP.items()
}

# ==============================================
# DATASET
# ==============================================
class DatasetLeaf():
   def __init__(self, data_dir, result_dir):
      self.data_dir = data_dir 
      self.result_dir = result_dir
      self.images = []
      self.species = []
      self.margin = []
      self.shape = []
      self.texture = []
      self.ids = []

   def clean(self):
      self.images = []
      self.species = []
      self.margin = []
      self.shape = []
      self.texture = []
      self.ids = []

   def getApiKey(self, name_api_key):
    with open(API_KEY_SECRET, "r", encoding="utf-8") as f:
      config = json.load(f)
    return config[name_api_key]

   def saveData(self, train: bool):
    dataset_orig = DATASET_H5PY_TEST
    dataset_des = os.path.join(DATASET_HUGGING_FACE, DATASET_HUGGING_FACE_TEST)
    if train: 
      dataset_orig = DATASET_H5PY_TRAIN
      dataset_des = os.path.join(DATASET_HUGGING_FACE, DATASET_HUGGING_FACE_TRAIN)
    dataset_dir = os.path.join(self.result_dir, dataset_orig)
    dataset_result_dir = os.path.join(self.result_dir, dataset_des)
    data = {
        "id": [],
        "image": [],
        "species": [],
        "margin": [],
        "shape": [],
        "texture": [],
        "idx_species": [],
        "idx_margin": [],
        "idx_shape": [],
        "idx_texture": []
    }
    f = h5py.File(dataset_dir, "r")
    ids = f["ids"].asstr()
    images = f["images"]
    species = f["species"]
    margin = f["margin"]
    shape = f["shape"]
    texture = f["texture"]
    for i in tqdm(range(len(ids)), desc= "Processing images"):
      img = Image.fromarray(images[i])
      data["id"].append(ids[i])
      data["image"].append(img)
      data["species"].append(SPECIES_MAP_INV[int(species[i])])
      data["margin"].append(MARGIN_MAP_INV[int(margin[i])])
      data["shape"].append(SHAPE_MAP_INV[int(shape[i])])
      data["texture"].append(TEXTURE_MAP_INV[int(texture[i])])
      data["idx_species"].append(int(species[i]))
      data["idx_margin"].append(int(margin[i]))
      data["idx_shape"].append(int(shape[i]))
      data["idx_texture"].append(int(texture[i]))
    print("Save....")
    dataset = Dataset.from_dict(data)
    dataset.save_to_disk(dataset_result_dir)

   def updateDataHuggingFace(self, name_api_key):
     token = self.getApiKey(name_api_key)
     dataset_result_dir = os.path.join(self.result_dir, DATASET_HUGGING_FACE)
     login(token=token)
     upload_folder(folder_path=dataset_result_dir, repo_id=HUGGING_FACE_HUB, repo_type="dataset")
           
   def h5py(self, train: bool):
      name_file = DATASET_H5PY_TEST
      if train: name_file = DATASET_H5PY_TRAIN
      string_dtype = h5py.string_dtype(encoding="utf-8")
      dataset_dir = os.path.join(self.result_dir, name_file)
      print("Save....")
      with h5py.File(dataset_dir, "w") as f:
        f.create_dataset(
            "images",
            data=self.images,
            compression="gzip"
        )
        f.create_dataset(
            "species",
            data=self.species
        )
        f.create_dataset(
            "margin",
            data=self.margin
        )
        f.create_dataset(
            "shape",
            data=self.shape
        )
        f.create_dataset(
            "texture",
            data=self.texture
        )
        f.create_dataset(
            "ids",
            data=self.ids,
            dtype=string_dtype
        )

   def conver(self):
      self.images = np.array(self.images, dtype=np.uint8)
      self.species = np.array(self.species, dtype=np.int32)
      self.margin = np.array(self.margin, dtype=np.int32)
      self.shape = np.array(self.shape, dtype=np.int32)
      self.texture = np.array(self.texture, dtype=np.int32)

   def splitData(self):
       train_annotations = []
       test_annotations = []
       train_ratio = 0.8
       file_dir = os.path.join(self.result_dir, LEAF_LABEL)
       with open(file_dir, "r", encoding="utf-8") as f:
         annotations = json.load(f)
       groups = defaultdict(list)
       for sample in annotations:
         groups[sample["species"]].append(sample)
       for species, samples in groups.items():
         random.shuffle(samples)
         split_idx = int(len(samples) * train_ratio)
         train_annotations.extend(samples[:split_idx])
         test_annotations.extend(samples[split_idx:])
       return train_annotations, test_annotations
         
   def create(self, annotations, train: bool):
       for sample in tqdm(annotations, desc="Loading images"):
        img_dir = os.path.join(self.data_dir, sample["image"])
        img = Image.open(img_dir).convert("RGB")
        img = np.array(img)
        self.images.append(img)
        self.species.append(
            SPECIES_MAP[sample["species"]]
        )
        self.margin.append(
            MARGIN_MAP[sample["margin"]]
        )
        self.shape.append(
            SHAPE_MAP[sample["shape"]]
        )
        self.texture.append(
            TEXTURE_MAP[sample["texture"]]
        )
        self.ids.append(sample["id"])
       self.conver()
       self.h5py(train)
       self.clean()    

# ==============================================
# Main
# ==============================================

if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("dataset")
  parser.add_argument("--api-key-name", type=str, default="HUGGING_FACE")
  args = parser.parse_args()

  # Parameters
  api_key_name = args.api_key_name

  # Obtiene el directorio donde está este archivo.py
  base_dir = os.path.dirname(os.path.abspath(__file__))
  # Une el directorio de base_dir con las carpetas "data" y "result"
  result_dir = os.path.join(base_dir, DATA_RESULT_PATH)
  dataset = DatasetLeaf(data_dir= base_dir, result_dir=result_dir)
  data_train, data_test = dataset.splitData()
  dataset.create(data_train, True)
  dataset.saveData(True)
  dataset.create(data_test, False)
  dataset.saveData(False)
  dataset.updateDataHuggingFace(api_key_name)