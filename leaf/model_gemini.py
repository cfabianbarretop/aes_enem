import os
from google import genai
from openai import OpenAI
from PIL import Image
from pathlib import Path
import json
import base64

from argparse import ArgumentParser

# ==============================================
# CONFIG
# ==============================================
DATA_LEAF_PATH = "data/leaf_11"     # Original dataset path
DATA_RESULT_PATH = "result"         # Result data path
API_KEY_SECRET = "api_key.json"     # Result data path

# ==============================================
# MODELOS IA
# ==============================================
class ModelLLM():
  def __init__(self, data_dir, result_dir):
    self.data_dir = data_dir
    self.result_dir = result_dir
    self.result_model = []

  def getApiKey(self, name_api_key):
    with open(API_KEY_SECRET, "r", encoding="utf-8") as f:
      config = json.load(f)
    return config[name_api_key]

  def modelGemini(self, name_api_key, path_img):
    api_key = self.getApiKey(name_api_key)
    client = genai.Client(api_key=api_key)
    image = Image.open(path_img)
    prompt = Path("promp_leaf.txt").read_text(encoding="utf-8")

    response = client.models.generate_content(
        # model="gemini-2.5-pro",
        model="gemini-2.5-flash",
        contents=[
            image,
            prompt
        ]
    )

    # print(response.text)
    return response

  def modelGPT(self, name_api_key, path_img):
    api_key = self.getApiKey(name_api_key)
    client = OpenAI(api_key=api_key)
    prompt = Path("promp_leaf.txt").read_text(encoding="utf-8")
    with open(path_img, "rb") as f:
        image_b64 = base64.b64encode(
            f.read()
        ).decode("utf-8")

    response = client.responses.create(
        model="gpt-5-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt
                    },
                    {
                        "type": "input_image",
                        "image_url":
                            f"data:image/jpeg;base64,{image_b64}"
                    }
                ]
            }
        ]
    )

    print(response.output_text)
    return response

  def createFile(self, name_file):
    species_output = os.path.join(self.result_dir, name_file)
    species_output.mkdir(
        parents=True,
        exist_ok=True
    )

  def saveResult(self, name_file):
    name_file = f"{self.result_dir}/{name_file}.json"
    with open(name_file, "w", encoding="utf-8") as f:
        json.dump(
            self.result_model,
            f,
            indent=2,
            ensure_ascii=False
        )

  def train(self, name_api_key):
    cont_file = 0
    dataset_root = Path(self.data_dir)
    for species_dir in dataset_root.iterdir():
      if not species_dir.is_dir():
        continue
      if cont_file > 0:
        break
    #   print("Species:", species_dir.name)
      specie_dir = os.path.join(self.data_dir, species_dir.name)
    #   self.createFile(specie_dir)
      self.result_model = []
      for idx, image_path in enumerate(species_dir.iterdir()):
        if idx >= 2:
          break
        if image_path.suffix.lower() not in [".jpg",".jpeg",".png"]:
            continue
        # print("  ", image_path.name)
        img_dir = os.path.join(specie_dir, image_path.name)
        prediction = self.modelGemini(name_api_key, img_dir)
        self.result_model.append({
            "image_name": image_path.name,
            "species": species_dir.name,
            "model": name_api_key,
            "prompt_version": "v1",
            "prediction": prediction.text
        })
      self.saveResult(img_dir)
      cont_file += 1

# ==============================================
# Main
# ==============================================

if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("leaf")
  parser.add_argument("--api-key-name", type=str, default="GEMINI")
  args = parser.parse_args()

  # Parameters
  api_key_name = args.api_key_name

  # Obtiene el directorio donde está este archivo.py
  base_dir = os.path.dirname(os.path.abspath(__file__))
  # Une el directorio de base_dir con las carpetas "data" y "result"
  data_dir = os.path.join(base_dir, DATA_LEAF_PATH)
  result_dir = os.path.join(base_dir, DATA_RESULT_PATH)
  model = ModelLLM(data_dir= data_dir, result_dir=result_dir)
  model.train(name_api_key=api_key_name)