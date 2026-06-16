from google import genai
from PIL import Image
from pathlib import Path

client = genai.Client(api_key="")

image = Image.open("/home/usuario/Documents/ENIAC - Artigo/lab/leaf/data/leaf_11/Citrus limon/0010_0001.JPG")

prompt = Path("promp_leaf.txt").read_text(encoding="utf-8")

response = client.models.generate_content(
    # model="gemini-2.5-pro",
    model="gemini-2.5-flash",
    contents=[
        image,
        prompt
    ]
)

print(response.text)