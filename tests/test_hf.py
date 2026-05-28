"""
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from PIL import Image

# Carga .env desde la raíz del proyecto (un nivel por encima de tests/)
load_dotenv(Path(__file__).parent.parent / ".env")

token = os.environ["HUGGING_FACE"]
client = InferenceClient(token=token)
image = client.text_to_image(
    prompt="a cat sitting on a table, photorealistic",
    model="stabilityai/stable-diffusion-xl-base-1.0",
)
image.save("test_image.png")
print("OK:", image.size)
"""
# test_hf.py
from huggingface_hub import InferenceClient
import os

token = os.environ.get("HUGGING_FACE")
print(f"Token: {token[:10]}...")

client = InferenceClient(token=token)
try:
    image = client.text_to_image(
        prompt="a cat",
        model="stabilityai/stable-diffusion-xl-base-1.0",
    )
    print("OK:", image.size)
except Exception as e:
    print(f"Error tipo: {type(e).__name__}")
    print(f"Error: {e}")