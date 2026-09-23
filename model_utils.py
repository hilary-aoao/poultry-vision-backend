import torch
import torch.nn as nn
from torchvision.models import resnet18
import requests
import os

MODEL_URL = "https://huggingface.co/HilaryBuilds/poultry-disease-classifier/resolve/main/poultry_model.pth"
MODEL_PATH = "poultry_model.pth"

CLASS_NAMES = ["Healthy", "Coccidiosis", "Salmonella", "New Castle Disease"]

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Local model not found, downloading from Hugging Face...")
        response = requests.get(MODEL_URL, stream=True, timeout=(10, 60))
        response.raise_for_status()

        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

        print("Model downloaded successfully")
    else:
        print("Using local model file, skipping download")

def load_model():
    download_model()

    model = resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 4)
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()

    return model