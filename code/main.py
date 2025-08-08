# Imports
import torch
import torchvision
import requests
import zipfile
from pathlib import Path
import os

# Setup device-agnostic code
if torch.cuda.is_available():
  device = "cuda"
elif torch.mps.is_available():
  device = "mps"
else:
  device = "cpu"

# Setup num_workers
if device == "mps":
  num_workers = 0
else:
  num_workers = os.cpu_count()

# Setup path to a data folder
data_path = Path("image_classifier_project/data/")
image_path = data_path / "pizza_steak_sushi_20"

# If the image folder doesn't exist, download it and prepare it..
if image_path.is_dir():
  print(f"{image_path} directory already exists... skipping download")
else:
  print(f"{image_path} doesn't exist; creating one now...")
  image_path.mkdir(parents=True, exist_ok = True)

  # Download pizza, steak, and sushi data
  with open(data_path/"pizza_steak_sushi_20.zip", "wb") as f:
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/refs/heads/main/data/pizza_steak_sushi_20_percent.zip")
    print("Downloaing pizza, steak, and sushi data...")
    f.write(request.content)

  # Unzip pizza, steak, and sushi data
  with zipfile.ZipFile(data_path / "pizza_steak_sushi_20.zip", "r") as zip_ref:
    print("Unzipping piza, steak, and sushi data...")
    zip_ref.extractall(image_path)

# Setup train_transform and test_transform
train_transform = torchvision.transforms.Compose([
  torchvision.transforms.Resize((64,64)),
  torchvision.transforms.TrivialAugmentWide(),
  torchvision.transforms.ToTensor()
])
test_transform = torchvision.transforms.Compose([
  torchvision.transforms.Resize((64,64)),
  torchvision.transforms.ToTensor()
])

# Create dataloaders
from data_setup import create_dataloaders
train_dataloader, test_dataloader = create_dataloaders(train_dir=image_path / "train",
                                    test_dir = image_path / "test",
                                    train_transform = train_transform,
                                    test_transform = test_transform,
                                    batch_size = 32,
                                    num_workers = 0,
                                    device=device)

# Create two models
from model_setup import ImageClassificationModelV2
model_0 = ImageClassificationModelV2().to(device)
model_1 = ImageClassificationModelV2().to(device)

# Train them (one with and one without weight decay)
from train_engine import train_with_state_dict
model_0_results, model_0_save_dict = train_with_state_dict(model = model_0,
                                                          train_dataloader = train_dataloader,
                                                          test_dataloader = test_dataloader,
                                                          optim = torch.optim.Adam(model_0.parameters(), lr = 1e-4),
                                                          loss_fn = torch.nn.CrossEntropyLoss(),
                                                          epochs = 1000,
                                                          device = device)
model_1_results, model_1_save_dict = train_with_state_dict(model = model_1,
                                                          train_dataloader = train_dataloader,
                                                          test_dataloader = test_dataloader,
                                                          optim = torch.optim.Adam(model_1.parameters(), lr = 1e-4, weight_decay=1e-5),
                                                          loss_fn = torch.nn.CrossEntropyLoss(),
                                                          epochs = 1000,
                                                          device = device)

# Save the model
from model_saver_utils import save_model
save_model(model_save_dict=model_0_save_dict,
                      target_dir="image_classifier_project/models",
                      model_name="image_classifier_model_0.pth")
save_model(model_save_dict=model_1_save_dict,
                      target_dir="image_classifier_project/models",
                      model_name="image_clasifier_model_1.pth")

# Print out the best model
print(f"Maximum test accuracy for model_0:")
print(max(model_0_results["test_acc"]))
print(f"Maximum test accuracy for model_1:")
print(max(model_1_results["test_acc"]))