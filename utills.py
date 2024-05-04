import torch
import torchvision.transforms as transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import wandb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_lables = {}
folder_path = "dataset/sportsdata/train"

# List all subdirectories in the 'classes' folder
subfolders = [f.name for f in os.scandir(folder_path) if f.is_dir()]

# Assign indices to class labels
for index, label in enumerate(subfolders):
    class_lables[label] = index

class ImageDataset(Dataset):
    def __init__(self, image_paths, targets, transform=None):
        self.image_paths = image_paths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item])
        image = self.transform(image)
        targets = self.targets[item]
        return image, torch.tensor(targets, dtype=torch.float32)
def transform(image):
    #Normalize Image
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image)/255.0
    image = transforms.ToTensor()(image)
    return image

def prepare_data(data_dir):
    images_type = ["jpg", "jpeg", "png","bmp"]
    target_dummy = [0]*100
    image_paths = []
    targets = []
    for class_name in os.listdir(data_dir):
        for img in os.listdir(os.path.join(data_dir, class_name)):
            if img.split(".")[-1] in images_type:
                image_paths.append(os.path.join(data_dir, class_name, img))
                #convert to one-hot encoding
                target = target_dummy.copy()
                target[class_lables[class_name]] = 1
                targets.append(target)
    df = pd.DataFrame({"image_paths": image_paths, "targets": targets})
    return df

def train_model(model, train_loader, eval_loader, epochs,lr, device):
    model.to(device)
    wandb.init(
    project="DL-Ops-Viva-Assignment",

    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "architecture": "VIT",
    "dataset": "Sports Dataset",
    "epochs": epochs,
    }
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i,data in progress_bar:
            images, targets = data
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs.logits, targets)
            loss.backward()
            optimizer.step()
            progress_bar.set_description(f"Epoch: {epoch}, Loss: {loss.item()}")
        
        

        accuracy = eval_model(model, eval_loader, device)
        print(f"Epoch: {epoch}, Loss: {round(loss.item(),4)}, Accuracy: {round(accuracy,4)}")
        wandb.log({"acc":accuracy,"loss": loss.item()})
    wandb.finish()
    torch.save(model, "models/pytorch_model.pth")
    return model

def eval_model(model, eval_loader, device):
    model.eval()
    num_correct = 0
    num_examples = 0
    for data in eval_loader:    
        images, targets = data
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs.logits, 1)
        _, targets = torch.max(targets, 1)
        num_correct += (predictions == targets).sum()
        num_examples += predictions.size(0)
    accuracy = num_correct/num_examples
    return accuracy.item()

def prepare_model(num_classes):
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")
    model.classifier = torch.nn.Linear(in_features=model.classifier.in_features, out_features=num_classes)
    #freeze the model
    for param in model.parameters():
        param.requires_grad = False
    #unfreez last 2 layers
    for param in model.vit.encoder.layer[-2:].parameters():
        param.requires_grad = True
    return model