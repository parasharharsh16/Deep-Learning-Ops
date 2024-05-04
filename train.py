import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from utills import *

mode = "test" #test, train, torchscript

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Defining the dataset directory
data_dir = "dataset/sportsdata"
train_dataset_df = prepare_data(data_dir+"/train")
eval_dataset_df = prepare_data(data_dir+"/valid")
test_dataset_df = prepare_data(data_dir+"/test")

if mode == "train":
    train_data = ImageDataset(
        image_paths=train_dataset_df["image_paths"].values,
        targets=train_dataset_df["targets"].values,
        transform=transform
    )
    eval_data = ImageDataset(
    image_paths=eval_dataset_df ["image_paths"].values,
    targets=eval_dataset_df ["targets"].values,
    transform=transform
    )

    batch_size=32
    epochs = 10
    learning_rate = 0.001
    num_classes = len(subfolders)

    train_loader =  DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    eval_loader =  DataLoader(dataset=eval_data, batch_size=1, shuffle=False)
    #load model
    model = prepare_model(num_classes)
    
    model = train_model(model, train_loader, eval_loader, epochs, lr=learning_rate, device=device)
    
    
if mode == "test":   
    #load model
    model = torch.load("models/pytorch_model.pth")
    model.eval()



if mode== "test" or mode == "train":
    test_data = ImageDataset(
    image_paths= test_dataset_df["image_paths"].values,
    targets=test_dataset_df["targets"].values,
    transform=transform
    )

    test_loader =  DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    accuracy = eval_model(model, test_loader, device)
    #Accuracy in percentage with last 2 decimal points
    accuracy = accuracy*100
    print(f"Accuracy on test dataset: {round(accuracy,2)}%")
    torch.save(test_data, 'Test_dataset.pt')

if mode == "torchscript":
    #load model
    model = torch.load("models/pytorch_model.pth")
    model.cpu().eval()
    sample_input = torch.rand(1, 3, 224, 224)

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    scripted_model = torch.jit.trace(model, [sample_input],strict=False)
    scripted_model.save("models/torch_script_model.pt")