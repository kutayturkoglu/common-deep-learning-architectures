import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

def plot(train_data):
    labels_map = {
        0: "Buildings",
        1: "Forest",
        2: "Glacier",
        3: "Mountain",
        4: "Sea",
        5: "Street"
    }
    figure = plt.figure(figsize=(8, 8))
    cols, rows,i = 3, 2,1
    labels = []

    while i < cols*rows + 1:
        sample_idx = torch.randint(len(train_data), size=(1,)).item()
        img, label = train_data[sample_idx]
        if label not in labels:
            labels.append(label)
            figure.add_subplot(rows, cols, i)
            plt.title(labels_map[label])
            plt.axis("off")
            plt.imshow(img.squeeze(), cmap="gray")  # Convert tensor to NumPy array
            i+=1
        
    plt.show()

def find_mean_std(dataset):
    r_channel = []
    g_channel = []
    b_channel = []

    # Iterate through the dataset to collect pixel values
    for img, _ in dataset:
        r_channel.append(torch.mean(img[0])) # Red channel
        g_channel.append(torch.mean(img[1])) # Green channel
        b_channel.append(torch.mean(img[2])) # Blue channel

    # Calculate mean and standard deviation
    mean = torch.mean(torch.tensor(r_channel)), torch.mean(torch.tensor(g_channel)), torch.mean(torch.tensor(b_channel))
    std = torch.std(torch.tensor(r_channel)), torch.std(torch.tensor(g_channel)), torch.std(torch.tensor(b_channel))
    return mean, std

def transform(mean, std):
    # Define the transformation pipeline
    transformation = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=mean, std=std)  # Normalize images
    ])
    return transformation
