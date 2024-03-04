from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from PIL import Image
from skimage.transform import resize
import torch
import numpy as np

class CustomDataset(Dataset):  # Renamed from DataLoader to CustomDataset
    def __init__(self, root_dir, transform=None):
        """
        Declarining the variables for dataset reading
        root_dir: Root directory where the dataset is stored. Classes are represented as subfolders within this directory.
        transform: Transformation applied to the data, converting it to tensor object if true.
        classes: List of classes extracted from the subfolders in root_dir.
        class_to_idx: Dictionary mapping class names to their respective indices for encoding.
        samples: List containing tuples of data samples and their corresponding labels.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._make_dataset()  # Call _make_dataset to populate samples

    def __len__(self):
        """
        returning the length of the dataset
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get the index specified item from the dataset
        """
        img_path, target = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, target

    def normalize(self, tensor):
        """
        Normalizing the input
        """
        mean = torch.mean(tensor)
        std = torch.std(tensor)
        return (tensor - mean)/std


    def _make_dataset(self):
        samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                print(cls_dir)
                continue
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                img = Image.open(img_path)
                if self.transform:
                    img = self.transform(img)
                img = np.array(img)
                target = self.class_to_idx[cls_name]
                samples.append((img, target))
        return samples