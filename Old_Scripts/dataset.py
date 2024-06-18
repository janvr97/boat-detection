import os
import json
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

""" Dataset Class Extention"""

class BoatDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):

        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.images = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
        self.annotations = [os.path.join(annotation_dir, f"{os.path.splitext(img)[0]}.json")
                            for img in os.listdir(image_dir)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        annotation_path = self.annotations[idx]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Load annotations
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)
        bbox = torch.tensor(annotation['bbox'])

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'bbox': bbox, 'class': annotation['class']}
        return sample


""" Transforms """

valid_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

