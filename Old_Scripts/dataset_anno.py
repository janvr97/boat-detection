import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class BoatDetectionDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.annotations = list(sorted(os.listdir(os.path.join(root, "annotations"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        annotation_path = os.path.join(self.root, "annotations", self.annotations[idx])

        img = Image.open(img_path).convert("RGB")

        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

        boxes = torch.tensor([obj['bbox'] for obj in annotation['annotations']], dtype=torch.float32)
        labels = torch.tensor([obj['category_id'] for obj in annotation['annotations']], dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)



def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)
