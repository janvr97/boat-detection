import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch.nn as nn

model = models.resnet18(weights=None)
num_classes = len(os.listdir(r"C:\Users\janva\PycharmProjects\boat\venv\boat-types-recognition"))
model.fc = nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load('boat_recognition_model.pth'))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = os.listdir(r"C:\Users\janva\PycharmProjects\boat\venv\boat-types-recognition")

def detect_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    class_name = class_names[predicted.item()]

    image = Image.open(image_path)
    plt.imshow(image)
    plt.title(f'Predicted: {class_name}')
    plt.axis('off')
    plt.show()

image_path = r"C:\Users\janva\PycharmProjects\boat\venv\images\example.jpg"
detect_image(image_path)
