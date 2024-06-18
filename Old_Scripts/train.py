import torch
import torch.optim as optim
#YOLO import statement, find replacement...
#from YOLO import train_loader, val_loader...


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = YOLO(num_classes=9).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        validate_model()


def validate_model():
    model.eval()
    with torch.no_grad():
        for images, targets in valid_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            print(f'Validation Loss: {loss.item():.4f}')


if __name__ == '__main__':
    train_model(num_epochs=25)
