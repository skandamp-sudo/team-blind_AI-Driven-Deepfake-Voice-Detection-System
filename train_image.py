import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

def train():

    device = "cpu"

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    # 👉 CHANGE THIS PATH IF USING FULL DATASET
    dataset = datasets.ImageFolder("data_image", transform=transform)

    print("Classes:", dataset.classes)
    print("Total images:", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=32,     # reduce to 16 if slow
        shuffle=True,
        num_workers=0      # 🔥 IMPORTANT: prevents Mac crash
    )

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # freeze backbone for speed
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

    EPOCHS = 5

    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        acc = (correct / total) * 100
        print(f"Epoch {epoch+1} | Loss: {total_loss:.2f} | Accuracy: {acc:.2f}%")

    torch.save(model.state_dict(), "model/image.pth")
    print("Image model trained successfully")


if __name__ == "__main__":
    train()