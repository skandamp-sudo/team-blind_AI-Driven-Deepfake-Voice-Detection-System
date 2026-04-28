import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Subset

def train():

    device = "cpu"

    # --- TRANSFORMS ---
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    # --- LOAD FULL DATASET ---
    dataset = datasets.ImageFolder(
        "data_image_full/real_vs_fake/real-vs-fake/train",
        transform=transform
    )

    print("Classes:", dataset.classes)
    print("Full dataset size:", len(dataset))

    # --- LIMIT TO 10K (SMART MOVE) ---
    dataset = Subset(dataset, range(10000))

    print("Training on subset size:", len(dataset))

    # --- DATALOADER ---
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0   # 🔥 IMPORTANT for Mac
    )

    # --- MODEL ---
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)

    # --- LOSS + OPTIMIZER ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

    # --- TRAINING ---
    EPOCHS = 2

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

    # --- SAVE MODEL ---
    torch.save(model.state_dict(), "model/image.pth")
    print("Image model trained successfully")


if __name__ == "__main__":
    train()