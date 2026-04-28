import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = "cpu"

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load("model/image.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def predict_image(path):
    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(img)
        prob = torch.softmax(out, dim=1)[0]

    return int(prob.argmax()), float(prob.max())