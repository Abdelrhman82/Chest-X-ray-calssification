import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2  # change this
class_names = ["class1", "class2"]  # change this

# Load Model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Prediction Function
def predict(image):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)

    return class_names[pred.item()]

# Gradio UI
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Image Classification (ResNet18)"
)

interface.launch()