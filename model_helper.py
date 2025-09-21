import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

trained_model = None
class_names = ['F_Banana',
 'F_Lemon',
 'F_Lulo',
 'F_Mango',
 'F_Orange',
 'F_Strawberry',
 'F_Tamarillo',
 'F_Tomato',
 'S_Banana',
 'S_Lemon',
 'S_Lulo',
 'S_Mango',
 'S_Orange',
 'S_Strawberry',
 'S_Tamarillo',
 'S_Tomato']

# Load the pre-trained ResNet model
class FruitCNNResNet(nn.Module):
    def __init__(self, num_classes=16):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')
        # Freeze all layers except the final fully connected layer
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze layer4 and fc layers
        for param in self.model.layer4.parameters():
            param.requires_grad = True

            # Replace the final fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x


def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    
    image_tensor = transform(image).unsqueeze(0)

    global trained_model

    if trained_model is None:
        trained_model = FruitCNNResNet()
        trained_model.load_state_dict(torch.load("model/saved_model.pth", map_location=torch.device('cpu')))
        trained_model.eval()

    with torch.no_grad():
        output = trained_model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        return class_names[predicted_class.item()]
    

if __name__ == "__main__":
    print(predict("FRUIT-16K/F_Tomato/3.jpg"))
