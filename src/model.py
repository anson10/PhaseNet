import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=2):
    # Load a pre-trained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
 
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

if __name__ == "__main__":
    # Test the model with a dummy tensor
    test_model = get_model()
    print(f"Model Architecture:\n{test_model.fc}")