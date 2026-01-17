import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size=16):
    # Verify path exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {os.path.abspath(data_dir)}")

    print(f"Loading data from: {os.path.abspath(data_dir)}")
    print(f"Subdirectories found: {os.listdir(data_dir)}")

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
    
    loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True)
    
    return loader, image_dataset

if __name__ == "__main__":
    # Test locally
    try:
        loader, dataset = get_dataloaders('data/train')
        print(f"Successfully loaded {len(dataset)} images.")
        print(f"Classes found: {dataset.classes}")
    except Exception as e:
        print(f"ERROR: {e}")