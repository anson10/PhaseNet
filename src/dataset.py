import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(data_dir, batch_size=16, val_split=0.15, test_split=0.15, seed=42):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {os.path.abspath(data_dir)}")

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(data_dir, transform=train_transforms)
    total = len(full_dataset)

    val_size  = int(total * val_split)
    test_size = int(total * test_split)
    train_size = total - val_size - test_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Val and test sets should not have augmentation
    val_dataset.dataset  = datasets.ImageFolder(data_dir, transform=eval_transforms)
    test_dataset.dataset = datasets.ImageFolder(data_dir, transform=eval_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Dataset split — Train: {train_size} | Val: {val_size} | Test: {test_size}")
    print(f"Classes: {full_dataset.classes}")

    return train_loader, val_loader, test_loader, full_dataset


if __name__ == "__main__":
    train_loader, val_loader, test_loader, dataset = get_dataloaders('data/train')
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")
