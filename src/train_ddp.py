import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import wandb
from datetime import timedelta

# Import your local project modules
from src.model import get_model
from src.dataset import get_dataloaders

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(
        "nccl", 
        rank=rank, 
        world_size=world_size,
        timeout=timedelta(seconds=300) 
    )
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    
    if rank == 0:
        wandb.init(
            project="CrystalVision-HPC",
            name="DDP-ResNet18-Copper-Melt",
            config={
                "learning_rate": 0.001,
                "architecture": "ResNet18",
                "dataset": "Copper-Melt-LAMMPS",
                "epochs": 10,
                "batch_size": 16,
                "world_size": world_size
            }
        )

    # Use the dataloader from your local dataset.py
    _, full_dataset = get_dataloaders('data/train')
    
    sampler = DistributedSampler(
        full_dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=True
    )
    
    train_loader = torch.utils.data.DataLoader(
        full_dataset, 
        batch_size=16, 
        sampler=sampler, 
        num_workers=4, 
        pin_memory=True
    )

    model = get_model(num_classes=2).to(rank)
    model = DDP(model, device_ids=[rank])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()

    model.train()
    for epoch in range(10):
        sampler.set_epoch(epoch)
        epoch_loss = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            # Unpack the tuple (image, label) returned by ImageFolder
            images, labels = batch_data
            images, labels = images.to(rank), labels.to(rank)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
            if rank == 0 and batch_idx % 5 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
                wandb.log({"batch_loss": loss.item(), "epoch": epoch})

        if rank == 0:
            avg_loss = epoch_loss / len(train_loader)
            wandb.log({"epoch_loss": avg_loss})
            print(f"--- Epoch {epoch} Completed. Average Loss: {avg_loss:.4f} ---")

    if rank == 0:
        os.makedirs("models", exist_ok=True)
        torch.save(model.module.state_dict(), "models/crystalline_classifier.pt")
        print("Training complete. Weights saved to models/crystalline_classifier.pt")
        wandb.finish()

    cleanup()

if __name__ == "__main__":
    # LOCAL_RANK is automatically set by torchrun
    if 'LOCAL_RANK' not in os.environ:
        print("Please run this script using torchrun:")
        print("torchrun --nproc_per_node=2 src/train_ddp.py")
    else:
        rank = int(os.environ['LOCAL_RANK'])
        world_size = torch.cuda.device_count()
        train(rank=rank, world_size=world_size)