import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from datetime import timedelta

from src.model import get_model
from src.dataset import get_dataloaders

EPOCHS        = 50
BATCH_SIZE    = 16
LR            = 1e-3
PATIENCE      = 7   # early stopping patience
LR_PATIENCE   = 3   # reduce LR after this many epochs of no val improvement


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

    train_loader, val_loader, _, full_dataset = get_dataloaders('data/train', batch_size=BATCH_SIZE)

    # Replace train_loader with a distributed sampler version
    train_sampler = DistributedSampler(
        train_loader.dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    model = get_model(num_classes=2).to(rank)
    model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=LR_PATIENCE, factor=0.5)
    scaler    = GradScaler()

    best_val_loss    = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        train_sampler.set_epoch(epoch)
        epoch_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(rank), labels.to(rank)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss    = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        # --- Validation (rank 0 only) ---
        if rank == 0:
            model.eval()
            val_loss    = 0
            correct     = 0
            total       = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(rank), labels.to(rank)
                    with autocast():
                        outputs = model(images)
                        loss    = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, preds  = torch.max(outputs, 1)
                    correct  += (preds == labels).sum().item()
                    total    += labels.size(0)

            avg_val_loss = val_loss / len(val_loader)
            val_acc      = 100 * correct / total

            print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}%")

            scheduler.step(avg_val_loss)

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss    = avg_val_loss
                patience_counter = 0
                os.makedirs("models", exist_ok=True)
                torch.save(model.module.state_dict(), "models/crystalline_classifier.pt")
                print(f"  --> Best model saved (val loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  --> No improvement ({patience_counter}/{PATIENCE})")
                if patience_counter >= PATIENCE:
                    print("Early stopping triggered.")
                    break

    if rank == 0:
        print("Training complete. Best model at models/crystalline_classifier.pt")

    cleanup()


if __name__ == "__main__":
    if 'LOCAL_RANK' not in os.environ:
        print("Run with: torchrun --nproc_per_node=2 src/train_ddp.py")
    else:
        rank       = int(os.environ['LOCAL_RANK'])
        world_size = torch.cuda.device_count()
        train(rank=rank, world_size=world_size)
