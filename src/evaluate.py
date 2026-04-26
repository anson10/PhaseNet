import os
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from src.model import get_model
from src.dataset import get_dataloaders


def evaluate(
    model_path: str = "models/crystalline_classifier.pt",
    data_dir: str = "data/train",
    batch_size: int = 16,
    val_split: float = 0.2,
):
    """
    Loads a trained PhaseNet model and evaluates it on a validation split of
    the dataset, printing accuracy, F1-score, precision, and recall.

    Args:
        model_path:  Path to the saved model weights (.pt file).
        data_dir:    Directory containing the labelled image folders.
        batch_size:  Batch size for the dataloader.
        val_split:   Fraction of data to use as validation set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    _, full_dataset = get_dataloaders(data_dir, batch_size=batch_size)
    class_names = full_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Total images: {len(full_dataset)}")

    # Split into train / validation
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    _, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Validation set size: {len(val_dataset)} images ({val_split*100:.0f}%)\n")

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ── Load model ────────────────────────────────────────────────────────────
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights not found at '{model_path}'.\n"
            "Train the model first with: torchrun --nproc_per_node=2 src/train_ddp.py"
        )

    model = get_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded weights from: {model_path}")

    # ── Run inference ─────────────────────────────────────────────────────────
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # ── Compute metrics ───────────────────────────────────────────────────────
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")

    print("\n" + "=" * 50)
    print("         EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Accuracy  : {acc * 100:.2f}%")
    print(f"  F1-Score  : {f1 * 100:.2f}%")
    print(f"  Precision : {precision * 100:.2f}%")
    print(f"  Recall    : {recall * 100:.2f}%")
    print("=" * 50)

    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    print("Confusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(all_labels, all_preds)
    print(f"  Labels : {class_names}")
    print(f"  {cm}")

    return {
        "accuracy": acc,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
    }


if __name__ == "__main__":
    evaluate()
