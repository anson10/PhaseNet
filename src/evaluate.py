import os
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

from src.model import get_model
from src.dataset import get_dataloaders


def evaluate(
    model_path: str = "models/crystalline_classifier.pt",
    data_dir:   str = "data/train",
    batch_size: int = 16,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, _, test_loader, full_dataset = get_dataloaders(data_dir, batch_size=batch_size)
    class_names = full_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Test set size: {len(test_loader.dataset)} images\n")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights not found at '{model_path}'.\n"
            "Train first with: torchrun --nproc_per_node=2 src/train_ddp.py"
        )

    model = get_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded weights from: {model_path}")

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images  = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc       = accuracy_score(all_labels, all_preds)
    f1        = f1_score(all_labels, all_preds, average="weighted")
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall    = recall_score(all_labels, all_preds, average="weighted")

    print("\n" + "=" * 50)
    print("         EVALUATION RESULTS (TEST SET)")
    print("=" * 50)
    print(f"  Accuracy  : {acc * 100:.2f}%")
    print(f"  F1-Score  : {f1 * 100:.2f}%")
    print(f"  Precision : {precision * 100:.2f}%")
    print(f"  Recall    : {recall * 100:.2f}%")
    print("=" * 50)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(f"  Labels: {class_names}")
    print(f"  {confusion_matrix(all_labels, all_preds)}")

    return {"accuracy": acc, "f1_score": f1, "precision": precision, "recall": recall}


if __name__ == "__main__":
    evaluate()
