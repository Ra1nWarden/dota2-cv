#!/usr/bin/env python3
"""Train an EfficientNet-B0 classifier on synthetic Dota 2 icon data.

Supports 3-phase progressive unfreezing, mixed precision (bfloat16),
weighted sampling, and TensorBoard logging.

Usage:
    python scripts/train.py \
        --data /workspace/data/synthetic \
        --category heroes \
        --output /workspace/models/hero_classifier.pt \
        --epochs 30 --batch-size 128
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(num_classes: int) -> nn.Module:
    """EfficientNet-B0 with custom classifier head."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1280, num_classes),
    )
    return model


def freeze_backbone(model: nn.Module):
    """Freeze all backbone (features) parameters."""
    for param in model.features.parameters():
        param.requires_grad = False


def unfreeze_last_blocks(model: nn.Module, n_blocks: int = 2):
    """Unfreeze last n EfficientNet feature blocks."""
    total = len(model.features)
    for i in range(total - n_blocks, total):
        for param in model.features[i].parameters():
            param.requires_grad = True


def unfreeze_all(model: nn.Module):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True


def build_optimizer(model: nn.Module, lr: float, phase: int) -> AdamW:
    """Build optimizer with differential learning rates per unfreezing phase."""
    if phase == 1:
        # Only classifier is trainable
        params = [{"params": model.classifier.parameters(), "lr": lr}]
    elif phase == 2:
        # Last 2 blocks + classifier
        backbone_params = []
        total = len(model.features)
        for i in range(total - 2, total):
            backbone_params.extend(model.features[i].parameters())
        params = [
            {"params": backbone_params, "lr": lr / 10},
            {"params": model.classifier.parameters(), "lr": lr},
        ]
    else:
        # All layers
        backbone_params = list(model.features.parameters())
        params = [
            {"params": backbone_params, "lr": lr / 100},
            {"params": model.classifier.parameters(), "lr": lr / 10},
        ]
    return AdamW(params, weight_decay=0.01)


def get_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    """Build weighted sampler for class balance."""
    targets = np.array(dataset.targets)
    class_counts = np.bincount(targets)
    weights_per_class = 1.0 / class_counts
    sample_weights = weights_per_class[targets]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    scaler: GradScaler,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch, return (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast("cuda", dtype=torch.bfloat16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validate, return (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with autocast("cuda", dtype=torch.bfloat16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def get_unfreezing_phase(epoch: int) -> int:
    """Determine which unfreezing phase we're in (1-indexed epoch)."""
    if epoch <= 5:
        return 1
    elif epoch <= 15:
        return 2
    return 3


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 icon classifier")
    parser.add_argument("--data", required=True, help="Root synthetic data directory")
    parser.add_argument("--category", required=True, choices=["heroes", "items"],
                        help="Which category to train")
    parser.add_argument("--output", required=True, help="Output .pt checkpoint path")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--size", type=int, default=128, help="Image size")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logdir", default="runs", help="TensorBoard log directory")
    args = parser.parse_args()

    set_seed(args.seed)

    data_root = Path(args.data)
    train_dir = data_root / "train" / args.category
    val_dir = data_root / "val" / args.category

    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training data not found: {train_dir}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Validation data not found: {val_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Data transforms
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes
    print(f"Category: {args.category}")
    print(f"Classes: {num_classes}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    sampler = get_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    # Model
    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler("cuda")

    # TensorBoard
    writer = SummaryWriter(log_dir=f"{args.logdir}/{args.category}")

    # Training loop
    best_val_acc = 0.0
    best_state = None
    best_epoch = 0
    patience_counter = 0
    current_phase = 0

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        phase = get_unfreezing_phase(epoch)

        # Reconfigure model and optimizer on phase transition
        if phase != current_phase:
            current_phase = phase
            if phase == 1:
                freeze_backbone(model)
            elif phase == 2:
                unfreeze_last_blocks(model, n_blocks=2)
            else:
                unfreeze_all(model)

            optimizer = build_optimizer(model, args.lr, phase)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"\n--- Unfreezing phase {phase} (epoch {epoch}) ---")
            print(f"Trainable: {trainable:,} / {total_params:,} params")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        # Log
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("accuracy", {"train": train_acc, "val": val_acc}, epoch)
        writer.add_scalar("lr/head", optimizer.param_groups[-1]["lr"], epoch)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        # Checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"  -> New best val_acc: {val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # Save best checkpoint
    checkpoint = {
        "model_state_dict": best_state,
        "class_names": class_names,
        "num_classes": num_classes,
        "epoch": best_epoch,
        "val_accuracy": best_val_acc,
        "image_size": args.size,
    }
    torch.save(checkpoint, args.output)
    print(f"\nSaved best model (epoch {best_epoch}, val_acc={best_val_acc:.4f}) to {args.output}")

    # Save class mapping
    classes_path = Path(args.output).parent.parent / "configs" / f"{args.category}_classes.json"
    classes_path.parent.mkdir(parents=True, exist_ok=True)
    with open(classes_path, "w") as f:
        json.dump({i: name for i, name in enumerate(class_names)}, f, indent=2)
    print(f"Saved class mapping to {classes_path}")

    writer.close()


if __name__ == "__main__":
    main()
