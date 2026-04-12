#!/usr/bin/env python3
"""Export a trained PyTorch checkpoint to ONNX format.

Usage:
    python scripts/export_onnx.py \
        --checkpoint /workspace/models/hero_classifier.pt \
        --output /workspace/models/hero_classifier.onnx
"""

import argparse

import torch
import onnx
from torchvision import models


def build_model(num_classes: int) -> torch.nn.Module:
    """Rebuild EfficientNet-B0 with custom classifier head."""
    model = models.efficientnet_b0(weights=None)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(1280, num_classes),
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch checkpoint to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", required=True, help="Output .onnx path")
    args = parser.parse_args()

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    num_classes = checkpoint["num_classes"]
    image_size = checkpoint["image_size"]
    class_names = checkpoint["class_names"]
    print(f"Classes: {num_classes}, Image size: {image_size}")
    print(f"Trained epoch: {checkpoint['epoch']}, Val accuracy: {checkpoint['val_accuracy']:.4f}")

    # Rebuild and load weights
    model = build_model(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Export
    dummy_input = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=17,
    )

    # Verify
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model exported and verified: {args.output}")


if __name__ == "__main__":
    main()
