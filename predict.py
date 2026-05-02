

import argparse
import yaml
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import numpy as np
from pathlib import Path
from PIL import Image


# ── Load config ────────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

IMG_SIZE    = cfg["data"]["image_size"]     # 224
NUM_CLASSES = cfg["data"]["num_classes"]    # 6
CLASS_MAP   = cfg["classes"]               # {Plastic:0, ...}

# Reverse mapping: index → class name
IDX_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}

# ── Model definition — must match training exactly ─────────────────────────────
def build_model(num_classes: int, feat_dim: int = 960) -> nn.Module:
    """Rebuild the exact same MobileNetTrash architecture used in training."""
    backbone = timm.create_model(
        "mobilenetv3_large_100",
        pretrained=False,       # no pretrained weights — we load our own below
        num_classes=0,
        global_pool="avg"
    )

    class MobileNetTrash(nn.Module):
        def __init__(self, backbone, feat_dim, num_classes):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Sequential(
                nn.BatchNorm1d(feat_dim),
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes)
            )

        def forward(self, x):
            return self.head(self.backbone(x))

    return MobileNetTrash(backbone, feat_dim, num_classes)


# ── Transform — same as val_transform in training ─────────────────────────────
# Must match exactly — same resize, same normalization values
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load model weights from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Get feature dim dynamically from a dummy forward pass
    dummy_backbone = timm.create_model(
        "mobilenetv3_large_100", pretrained=False, num_classes=0, global_pool="avg"
    )
    with torch.no_grad():
        feat_dim = dummy_backbone(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)).shape[1]

    model = build_model(num_classes=NUM_CLASSES, feat_dim=feat_dim)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    print(f"Model loaded — trained at epoch {ckpt['epoch']}, "
          f"val_acc={ckpt['val_acc']:.4f}")
    return model


def predict(image_path: str, model: nn.Module, device: torch.device) -> dict:
    """
    Run inference on a single image.

    Returns:
        dict with keys:
            predicted_class  — e.g. "Plastic"
            confidence       — e.g. 0.9231 (92.31%)
            all_scores       — dict of {class: probability} for all 6 classes
    """
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    tensor = val_transform(image).unsqueeze(0).to(device)  # add batch dim

    with torch.no_grad():
        logits = model(tensor)                          # raw scores shape [1, 6]
        probs  = torch.softmax(logits, dim=1)[0]       # probabilities shape [6]

    probs_np = probs.cpu().numpy()
    pred_idx = int(np.argmax(probs_np))

    return {
        "predicted_class" : IDX_TO_CLASS[pred_idx],
        "confidence"      : float(probs_np[pred_idx]),
        "all_scores"      : {
            IDX_TO_CLASS[i]: float(probs_np[i])
            for i in range(NUM_CLASSES)
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Trash classification inference")
    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image (jpg, png)"
    )
    parser.add_argument(
        "--checkpoint",
        default="models/checkpoints/best_model.pth",
        help="Path to best_model.pth (default: models/checkpoints/best_model.pth)"
    )
    args = parser.parse_args()

    # Validate paths
    if not Path(args.image).exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Download best_model.pth from Google Drive and place it at "
            "models/checkpoints/best_model.pth"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model and predict
    model  = load_model(args.checkpoint, device)
    result = predict(args.image, model, device)

    # Print results
    print(f"\nImage      : {args.image}")
    print(f"Prediction : {result['predicted_class']}")
    print(f"Confidence : {result['confidence']*100:.1f}%")
    print(f"\nAll class scores:")
    for cls, score in sorted(result["all_scores"].items(),
                             key=lambda x: x[1], reverse=True):
        bar = "█" * int(score * 30)
        print(f"  {cls:<10}  {score*100:5.1f}%  {bar}")

    return result


if __name__ == "__main__":
    main()
