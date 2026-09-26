import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.transforms import InterpolationMode


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class DatasetInfo:
    class_to_idx: dict[str, int]
    idx_to_class: dict[int, str]
    train_size: int
    val_size: int
    test_size: int


class MRIPreprocessor:
    def __init__(self, margin_percent: float = 0.05, min_foreground_ratio: float = 0.02) -> None:
        self.margin_percent = margin_percent
        self.min_foreground_ratio = min_foreground_ratio

    def _remove_black_margin(self, image: Image.Image) -> Image.Image:
        gray = ImageOps.grayscale(image)
        array = np.array(gray)

        threshold = max(8, int(np.percentile(array, 10)))
        foreground_mask = array > threshold

        if foreground_mask.mean() < self.min_foreground_ratio:
            return gray

        ys, xs = np.where(foreground_mask)
        y1, y2 = ys.min(), ys.max() + 1
        x1, x2 = xs.min(), xs.max() + 1

        height, width = array.shape
        pad = int(max(y2 - y1, x2 - x1) * self.margin_percent)

        y1 = max(0, y1 - pad)
        y2 = min(height, y2 + pad)
        x1 = max(0, x1 - pad)
        x2 = min(width, x2 + pad)

        cropped = array[y1:y2, x1:x2]
        return Image.fromarray(cropped)

    def __call__(self, image: Image.Image) -> Image.Image:
        if image.mode != "L":
            image = ImageOps.grayscale(image)

        image = self._remove_black_margin(image)
        image = ImageOps.autocontrast(image)
        return image.convert("RGB")


def build_train_transform(image_size: int) -> transforms.Compose:
    preprocess = MRIPreprocessor()

    return transforms.Compose(
        [
            preprocess,
            transforms.Resize((image_size + 32, image_size + 32), interpolation=InterpolationMode.BILINEAR),
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15, interpolation=InterpolationMode.BILINEAR, fill=0),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1), fill=0),
            transforms.ColorJitter(brightness=0.15, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.08), ratio=(0.3, 3.3), value=0),
        ]
    )


def build_eval_transform(image_size: int) -> transforms.Compose:
    preprocess = MRIPreprocessor()

    return transforms.Compose(
        [
            preprocess,
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def stratified_split_indices(targets: list[int], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    per_class_indices: dict[int, list[int]] = {}

    for index, label in enumerate(targets):
        per_class_indices.setdefault(label, []).append(index)

    train_indices: list[int] = []
    val_indices: list[int] = []

    for label_indices in per_class_indices.values():
        shuffled = label_indices[:]
        rng.shuffle(shuffled)

        val_size = max(1, int(len(shuffled) * val_ratio))
        val_indices.extend(shuffled[:val_size])
        train_indices.extend(shuffled[val_size:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None

    try:
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def compute_metrics(targets: list[int], predictions: list[int], num_classes: int) -> dict[str, float]:
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for target, pred in zip(targets, predictions):
        confusion[target, pred] += 1

    tp = confusion.diag().to(torch.float32)
    fp = confusion.sum(dim=0).to(torch.float32) - tp
    fn = confusion.sum(dim=1).to(torch.float32) - tp

    precision_per_class = tp / torch.clamp(tp + fp, min=1.0)
    recall_per_class = tp / torch.clamp(tp + fn, min=1.0)
    f1_per_class = 2 * precision_per_class * recall_per_class / torch.clamp(precision_per_class + recall_per_class, min=1e-8)

    accuracy = (tp.sum() / torch.clamp(confusion.sum(), min=1)).item()
    macro_precision = precision_per_class.mean().item()
    macro_recall = recall_per_class.mean().item()
    macro_f1 = f1_per_class.mean().item()

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    optimizer: optim.Optimizer | None = None,
) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    running_loss = 0.0
    total = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, targets)

            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

        predictions = torch.argmax(logits, dim=1)
        all_targets.extend(targets.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    avg_loss = running_loss / max(total, 1)
    metrics = compute_metrics(all_targets, all_predictions, num_classes=num_classes)
    return avg_loss, metrics


def create_dataloaders(
    train_dir: Path,
    test_dir: Path,
    image_size: int,
    val_ratio: float,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader, DatasetInfo]:
    base_train = datasets.ImageFolder(root=str(train_dir))
    class_to_idx = base_train.class_to_idx
    idx_to_class = {index: name for name, index in class_to_idx.items()}

    train_transform = build_train_transform(image_size)
    eval_transform = build_eval_transform(image_size)

    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(train_dir), transform=eval_transform)
    test_dataset = datasets.ImageFolder(root=str(test_dir), transform=eval_transform)

    train_indices, val_indices = stratified_split_indices(base_train.targets, val_ratio=val_ratio, seed=seed)

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    num_workers = min(8, (torch.get_num_threads() or 1))
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    info = DatasetInfo(
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        train_size=len(train_subset),
        val_size=len(val_subset),
        test_size=len(test_dataset),
    )
    return train_loader, val_loader, test_loader, info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train robust MRI tumor classifier.")
    parser.add_argument("--train-dir", type=Path, default=Path("archive/Training"))
    parser.add_argument("--test-dir", type=Path, default=Path("archive/Testing"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("mri_tumor_classifier.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not args.train_dir.exists() or not args.test_dir.exists():
        raise FileNotFoundError("Expected dataset folders at ./archive/Training and ./archive/Testing")

    device = get_best_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, info = create_dataloaders(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    print("Class to Index Mapping:", info.class_to_idx)
    print(f"Train samples: {info.train_size} | Val samples: {info.val_size} | Test samples: {info.test_size}")

    model = build_model(num_classes=len(info.class_to_idx), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_state = None
    best_val_f1 = -1.0
    num_classes = len(info.class_to_idx)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            num_classes=num_classes,
            optimizer=optimizer,
        )
        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            num_classes=num_classes,
            optimizer=None,
        )

        scheduler.step()

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_metrics['accuracy']:.4f} train_f1={train_metrics['macro_f1']:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Failed to capture best model state.")

    model.load_state_dict(best_state)
    test_loss, test_metrics = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        num_classes=num_classes,
        optimizer=None,
    )

    print("\nTest metrics:")
    print(f"Loss          : {test_loss:.4f}")
    print(f"Accuracy      : {test_metrics['accuracy']:.4f}")
    print(f"Macro Precision: {test_metrics['macro_precision']:.4f}")
    print(f"Macro Recall  : {test_metrics['macro_recall']:.4f}")
    print(f"Macro F1      : {test_metrics['macro_f1']:.4f}")

    checkpoint = {
        "model_name": "resnet18",
        "model_state_dict": best_state,
        "class_to_idx": info.class_to_idx,
        "idx_to_class": info.idx_to_class,
        "image_size": args.image_size,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "best_val_macro_f1": best_val_f1,
    }

    torch.save(checkpoint, str(args.output))
    print(f"Saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()