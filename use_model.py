import argparse
from pathlib import Path

import torch
from PIL import Image, ImageOps

from main import build_eval_transform, build_model, get_best_device


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    required_keys = {"model_state_dict", "class_to_idx", "idx_to_class", "image_size"}
    missing = required_keys.difference(checkpoint.keys())
    if missing:
        raise KeyError(f"Checkpoint missing keys: {sorted(missing)}")

    return checkpoint


def build_tta_images(image: Image.Image) -> list[Image.Image]:
    return [
        image,
        ImageOps.mirror(image),
        image.rotate(8, resample=Image.Resampling.BILINEAR, fillcolor=0),
        image.rotate(-8, resample=Image.Resampling.BILINEAR, fillcolor=0),
    ]


def predict_image(
    image_path: Path,
    model: torch.nn.Module,
    transform,
    idx_to_class: dict[int, str],
    device: torch.device,
    use_tta: bool,
) -> tuple[str, float, list[tuple[str, float]]]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path)
    tta_images = build_tta_images(image) if use_tta else [image]

    model.eval()
    probs_list = []

    with torch.no_grad():
        for aug_image in tta_images:
            tensor = transform(aug_image).unsqueeze(0).to(device)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            probs_list.append(probs)

    avg_probs = torch.mean(torch.cat(probs_list, dim=0), dim=0)
    pred_idx = int(torch.argmax(avg_probs).item())
    confidence = float(avg_probs[pred_idx].item())
    predicted_class = idx_to_class[pred_idx]

    sorted_probs = torch.argsort(avg_probs, descending=True)
    top_predictions = [(idx_to_class[int(i)], float(avg_probs[int(i)].item())) for i in sorted_probs[:4]]
    return predicted_class, confidence, top_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robust multiclass MRI tumor prediction.")
    parser.add_argument("image", type=Path, help="Path to an MRI image file")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("mri_tumor_classifier.pt"),
        help="Path to the saved classifier checkpoint (.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=get_best_device().type,
        choices=["cpu", "cuda", "mps"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable test-time augmentation",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this machine.")
    if args.device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_built() and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS requested but not available on this machine.")

    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)

    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {int(index): class_name for index, class_name in checkpoint["idx_to_class"].items()}

    model = build_model(num_classes=len(class_to_idx), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    image_size = int(checkpoint["image_size"])
    transform = build_eval_transform(image_size)

    predicted_class, confidence, top_predictions = predict_image(
        image_path=args.image,
        model=model,
        transform=transform,
        idx_to_class=idx_to_class,
        device=device,
        use_tta=not args.no_tta,
    )

    print(f"Image: {args.image}")
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print("Top probabilities:")
    for class_name, probability in top_predictions:
        print(f"  - {class_name:12s} {probability:.4f}")


if __name__ == "__main__":
    main()