import argparse
from pathlib import Path
import h5py
import torch
from torchvision import transforms
from PIL import Image

try:
    from dinov2.models import vit_small, load_pretrained_weights
except Exception:  # pragma: no cover - dinov2 may not be installed in tests
    vit_small = None


def load_model(device: str):
    if vit_small is None:
        raise RuntimeError("dinov2 is required for embedding extraction")
    model = vit_small()
    load_pretrained_weights(model, model_name="dinov2_vits14")
    model.to(device)
    model.eval()
    return model


def transform_image():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
    ])


def encode_image(img_path: Path, model, transform, device: str):
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor).squeeze(0).cpu()
    return emb


def process_sequence(seq_dir: Path, model, transform, device: str):
    images = sorted(seq_dir.glob("*.png"))
    embeddings = []
    for img in images:
        embeddings.append(encode_image(img, model, transform, device))
    if embeddings:
        return torch.stack(embeddings)
    return torch.empty(0, 384)


def main() -> None:
    p = argparse.ArgumentParser(description="Cache DINOv2 embeddings to H5")
    p.add_argument("--image-dir", type=str, required=True, help="Root directory with frame folders")
    p.add_argument("--output", type=str, required=True, help="Output H5 file")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    transform = transform_image()
    model = load_model(args.device)

    with h5py.File(args.output, "w") as h5f:
        for seq_dir in sorted(Path(args.image_dir).iterdir()):
            if not seq_dir.is_dir():
                continue
            emb = process_sequence(seq_dir, model, transform, args.device)
            h5f.create_dataset(seq_dir.name, data=emb.numpy())


if __name__ == "__main__":
    main()
