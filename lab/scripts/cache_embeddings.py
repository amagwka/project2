import argparse
from pathlib import Path
import h5py
import torch
from torchvision import transforms
from PIL import Image

try:
    from transformers import AutoModel
except Exception:  # pragma: no cover - transformers may not be installed
    AutoModel = None


def load_model(device: str, variant: str):
    if AutoModel is None:
        raise RuntimeError("transformers is required for embedding extraction")

    model_map = {
        "small": "facebook/dinov2-small",
        "large": "facebook/dinov2-large",
        "aimv2": "apple/aimv2-large-patch14-224-distilled",
    }
    repo = model_map.get(variant, "facebook/dinov2-small")
    model = AutoModel.from_pretrained(repo)
    hidden_dim = getattr(model.config, "hidden_size", 768)
    model.to(device)
    model.eval()
    return model, hidden_dim



def transform_image():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def encode_image(img_path: Path, model, transform, device: str):
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        if hasattr(outputs, "last_hidden_state"):
            emb = outputs.last_hidden_state[:, 1:, :].mean(dim=1).squeeze(0)
        else:
            out = outputs[0] if isinstance(outputs, tuple) else outputs
            emb = out[:, 1:, :].mean(dim=1).squeeze(0)
        emb = emb.cpu()
    return emb


def process_sequence(seq_dir: Path, model, transform, device: str, hidden_dim: int):

    images = sorted(seq_dir.glob("*.png"))
    embeddings = []
    for img in images:
        embeddings.append(encode_image(img, model, transform, device))
    if embeddings:
        return torch.stack(embeddings)
    return torch.empty(0, hidden_dim)



def main() -> None:
    p = argparse.ArgumentParser(description="Cache DINOv2 embeddings to H5")
    p.add_argument("--image-dir", type=str, required=True, help="Root directory with frame folders")
    p.add_argument("--output", type=str, required=True, help="Output H5 file")
    p.add_argument(
        "--model",
        type=str,
        default="small",
        choices=["small", "large", "aimv2"],
        help="Embedding model variant",
    )

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    transform = transform_image()
    model, hidden_dim = load_model(args.device, args.model)


    with h5py.File(args.output, "w") as h5f:
        for seq_dir in sorted(Path(args.image_dir).iterdir()):
            if not seq_dir.is_dir():
                continue
            emb = process_sequence(seq_dir, model, transform, args.device, hidden_dim)
            h5f.create_dataset(seq_dir.name, data=emb.numpy())


if __name__ == "__main__":
    main()
