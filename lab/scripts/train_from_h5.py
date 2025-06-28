import argparse
from torch.utils.data import DataLoader
import torch
from pathlib import Path
import sys

# Ensure imports work when the script is executed from within this directory
sys.path.append(str(Path(__file__).resolve().parents[1]))

from tmdn import create_tmdn_model, train_step
from dataset import EmbeddingH5Dataset


def parse_args():
    p = argparse.ArgumentParser(description="Train TMDN using cached embeddings")
    p.add_argument("--h5", type=str, required=True, help="H5 file with embeddings")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset = EmbeddingH5Dataset(args.h5)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = create_tmdn_model().to(args.device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        for seq, target in loader:
            seq = seq.to(args.device)
            target = target.to(args.device)
            loss, _ = train_step(model, seq, target)
            optim.zero_grad()
            loss.backward()
            optim.step()
        print(f"Epoch {epoch+1}, loss {loss.item():.4f}")

    save_path = Path("tmdn_model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
