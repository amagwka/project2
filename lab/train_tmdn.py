import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tmdn import create_tmdn_model, train_step
from dataset import VideoDataset


def parse_args():
    p = argparse.ArgumentParser(description="Train TMDN on a video dataset")
    p.add_argument('--data-dir', type=str, required=True, help='Directory with mp4 files')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main():
    args = parse_args()
    dataset = VideoDataset(args.data_dir)
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

    save_path = Path('tmdn_model.pt')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == '__main__':
    main()
