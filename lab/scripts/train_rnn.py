import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add lab directory to path for imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dataset import EmbeddingH5Dataset
from rnn_baseline import RNNPredictor, rnn_loss


def parse_args():
    p = argparse.ArgumentParser(description="Train RNN baseline on embeddings")
    p.add_argument("--h5", type=str, required=True, help="H5 file with embeddings")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--model", type=str, choices=["lstm", "gru"], default="lstm")
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--log-dir", type=str, default="runs/rnn_baseline")
    p.add_argument("--frame-gap", type=int, default=1, help="Target frames ahead to predict")
    p.add_argument("--checkpoint", type=str, help="Optional checkpoint to resume from")
    return p.parse_args()


def main():
    args = parse_args()
    dataset = EmbeddingH5Dataset(args.h5, 30, args.frame_gap)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = RNNPredictor(hidden_dim=args.hidden_dim, rnn_type=args.model).to(args.device)
    if args.checkpoint:
        try:
            state = torch.load(args.checkpoint, map_location=args.device)
            model.load_state_dict(state)
            print(f"Loaded checkpoint {args.checkpoint}")
        except Exception as e:
            print(f"Failed to load {args.checkpoint}: {e}")
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter(log_dir=args.log_dir)

    global_step = 0
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for seq, target in loader:
            seq = seq.to(args.device)
            target = target.to(args.device)
            pred = model(seq)
            loss = rnn_loss(pred, target)
            optim.zero_grad()
            loss.backward()
            optim.step()

            writer.add_scalar("loss/step", loss.item(), global_step)
            epoch_loss += loss.item() * seq.size(0)
            global_step += 1
        avg = epoch_loss / len(loader.dataset)
        writer.add_scalar("loss/epoch", avg, epoch)
        print(f"Epoch {epoch+1}, avg loss {avg:.4f}")

    writer.close()
    save_path = Path(f"rnn_{args.model}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
