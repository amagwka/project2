import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dataset import EmbeddingH5Dataset
from mlp_world_model import MLPWorldModel, mlp_loss


def parse_args():
    p = argparse.ArgumentParser(description="Train MLP world model on embeddings")
    p.add_argument("--h5", type=str, required=True, help="H5 file with embeddings")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--log-dir", type=str, default="runs/mlp_world_model")
    p.add_argument("--checkpoint", type=str, help="Optional checkpoint to resume from")
    p.add_argument("--no-bf16", dest="use_bf16", action="store_false", help="Disable bfloat16 mixed precision")
    p.set_defaults(use_bf16=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset = EmbeddingH5Dataset(args.h5, sequence_length=1)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = MLPWorldModel(hidden_dim=args.hidden_dim, num_layers=args.layers).to(args.device)
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
        epoch_mse = 0.0
        epoch_cos = 0.0
        for seq, target in loader:
            seq = seq.squeeze(1).to(args.device)
            target = target.to(args.device)
            with torch.amp.autocast(args.device, dtype=torch.bfloat16, enabled=args.use_bf16):
                pred = model(seq)
                mse_loss, cos_loss, loss = mlp_loss(pred, target)
            optim.zero_grad()
            loss.backward()
            optim.step()

            writer.add_scalar("loss/mse_step", mse_loss.item(), global_step)
            writer.add_scalar("loss/cosine_step", cos_loss.item(), global_step)
            writer.add_scalar("loss/step", loss.item(), global_step)
            epoch_loss += loss.item() * seq.size(0)
            epoch_mse += mse_loss.item() * seq.size(0)
            epoch_cos += cos_loss.item() * seq.size(0)
            global_step += 1
        dataset_len = len(loader.dataset)
        avg = epoch_loss / dataset_len
        avg_mse = epoch_mse / dataset_len
        avg_cos = epoch_cos / dataset_len
        writer.add_scalar("loss/epoch", avg, epoch)
        writer.add_scalar("loss/mse_epoch", avg_mse, epoch)
        writer.add_scalar("loss/cosine_epoch", avg_cos, epoch)
        print(f"Epoch {epoch+1}, avg loss {avg:.4f}")

    writer.close()
    save_path = Path("mlp_world_model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
