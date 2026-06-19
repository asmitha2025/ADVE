import json
import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from training.model import ReconstructionMLP, ReconstructionLoss


class ReconstructionDataset(Dataset):
    def __init__(self, json_path: str):
        with open(json_path) as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s = self.data[idx]
        return {
            "anchor_emb":  torch.tensor(s["anchor_emb"],  dtype=torch.float32),
            "object_pool": torch.tensor(s["object_pool"], dtype=torch.float32),
            "delta_vec":   torch.tensor(s["delta_vec"][:128], dtype=torch.float32),
            "target_emb":  torch.tensor(s["target_emb"],  dtype=torch.float32),
        }


def train(json_path: str, epochs: int = 50, device: str = "cuda" if torch.cuda.is_available() else "cpu", checkpoint_path: str = "training/checkpoints/best_model.pt"):
    dataset    = ReconstructionDataset(json_path)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

    model     = ReconstructionMLP().to(device)
    criterion = ReconstructionLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_sim  = 0.0

        for batch in dataloader:
            anchor  = batch["anchor_emb"].to(device)
            pool    = batch["object_pool"].to(device)
            delta   = batch["delta_vec"].to(device)
            target  = batch["target_emb"].to(device)

            pred = model(anchor, pool, delta)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            sim = (pred * target).sum(dim=-1).mean().item()
            total_loss += loss.item()
            total_sim  += sim

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        avg_sim  = total_sim  / len(dataloader)

        print(f"Epoch {epoch+1:3d} | loss={avg_loss:.4f} | cosine_sim={avg_sim:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  → Saved best model (sim={avg_sim:.4f})")

    print(f"\nTraining complete. Best cosine sim: {1 - best_loss:.4f}")
    return model


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data",   default="training/data/samples.json")
    p.add_argument("--epochs", type=int, default=50)
    args = p.parse_args()

    os.makedirs("training/checkpoints", exist_ok=True)
    train(args.data, args.epochs)
