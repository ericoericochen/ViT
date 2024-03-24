import argparse
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import json
import torch

import sys

sys.path.append("../")

from vision_transformer import ViT, Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--checkpoint_epoch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, required=True)

    return parser.parse_args()


def main(args):
    torch.manual_seed(args.seed)

    train_dataset = CIFAR10(root="../data", train=True, download=True)
    val_dataset = CIFAR10(root="../data", train=False, download=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    with open(args.model_config, "r") as f:
        model_config = json.load(f)

    model = ViT(**model_config)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        epochs=args.epochs,
        checkpoint_epoch=args.checkpoint_epoch,
        save_dir=args.save_dir,
    )

    trainer.train()


if __name__ == "__main__":
    args = parse_args()

    main(args)