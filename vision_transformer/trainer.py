from .vit import ViT
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch


def get_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


def eval_accuracy(
    model: ViT, dataloader: DataLoader, show_progress: bool = False
) -> float:
    device = next(model.parameters()).device

    total_correct = 0

    iterable = tqdm(dataloader) if show_progress else dataloader
    for X, Y in iterable:
        X = X.to(device)
        Y = Y.to(device)

        pred = model(X)
        pred_classes = pred.max(dim=-1).indices

        num_correct = (pred_classes == Y).sum().item()
        total_correct += num_correct

    accuracy = total_correct / len(dataloader.dataset)

    return accuracy


def eval_loss(model: ViT, dataloader: DataLoader, show_progress: bool = False) -> float:
    device = next(model.parameters()).device

    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    iterable = tqdm(dataloader) if show_progress else dataloader
    for X, Y in iterable:
        X = X.to(device)
        Y = Y.to(device)

        pred = model(X)
        loss = criterion(pred, Y)
        total_loss += loss.item()

    loss = total_loss / len(dataloader)
    return loss


class Trainer:
    def __init__(
        self,
        model: ViT,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        lr: float = 3e-4,
        epochs: int = 20,
        weight_decay: float = 0.0,
        checkpoint_epoch: int = 1,
        save_dir: str = None,
        show_eval_progress: bool = False,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.checkpoint_epoch = checkpoint_epoch
        self.save_dir = save_dir
        self.show_eval_progress = show_eval_progress

    def train(self):
        print("training...")

        num_iters = self.epochs * len(self.train_loader)
        device = get_device()
        criterion = torch.nn.CrossEntropyLoss()

        model = self.model
        optimizer = self.optimizer

        model.to(device)

        losses = []
        with tqdm(total=num_iters, position=0) as pbar:
            postfix = {
                "loss": 0.0,
                "train_accuracy": 0.0,
                "val_loss": "N/A",
                "val_accuracy": 0,
            }

            for epoch in range(self.epochs):
                for X, Y in self.train_loader:
                    X = X.to(device)
                    Y = Y.to(device)

                    pred = model(X)
                    loss = criterion(pred, Y)

                    postfix["loss"] = loss.item()
                    losses.append(loss.item())

                    # backprop
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pbar.set_postfix(postfix)
                    pbar.update(1)

                train_accuracy = eval_accuracy(
                    model, self.train_loader, self.show_eval_progress
                )
                postfix["train_accuracy"] = train_accuracy

                if self.val_loader:
                    val_accuracy = eval_accuracy(
                        model, self.val_loader, self.show_eval_progress
                    )
                    val_loss = eval_loss(
                        model, self.val_loader, self.show_eval_progress
                    )

                    postfix["val_accuracy"] = val_accuracy
                    postfix["val_loss"] = val_loss

        return {"losses": losses}
