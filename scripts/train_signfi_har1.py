#!/usr/bin/env python3
"""Train the published SignFi CNN architecture on official HAR-1 CSI."""

import argparse
import csv
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


class SignFiH5Dataset(Dataset):
    def __init__(self, path, indices, ranges):
        self.path = str(path)
        self.indices = np.asarray(indices)
        self.ranges = ranges
        self.handle = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        if self.handle is None:
            self.handle = h5py.File(self.path, "r")
        x = self.handle["x"][self.indices[item]].astype(np.float32)
        for channel, (low, high) in enumerate(self.ranges):
            x[channel] = np.clip((x[channel] - low) / max(high - low, 1e-8), 0, 1)
        # Official SignFi concatenates amplitude and phase along image width.
        x = np.concatenate([x[0], x[1]], axis=1)[None, ...]
        return torch.from_numpy(x), int(self.handle["y"][self.indices[item]])


class OriginalSignFiCNN(nn.Module):
    """SignFi example CNN: Conv4x4-BN-ReLU-Pool4x4-FC."""
    def __init__(self, height, width, classes=20):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=4, padding=0),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
        )
        with torch.no_grad():
            flattened = self.features(torch.zeros(1, 1, height, width)).numel()
        self.classifier = nn.Linear(flattened, classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.flatten(1))


class BeamSenseConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class BeamSenseSignFiVGG(nn.Module):
    """Single-monitor adaptation of the VGG classifier in BeamSense Fig. 7."""
    def __init__(self, height, width, classes=20):
        super().__init__()
        self.features = nn.Sequential(
            BeamSenseConvBlock(1, 128),
            BeamSenseConvBlock(128, 64),
            BeamSenseConvBlock(64, 32),
            nn.MaxPool2d(2),
        )
        with torch.no_grad():
            flattened = self.features(torch.zeros(1, 1, height, width)).numel()
        self.classifier = nn.Linear(flattened, classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def split_indices(labels, participants, split, test_participant, seed):
    all_idx = np.arange(len(labels))
    if split == "participant":
        test = all_idx[participants == test_participant]
        rest = all_idx[participants != test_participant]
        if not len(test):
            raise ValueError(f"participant {test_participant} has no test samples")
        if not len(rest):
            raise ValueError(f"participant {test_participant} leaves no training samples")
        train, val = train_test_split(
            rest, test_size=.1, random_state=seed, stratify=labels[rest]
        )
    else:
        train, temp = train_test_split(
            all_idx, test_size=.3, random_state=seed, stratify=labels
        )
        val, test = train_test_split(
            temp, test_size=.5, random_state=seed, stratify=labels[temp]
        )
    return train, val, test


def channel_ranges(path, indices, chunk=256):
    lows = np.full(2, np.inf)
    highs = np.full(2, -np.inf)
    with h5py.File(path, "r") as f:
        for begin in range(0, len(indices), chunk):
            batch = f["x"][np.sort(indices[begin:begin + chunk])]
            lows = np.minimum(lows, batch.min(axis=(0, 2, 3)))
            highs = np.maximum(highs, batch.max(axis=(0, 2, 3)))
    return list(zip(lows.tolist(), highs.tolist()))


def evaluate(model, loader, device):
    model.eval()
    true, pred = [], []
    with torch.no_grad():
        for x, y in loader:
            true.extend(y.numpy())
            pred.extend(model(x.to(device)).argmax(1).cpu().numpy())
    return {
        "accuracy": accuracy_score(true, pred),
        "macro_f1": f1_score(true, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(true, pred, average="macro", zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=["original-signfi", "beamsense-vgg"], default="beamsense-vgg")
    parser.add_argument("--split", choices=["random-window", "participant", "external"], default="random-window")
    parser.add_argument("--test-data", type=Path)
    parser.add_argument("--test-participant", type=int)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float,
                        help="default: 0.01 for original SignFi, 0.001 for BeamSense VGG")
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=.9)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.split == "participant" and args.test_participant is None:
        parser.error("participant split requires --test-participant")
    if args.split == "external" and args.test_data is None:
        parser.error("external split requires --test-data")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    with h5py.File(args.data, "r") as f:
        labels = f["y"][:]
        participants = f["participant"][:]
        _, channels, height, feature_width = f["x"].shape
        representation = str(f.attrs.get("representation", "SignFi amplitude + phase"))
    if channels != 2:
        raise ValueError(f"expected amplitude/phase channels=2, got {channels}")

    if args.split == "external":
        all_idx = np.arange(len(labels))
        train, val = train_test_split(
            all_idx, test_size=.1, random_state=args.seed, stratify=labels
        )
        with h5py.File(args.test_data, "r") as f:
            test_labels = f["y"][:]
            test_input_shape = tuple(f["x"].shape[1:])
        if test_input_shape != (channels, height, feature_width):
            raise ValueError(
                f"train input shape {(channels, height, feature_width)} differs "
                f"from external test shape {test_input_shape}"
            )
        if set(np.unique(test_labels)) - set(np.unique(labels)):
            raise ValueError("external test data contains labels absent from training data")
        test = np.arange(len(test_labels))
    else:
        train, val, test = split_indices(
            labels, participants, args.split, args.test_participant, args.seed
        )
    ranges = channel_ranges(args.data, train)
    paths = {"train": args.data, "val": args.data,
             "test": args.test_data if args.split == "external" else args.data}
    loaders = {
        name: DataLoader(
            SignFiH5Dataset(paths[name], indices, ranges),
            batch_size=args.batch_size, shuffle=name == "train",
            num_workers=2, pin_memory=True
        )
        for name, indices in [("train", train), ("val", val), ("test", test)]
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "original-signfi":
        learning_rate = args.lr if args.lr is not None else 1e-2
        model = OriginalSignFiCNN(height, feature_width * 2).to(device)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        optimizer_name = "SGD with momentum (official SignFi example settings)"
    else:
        learning_rate = args.lr if args.lr is not None else 1e-3
        model = BeamSenseSignFiVGG(height, feature_width * 2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optimizer_name = "Adam (BeamSense paper does not report optimizer hyperparameters)"
    criterion = nn.CrossEntropyLoss()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best, stale, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in loaders["train"]:
            optimizer.zero_grad()
            loss = criterion(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        val_metrics = evaluate(model, loaders["val"], device)
        row = {"epoch": epoch, "loss": float(np.mean(losses)),
               **{f"val_{key}": value for key, value in val_metrics.items()}}
        history.append(row)
        print(row)
        if val_metrics["accuracy"] > best:
            best = val_metrics["accuracy"]
            stale = 0
            torch.save(model.state_dict(), args.output_dir / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(torch.load(args.output_dir / "best.pt", map_location=device))
    result = {
        "model": args.model,
        "model_source": ("official SignFi example architecture; output 276->20"
                         if args.model == "original-signfi" else
                         "BeamSense Fig. 7 VGG classifier + SignFi CSI preprocessing; single-monitor adaptation"),
        "dataset": "CSI-BFI-HAR / HAR-1 official CSI",
        "input_representation": representation,
        "input_shape": [1, height, feature_width * 2],
        "split": args.split,
        "test_participant": args.test_participant,
        "train_data": str(args.data),
        "test_data": str(paths["test"]),
        "train_samples": len(train),
        "validation_samples": len(val),
        "test_samples": len(test),
        "normalization": "train-global per-component minmax",
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "seed": args.seed,
        **evaluate(model, loaders["test"], device),
    }
    with open(args.output_dir / "history.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0])
        writer.writeheader()
        writer.writerows(history)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
