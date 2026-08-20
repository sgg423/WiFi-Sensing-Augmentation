#!/usr/bin/env python3
"""Train published SenseFi UT-HAR architectures on HAR-1 CSI amplitude."""

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
from torch.utils.data import ConcatDataset, DataLoader, Dataset


class H5Dataset(Dataset):
    def __init__(self, path, indices, low, high):
        self.path, self.indices = str(path), np.asarray(indices)
        self.low, self.scale, self.handle = low, max(high - low, 1e-8), None

    def __len__(self): return len(self.indices)

    def __getitem__(self, item):
        if self.handle is None:
            self.handle = h5py.File(self.path, "r")
        x = self.handle["x"][self.indices[item]].astype(np.float32)
        x = np.clip((x - self.low) / self.scale, 0, 1)
        return torch.from_numpy(x), int(self.handle["y"][self.indices[item]])


class SenseFiLeNet(nn.Module):
    """Official SenseFi UT_HAR_LeNet, with a configurable class count."""
    def __init__(self, classes=20):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=(3, 1)), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (5, 4), stride=(2, 2), padding=(1, 0)),
            nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64, 96, (3, 3), stride=1), nn.ReLU(True), nn.MaxPool2d(2),
        )
        # Official UT-HAR inputs reach 4x4 here. Official HAR-1 CSI has 242
        # subcarriers and reaches 4x13, so pool the spatial dimensions while
        # retaining the published fully-connected layer size.
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Linear(96 * 4 * 4, 128), nn.ReLU(), nn.Linear(128, classes)
        )

    def forward(self, x):
        x = self.pool(self.encoder(x))
        return self.fc(torch.flatten(x, 1))


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
                                 nn.Conv2d(cout, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout))
        self.skip = nn.Identity() if stride == 1 and cin == cout else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        self.relu = nn.ReLU()

    def forward(self, x): return self.relu(self.net(x) + self.skip(x))


class SenseFiResNet18(nn.Module):
    """Official SenseFi UT_HAR_ResNet18, with a configurable class count."""
    def __init__(self, classes=20):
        super().__init__()
        self.reshape = nn.Sequential(
            nn.Conv2d(1, 3, 7, stride=(3, 1)), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(3, 3, kernel_size=(10, 11), stride=1), nn.ReLU(),
        )
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, 2, 1))
        layers, cin = [], 64
        for cout, stride in [(64, 1), (128, 2), (256, 2), (512, 2)]:
            layers += [Block(cin, cout, stride), Block(cout, cout)]
            cin = cout
        self.layers = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, classes))

    def forward(self, x): return self.head(self.layers(self.stem(self.reshape(x))))


def split_indices(labels, participants, split, test_participant, seed):
    all_idx = np.arange(len(labels))
    if split == "participant":
        test = all_idx[participants == test_participant]
        rest = all_idx[participants != test_participant]
        train, val = train_test_split(rest, test_size=.1, random_state=seed, stratify=labels[rest])
    else:
        train, temp = train_test_split(all_idx, test_size=.3, random_state=seed, stratify=labels)
        val, test = train_test_split(temp, test_size=.5, random_state=seed, stratify=labels[temp])
    return train, val, test


def split_source_traces(labels, sources, seed):
    """Split whole source traces while retaining every class in each fold."""
    labels = np.asarray(labels)
    sources = np.asarray(sources).astype(str)
    source_labels = {}
    for source, label in zip(sources, labels):
        previous = source_labels.setdefault(source, int(label))
        if previous != int(label):
            raise ValueError(f"source trace {source!r} contains multiple activity labels")

    rng = np.random.default_rng(seed)
    train_sources, val_sources, test_sources = [], [], []
    for label in sorted(np.unique(labels)):
        class_sources = np.array(
            [source for source, source_label in source_labels.items() if source_label == label],
            dtype=object,
        )
        rng.shuffle(class_sources)
        if len(class_sources) < 3:
            raise ValueError(
                f"source-trace split requires at least 3 source traces per class; "
                f"class {label} has {len(class_sources)}"
            )
        n_test = max(1, int(round(len(class_sources) * .15)))
        n_val = max(1, int(round(len(class_sources) * .15)))
        if n_test + n_val >= len(class_sources):
            n_test = n_val = 1
        test_sources.extend(class_sources[:n_test])
        val_sources.extend(class_sources[n_test:n_test + n_val])
        train_sources.extend(class_sources[n_test + n_val:])

    train = np.flatnonzero(np.isin(sources, train_sources))
    val = np.flatnonzero(np.isin(sources, val_sources))
    test = np.flatnonzero(np.isin(sources, test_sources))
    return train, val, test


def select_sources_per_class(indices, labels, sources, count, seed):
    """Keep a fixed number of source files per class from a training fold."""
    rng = np.random.default_rng(seed)
    selected_sources = []
    for label in sorted(np.unique(labels[indices])):
        class_sources = np.unique(sources[indices][labels[indices] == label])
        rng.shuffle(class_sources)
        if len(class_sources) < count:
            raise ValueError(
                f"requested {count} training sources for class {label}, "
                f"but only {len(class_sources)} are available"
            )
        selected_sources.extend(class_sources[:count])
    return indices[np.isin(sources[indices], selected_sources)]


def minmax(path, indices, chunk=256):
    low, high = np.inf, -np.inf
    with h5py.File(path, "r") as f:
        for begin in range(0, len(indices), chunk):
            for idx in indices[begin:begin + chunk]:
                a = f["x"][idx]
                low, high = min(low, float(a.min())), max(high, float(a.max()))
    return low, high


def evaluate(model, loader, device):
    model.eval(); true, pred = [], []
    with torch.no_grad():
        for x, y in loader:
            true.extend(y.numpy()); pred.extend(model(x.to(device)).argmax(1).cpu().numpy())
    return dict(accuracy=accuracy_score(true, pred), macro_f1=f1_score(true, pred, average="macro", zero_division=0),
                macro_recall=recall_score(true, pred, average="macro", zero_division=0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", choices=["lenet", "resnet18"], default="lenet")
    p.add_argument("--split", choices=["random-window", "source-trace", "participant", "external"], default="random-window")
    p.add_argument("--test-data", type=Path,
                   help="separate HDF5 used entirely for testing with --split external")
    p.add_argument("--augment-data", type=Path,
                   help="synthetic HDF5 added only to the real training split")
    p.add_argument("--augment-ratio", type=float, default=1.0,
                   help="synthetic/real-train sample ratio (default: 1.0, i.e. +100%%)")
    p.add_argument("--real-train-ratio", type=float, default=1.0,
                   help="fraction of the original real training split to use (default: 1.0)")
    p.add_argument("--train-sources-per-class", type=int,
                   help="with --split source-trace, retain exactly this many real training sources per class")
    p.add_argument("--test-participant", type=int); p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64); p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=8); p.add_argument("--seed", type=int, default=111)
    args = p.parse_args()
    if args.split == "participant" and args.test_participant is None: p.error("participant split requires --test-participant")
    if args.split == "external" and args.test_data is None: p.error("external split requires --test-data")
    if args.augment_ratio <= 0: p.error("--augment-ratio must be positive")
    if not 0 < args.real_train_ratio <= 1:
        p.error("--real-train-ratio must be in (0, 1]")
    if args.train_sources_per_class is not None:
        if args.split != "source-trace":
            p.error("--train-sources-per-class requires --split source-trace")
        if args.train_sources_per_class <= 0:
            p.error("--train-sources-per-class must be positive")
        if args.real_train_ratio != 1.0:
            p.error("do not combine --train-sources-per-class with --real-train-ratio")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    with h5py.File(args.data, "r") as f:
        labels, participants = f["y"][:], f["participant"][:]
        input_shape = tuple(f["x"].shape[1:])
        if args.split == "source-trace":
            if "source" not in f:
                raise ValueError("source-trace split requires a 'source' dataset in the input HDF5")
            sources = np.array([
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in f["source"][:]
            ])
    if args.split == "external":
        all_idx = np.arange(len(labels))
        train, val = train_test_split(all_idx, test_size=.1, random_state=args.seed, stratify=labels)
        with h5py.File(args.test_data, "r") as f:
            test_labels = f["y"][:]
            test_shape = tuple(f["x"].shape[1:])
        if test_shape != input_shape:
            raise ValueError(f"train input shape {input_shape} differs from external test shape {test_shape}")
        if set(np.unique(test_labels)) - set(np.unique(labels)):
            raise ValueError("external test data contains labels absent from training data")
        test = np.arange(len(test_labels))
    elif args.split == "source-trace":
        train, val, test = split_source_traces(labels, sources, args.seed)
    else:
        train, val, test = split_indices(labels, participants, args.split, args.test_participant, args.seed)
    available_real_train_samples = len(train)
    available_real_train_sources = (
        len(np.unique(sources[train])) if args.split == "source-trace" else None
    )
    if args.train_sources_per_class is not None:
        train = select_sources_per_class(
            train, labels, sources, args.train_sources_per_class, args.seed
        )
    elif args.real_train_ratio < 1.0:
        train, _ = train_test_split(
            train, train_size=args.real_train_ratio, random_state=args.seed,
            stratify=labels[train]
        )
    low, high = minmax(args.data, train)
    paths = {"train": args.data, "val": args.data,
             "test": args.test_data if args.split == "external" else args.data}
    real_train_dataset = H5Dataset(args.data, train, low, high)
    synthetic_samples = 0
    available_synthetic_samples = 0
    if args.augment_data is not None:
        with h5py.File(args.augment_data, "r") as f:
            augment_shape = tuple(f["x"].shape[1:])
            augment_labels = f["y"][:]
            available_synthetic_samples = len(augment_labels)
        if augment_shape != input_shape:
            raise ValueError(f"real input shape {input_shape} differs from augmentation shape {augment_shape}")
        if set(np.unique(augment_labels)) - set(np.unique(labels)):
            raise ValueError("augmentation data contains labels absent from real training data")
        requested = int(round(len(train) * args.augment_ratio))
        if requested > available_synthetic_samples:
            raise ValueError(
                f"augmentation ratio {args.augment_ratio} requests {requested} samples, "
                f"but only {available_synthetic_samples} are available"
            )
        all_synthetic = np.arange(available_synthetic_samples)
        if requested < available_synthetic_samples:
            synthetic_indices, _ = train_test_split(
                all_synthetic, train_size=requested, random_state=args.seed,
                stratify=augment_labels
            )
        else:
            synthetic_indices = all_synthetic
        synthetic_samples = len(synthetic_indices)
        synthetic_dataset = H5Dataset(args.augment_data, synthetic_indices, low, high)
        train_dataset = ConcatDataset([real_train_dataset, synthetic_dataset])
    else:
        train_dataset = real_train_dataset
    loaders = {
        "train": DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=2, pin_memory=True),
        "val": DataLoader(H5Dataset(args.data, val, low, high), batch_size=args.batch_size,
                          shuffle=False, num_workers=2, pin_memory=True),
        "test": DataLoader(H5Dataset(paths["test"], test, low, high), batch_size=args.batch_size,
                           shuffle=False, num_workers=2, pin_memory=True),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (SenseFiLeNet() if args.model == "lenet" else SenseFiResNet18()).to(device)
    optimizer, criterion = torch.optim.Adam(model.parameters(), lr=args.lr), nn.CrossEntropyLoss()
    args.output_dir.mkdir(parents=True, exist_ok=True); best, stale, history = -1, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for x, y in loaders["train"]:
            optimizer.zero_grad(); loss = criterion(model(x.to(device)), y.to(device)); loss.backward(); optimizer.step(); losses.append(loss.item())
        val_metrics = evaluate(model, loaders["val"], device)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_metrics.items()}})
        print(history[-1])
        if val_metrics["accuracy"] > best:
            best, stale = val_metrics["accuracy"], 0; torch.save(model.state_dict(), args.output_dir / "best.pt")
        else:
            stale += 1
            if stale >= args.patience: break
    model.load_state_dict(torch.load(args.output_dir / "best.pt", map_location=device))
    model_source = ("official SenseFi UT-HAR LeNet architecture; output 7->20; "
                    "adaptive 4x4 pooling for 242-subcarrier input"
                    if args.model == "lenet" else
                    "official SenseFi UT-HAR ResNet18 architecture; output 7->20")
    result = {"model": args.model, "model_source": model_source,
              "split": args.split, "test_participant": args.test_participant,
              "train_data": str(args.data), "test_data": str(paths["test"]),
              "augmentation_data": str(args.augment_data) if args.augment_data else None,
              "augmentation_ratio": args.augment_ratio if args.augment_data else 0.0,
              "real_train_ratio": args.real_train_ratio,
              "train_sources_per_class": args.train_sources_per_class,
              "available_real_train_samples": available_real_train_samples,
              "available_synthetic_samples": available_synthetic_samples,
              "real_train_samples": len(train), "synthetic_train_samples": synthetic_samples,
              "train_samples": len(train) + synthetic_samples,
              "validation_samples": len(val), "test_samples": len(test),
              "normalization": "train-global-minmax", "seed": args.seed, **evaluate(model, loaders["test"], device)}
    if args.split == "source-trace":
        result.update(
            train_source_traces=len(np.unique(sources[train])),
            available_train_source_traces=available_real_train_sources,
            validation_source_traces=len(np.unique(sources[val])),
            test_source_traces=len(np.unique(sources[test])),
            source_trace_overlap=False,
        )
    with open(args.output_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0]); writer.writeheader(); writer.writerows(history)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
