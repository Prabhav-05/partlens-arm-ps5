"""Train, evaluate and export the five-class metal nut inspection model."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

SIZE = 192
MASK_SIZE = 48
LABELS = ("good", "scratch", "bent", "color", "flip")
SEED = 42


def image_files(folder):
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})


def make_splits(root):
    rng = random.Random(SEED)
    splits = {"train": [], "val": [], "test": []}
    good_train = image_files(root / "train/good")
    good_test = image_files(root / "test/good")
    assert len(good_train) == 220 and len(good_test) == 22, (len(good_train), len(good_test))
    rng.shuffle(good_test)
    for split, images in (("train", good_train), ("val", good_test[:11]), ("test", good_test[11:])):
        splits[split].extend((image, 0, None) for image in images)
    defect_count = 0
    for label, name in enumerate(LABELS[1:], 1):
        images = image_files(root / "test" / name)
        defect_count += len(images)
        rng.shuffle(images)
        train_end = round(len(images) * 0.60)
        val_end = train_end + round(len(images) * 0.20)
        for split, subset in (("train", images[:train_end]), ("val", images[train_end:val_end]), ("test", images[val_end:])):
            for image in subset:
                mask = root / "ground_truth" / name / f"{image.stem}_mask.png"
                if not mask.is_file():
                    raise FileNotFoundError(mask)
                splits[split].append((image, label, mask))
    assert defect_count == 93, defect_count
    return splits


class MetalNuts(Dataset):
    def __init__(self, records, augment=False):
        self.augment = augment
        self.images = []
        for path, label, mask_path in records:
            with Image.open(path) as image:
                rgb = image.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR)
                pixels = torch.from_numpy(np.array(rgb)).permute(2, 0, 1)
            mask = torch.zeros((1, MASK_SIZE, MASK_SIZE), dtype=torch.float32)
            if mask_path is not None:
                with Image.open(mask_path) as image:
                    resized = image.convert("L").resize((MASK_SIZE, MASK_SIZE), Image.Resampling.BILINEAR)
                    mask = torch.from_numpy(np.array(resized) > 24).unsqueeze(0).float()
            self.images.append((pixels, label, mask))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        pixels, label, mask = self.images[index]
        pixels = pixels.float().div(255)
        if self.augment:
            turns = random.randrange(4)
            pixels, mask = pixels.rot90(turns, (-2, -1)), mask.rot90(turns, (-2, -1))
            if random.random() < 0.5:
                pixels, mask = pixels.flip((-1,)), mask.flip((-1,))
            pixels = (pixels * random.uniform(0.92, 1.08) + random.uniform(-0.025, 0.025)).clamp(0, 1)
        return pixels, label, mask


def block(input_channels, output_channels, stride):
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 3, stride, padding=1, bias=False),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True),
    )


class InspectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.detail = nn.Sequential(block(3, 16, 2), block(16, 24, 2), block(24, 32, 1))
        self.context = nn.Sequential(block(32, 48, 2), block(48, 64, 2), block(64, 96, 2))
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(384, len(LABELS)))
        self.mask = nn.Sequential(block(128, 24, 1), nn.Conv2d(24, 1, 1))

    def forward(self, pixels):
        detail = self.detail(pixels)
        context = self.context(detail)
        upscaled = F.interpolate(context, size=(MASK_SIZE, MASK_SIZE), mode="nearest")
        return self.classifier(context), self.mask(torch.cat((detail, upscaled), dim=1))


def loss_function(outputs, labels, masks):
    logits, heat = outputs
    targets = heat.sigmoid()
    intersection = (targets * masks).sum((-2, -1))
    dice = (2 * intersection + 1) / (targets.sum((-2, -1)) + masks.sum((-2, -1)) + 1)
    pixels = F.binary_cross_entropy_with_logits(heat, masks, reduction="none")
    pixels = (pixels * (1 + 5 * masks)).mean()
    return F.cross_entropy(logits, labels, label_smoothing=0.03) + 0.85 * (pixels + (1 - dice).mean())


@torch.inference_mode()
def predict(model, loader, device):
    result = []
    model.eval()
    for images, labels, masks in loader:
        logits, heat = model(images.to(device))
        for actual, truth, scores, mask in zip(labels, masks, logits.softmax(1).cpu(), heat.sigmoid().cpu()):
            confidence, predicted = scores.max(0)
            result.append((actual.item(), predicted.item(), confidence.item(), truth.bool(), mask))
    return result


def mask_fraction(mask, threshold):
    components, count = ndimage.label((mask >= threshold).squeeze(0).numpy())
    if not count:
        return 0.0
    pixels = np.bincount(components.ravel())[1:].max()
    return float(pixels / mask.numel()) if pixels >= 3 else 0.0


def summarize(results, class_threshold, mask_threshold, anomaly_threshold=1.01):
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    ious, reviews, false_passes, normal_reviews = [], 0, 0, 0
    for actual, predicted, confidence, truth, mask in results:
        matrix[actual, predicted] += 1
        if actual != 0:
            positive = mask >= mask_threshold
            ious.append((truth & positive).sum().item() / max(1, (truth | positive).sum().item()))
        review = confidence < class_threshold or (
            predicted == 0 and anomaly_threshold <= 1 and mask_fraction(mask, mask_threshold) >= anomaly_threshold
        )
        reviews += review
        normal_reviews += actual == 0 and review
        false_passes += actual != 0 and predicted == 0 and not review
    true_positive = matrix[1:, 1:].sum()
    predicted_positive = matrix[:, 1:].sum()
    actual_positive = matrix[1:, :].sum()
    return {
        "count": int(matrix.sum()),
        "accuracy": float(matrix.trace() / matrix.sum()),
        "macro_recall": float(np.mean(matrix.diagonal() / np.maximum(1, matrix.sum(1)))),
        "defect_precision": float(true_positive / max(1, predicted_positive)),
        "defect_recall": float(true_positive / max(1, actual_positive)),
        "defect_mask_mean_iou": float(np.mean(ious)),
        "review_count": int(reviews),
        "normal_review_count": int(normal_reviews),
        "false_passes_above_threshold": int(false_passes),
        "confusion_matrix": matrix.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(".local/data/metal_nut"))
    parser.add_argument("--output", type=Path, default=Path(".local/vision"))
    parser.add_argument("--epochs", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, help="Calibrate and export an existing trained checkpoint")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)

    splits = make_splits(args.data)
    datasets = {name: MetalNuts(rows, augment=name == "train") for name, rows in splits.items()}
    counts = Counter(row[1] for row in splits["train"])
    weights = [1 / counts[row[1]] for row in splits["train"]]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    loaders = {
        name: DataLoader(dataset, batch_size=16, sampler=sampler if name == "train" else None)
        for name, dataset in datasets.items()
    }
    args.output.mkdir(parents=True, exist_ok=True)
    assets = args.output / "assets"
    assets.mkdir(exist_ok=True)
    model = InspectionModel().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best = -1.0
    checkpoint = args.output / "vision.pt"
    if args.checkpoint is None:
        for epoch in range(1, args.epochs + 1):
            model.train()
            total = 0.0
            for images, labels, masks in loaders["train"]:
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(model(images.to(args.device)), labels.to(args.device), masks.to(args.device))
                loss.backward()
                optimizer.step()
                total += loss.detach().item()
            scheduler.step()
            stats = summarize(predict(model, loaders["val"], args.device), 0.60, 0.45)
            score = stats["macro_recall"] + 0.35 * stats["defect_mask_mean_iou"]
            if score > best:
                best = score
                torch.save(model.state_dict(), checkpoint)
            print(f"epoch={epoch:02} loss={total/len(loaders['train']):.3f} "
                  f"val_macro_recall={stats['macro_recall']:.3f} val_mask_iou={stats['defect_mask_mean_iou']:.3f}", flush=True)

    model.load_state_dict(torch.load(args.checkpoint or checkpoint, map_location="cpu", weights_only=True))
    model.cpu().eval()
    validation = predict(model, loaders["val"], "cpu")
    mask_threshold = max((0.30, 0.40, 0.45, 0.50, 0.60), key=lambda t: summarize(validation, 0.60, t)["defect_mask_mean_iou"])
    choices = [
        (confidence, anomaly)
        for confidence in (0.55, 0.60, 0.65, 0.70, 0.75)
        for anomaly in (0.006, 0.012, 0.025, 0.05, 1.01)
    ]

    def policy_cost(choice):
        stats = summarize(validation, choice[0], mask_threshold, choice[1])
        return (5 * stats["false_passes_above_threshold"] + stats["review_count"],
                stats["normal_review_count"], -choice[1], choice[0])

    class_threshold, anomaly_threshold = min(choices, key=policy_cost)
    testing = predict(model, loaders["test"], "cpu")
    metrics = {
        "seed": SEED,
        "labels": LABELS,
        "class_threshold": class_threshold,
        "mask_threshold": mask_threshold,
        "anomaly_review_threshold": anomaly_threshold,
        "counts": {name: dict(Counter(LABELS[row[1]] for row in rows)) for name, rows in splits.items()},
        "validation": summarize(validation, class_threshold, mask_threshold, anomaly_threshold),
        "test": summarize(testing, class_threshold, mask_threshold, anomaly_threshold),
        "test_samples": [
            {"path": str(source.relative_to(args.data)), "actual": LABELS[actual], "prediction": LABELS[predicted],
             "confidence": confidence}
            for (source, _, _), (actual, predicted, confidence, _, _) in zip(splits["test"], testing)
        ],
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (assets / "vision.json").write_text(json.dumps({
        "class_threshold": class_threshold, "mask_threshold": mask_threshold,
        "anomaly_review_threshold": anomaly_threshold,
    }) + "\n")
    print(json.dumps(metrics["test"], indent=2), flush=True)

    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower
    exported = torch.export.export(model, (torch.zeros((1, 3, SIZE, SIZE)),))
    program = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()]).to_executorch()
    (assets / "vision.pte").write_bytes(program.buffer)
    print(f"Exported {assets / 'vision.pte'}", flush=True)


if __name__ == "__main__":
    main()
