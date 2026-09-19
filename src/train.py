"""Run from project root: python -m src.train --model mlp --epochs 3."""
import argparse
import json
import platform
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision

from .config import BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_NAME, OUTPUT_DIR, SEED
from .dataset import make_loaders
from .model import MLP, SmallCNN
from .evaluate import evaluate, save_curves, save_samples


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()          # Clear gradients from the previous batch.
        logits = model(images)         # Forward: [batch, 1, 28, 28] -> [batch, 10].
        loss = criterion(logits, labels)  # CrossEntropyLoss expects raw logits.
        loss.backward()               # Compute gradients, but do not change weights.
        optimizer.step()              # Update weights using those gradients.
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description='Fashion-MNIST learning experiment')
    parser.add_argument('--model', choices=['mlp', 'cnn'], default=MODEL_NAME)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--limit-train', type=int, default=0, help='0 uses all 54000 training examples')
    parser.add_argument('--threads', type=int, default=4, help='CPU threads; avoids excessive overhead')
    args = parser.parse_args()
    if args.epochs < 1 or args.limit_train < 0 or args.threads < 1:
        parser.error('epochs and threads must be positive; limit-train must be nonnegative')
    seed_everything()
    torch.set_num_threads(args.threads)
    run_dir = OUTPUT_DIR / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}, model={args.model}', flush=True)
    train_loader, val_loader, test_loader = make_loaders()
    if args.limit_train:
        train_loader.dataset.indices = train_loader.dataset.indices[:args.limit_train]
    print(f'split: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}', flush=True)
    model = (MLP() if args.model == 'mlp' else SmallCNN()).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = {key: [] for key in ['train_loss', 'train_acc', 'val_loss', 'val_acc']}
    initial_loss, initial_acc, _ = evaluate(model, val_loader, criterion, device)
    best_val = -1
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device)
        for key, value in zip(history, [train_loss, train_acc, val_loss, val_acc]):
            history[key].append(value)
        print(f'epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}', flush=True)
        if val_acc > best_val:
            best_val = val_acc
            torch.save({'model_state': model.state_dict(), 'model': args.model,
                        'epoch': epoch, 'seed': SEED}, run_dir / 'model_best.pt')
    # Test only after model selection, using the saved best model.
    checkpoint = torch.load(run_dir / 'model_best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state'])
    test_loss, test_acc, samples = evaluate(model, test_loader, criterion, device, collect_samples=True)
    elapsed = time.perf_counter() - start
    print(f"best_epoch={checkpoint['epoch']} test_loss={test_loss:.4f} test_acc={test_acc:.4f}", flush=True)
    save_curves(history, run_dir / 'figures' / 'training_curves.png')
    for group in ['correct', 'incorrect']:
        save_samples(samples[group], run_dir / 'predictions' / f'{group}.png', f'First {len(samples[group])} {group} test predictions')
    result = {
        'model': args.model, 'epochs': args.epochs, 'best_epoch': checkpoint['epoch'],
        'seed': SEED, 'device': str(device), 'threads': args.threads,
        'batch_size': BATCH_SIZE, 'learning_rate': LEARNING_RATE, 'optimizer': 'Adam',
        'train_size': len(train_loader.dataset), 'val_size': len(val_loader.dataset),
        'test_size': len(test_loader.dataset), 'initial_val_loss': initial_loss,
        'initial_val_accuracy': initial_acc, 'test_loss': test_loss,
        'test_accuracy': test_acc, 'history': history, 'elapsed_seconds': elapsed,
        'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                     'torchvision': torchvision.__version__, 'numpy': np.__version__},
        'samples': {group: [{'true': label, 'pred': pred} for _, label, pred in values]
                    for group, values in samples.items()},
    }
    (run_dir / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'Saved outputs to {run_dir}', flush=True)


if __name__ == '__main__':
    main()
