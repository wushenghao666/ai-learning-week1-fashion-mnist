"""Evaluation and plots: no gradients, no parameter updates."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


@torch.no_grad()
def evaluate(model, loader, criterion, device, collect_samples=False):
    model.eval()  # Disables dropout; no_grad separately disables gradient recording.
    total_loss = 0.0
    correct = total = 0
    samples = {'correct': [], 'incorrect': []}
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        if collect_samples:
            for image, label, pred in zip(images.cpu(), labels.cpu(), preds.cpu()):
                label, pred = int(label), int(pred)
                group = 'correct' if label == pred else 'incorrect'
                if len(samples[group]) < 8:
                    samples[group].append((image, label, pred))
    return total_loss / total, correct / total, samples


def save_curves(history, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in zip(axes, ['loss', 'acc'], ['Cross-entropy loss', 'Accuracy']):
        ax.plot(epochs, history[f'train_{metric}'], 'o-', label='Train (during updates)')
        ax.plot(epochs, history[f'val_{metric}'], 's--', label='Validation')
        ax.set(xlabel='Epoch', ylabel=title, title=title, xticks=list(epochs))
        ax.grid(alpha=0.2)
        ax.legend()
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_samples(samples, path, title):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(10, 6))
    for ax in axes.flat:
        ax.axis('off')
    for ax, (image, label, pred) in zip(axes.flat, samples):
        ax.imshow(image.squeeze() * 0.5 + 0.5, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'True: {CLASSES[label]}\nPred: {CLASSES[pred]}', fontsize=10)
    fig.suptitle(title if samples else f'{title}: no samples found')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    from .config import OUTPUT_DIR
    from .dataset import make_loaders
    from .model import MLP, SmallCNN

    parser = argparse.ArgumentParser(description='Re-evaluate a saved checkpoint')
    parser.add_argument('--checkpoint', type=Path, default=OUTPUT_DIR / 'mlp' / 'model_best.pt')
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = MLP() if checkpoint['model'] == 'mlp' else SmallCNN()
    model.load_state_dict(checkpoint['model_state'])
    _, _, test_loader = make_loaders()
    loss, accuracy, _ = evaluate(model, test_loader, torch.nn.CrossEntropyLoss(), 'cpu')
    print(json.dumps({'test_loss': loss, 'test_accuracy': accuracy}, indent=2))


if __name__ == '__main__':
    main()
