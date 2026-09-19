from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from .config import BATCH_SIZE, DATA_DIR, SEED, VAL_RATIO
import torch


def make_loaders(batch_size=BATCH_SIZE):
    # Convert uint8 [0,255] to float32 [0,1], then use a fixed scaling to [-1,1].
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_full = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=transform)
    test = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=transform)
    val_size = int(len(train_full) * VAL_RATIO)
    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(SEED)
    train, val = random_split(train_full, [train_size, val_size], generator=generator)
    kwargs = dict(batch_size=batch_size, num_workers=0, pin_memory=torch.cuda.is_available())
    return (DataLoader(train, shuffle=True, generator=torch.Generator().manual_seed(SEED), **kwargs), DataLoader(val, shuffle=False, **kwargs),
            DataLoader(test, shuffle=False, **kwargs))
