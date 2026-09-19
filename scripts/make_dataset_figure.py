from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import torch
from torchvision import datasets, transforms

root = Path(__file__).resolve().parents[1]
data = datasets.FashionMNIST(root / 'data', train=True, download=False, transform=transforms.ToTensor())
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
chosen = {}
for image, label in data:
    if int(label) not in chosen:
        chosen[int(label)] = image.squeeze(0)
    if len(chosen) == 10:
        break
font_path = Path(r'C:\Windows\Fonts\msyh.ttc')
cn_font = FontProperties(fname=str(font_path), size=13) if font_path.exists() else None
fig, axes = plt.subplots(2, 5, figsize=(11, 5))
for label, ax in zip(range(10), axes.flat):
    ax.imshow(chosen[label], cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'{label}: {classes[label]}', fontsize=10)
    ax.set_xticks([0, 14, 27]); ax.set_yticks([0, 14, 27])
    ax.grid(color='white', alpha=.25, linewidth=.4)
fig.suptitle('Fashion-MNIST：每个类别各展示一张 28×28 灰度图片', fontproperties=cn_font)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = root / 'outputs' / 'mlp' / 'figures' / 'fashion_mnist_samples.png'
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=160, bbox_inches='tight')
plt.close(fig)
print(out)
