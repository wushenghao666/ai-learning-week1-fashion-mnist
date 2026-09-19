from pathlib import Path
import json

root = Path(__file__).resolve().parents[1]

def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)}

def code(text):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': text.splitlines(keepends=True)}

cells = [
md('# 第 1 周：从张量到服饰分类\n\n这是配合 src/ 的学习笔记。先运行完整训练，再从上到下执行本笔记；小批量演示使用独立模型，不会覆盖已保存模型。'),
md('## 1. 准备环境\n\n在项目虚拟环境中运行 Jupyter。这里兼容从项目根目录或 notebooks 目录启动。'),
code("from pathlib import Path\nimport sys\nROOT = Path.cwd().resolve()\nif not (ROOT / 'src').is_dir():\n    ROOT = ROOT.parent\nassert (ROOT / 'src').is_dir(), '请从项目根目录或 notebooks 目录启动'\nsys.path.insert(0, str(ROOT))\nimport torch\nfrom src.train import seed_everything\nseed_everything()\ntorch.set_num_threads(4)\nprint('PyTorch:', torch.__version__)"),
md('## 2. 张量的形状和基本操作\n\n张量可以理解为带形状和类型的多维数组。`reshape` 改变形状；`sum` 求和；`@` 是矩阵乘法，`*` 是逐元素乘法。'),
code("x = torch.arange(6, dtype=torch.float32).reshape(2, 3)\nprint('x =', x)\nprint('shape / dtype:', x.shape, x.dtype)\nprint('第一行:', x[0])\nprint('逐元素乘 2:', x * 2)\nprint('每行之和:', x.sum(dim=1))\nprint('矩阵乘法:', x @ x.T)"),
md('## 3. Dataset 和 DataLoader\n\nDataset 按索引返回一张图片和标签；DataLoader 打乱、分批返回数据。训练和验证来自固定种子划分的原始训练集，测试使用官方独立测试集。'),
code("from src.dataset import make_loaders\ntrain_loader, val_loader, test_loader = make_loaders()\nimages, labels = next(iter(train_loader))\nprint('样本数:', len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))\nprint('图片:', images.shape, images.dtype)\nprint('标签:', labels.shape, labels.dtype)\nprint('展平后:', images.flatten(1).shape)\nassert images.shape == (128, 1, 28, 28)\nassert set(train_loader.dataset.indices).isdisjoint(val_loader.dataset.indices)"),
md('## 4. 一个 batch 的前向、反向和更新\n\n模型输出 `[128,10]` 的原始分数 logits，不预先做 softmax。交叉熵内部会处理这些分数。`backward()` 产生梯度，`step()` 才改变参数。'),
code("from src.model import MLP\nmodel = MLP()\noptimizer = torch.optim.Adam(model.parameters(), lr=0.001)\ncriterion = torch.nn.CrossEntropyLoss()\nweight = next(model.parameters())\nbefore = weight.detach().clone()\noptimizer.zero_grad()\nlogits = model(images)\nloss = criterion(logits, labels)\nloss.backward()\nprint('logits:', logits.shape, 'loss:', loss.item())\nprint('梯度范数:', weight.grad.norm().item())\nassert torch.equal(before, weight)\noptimizer.step()\nprint('参数变化总量:', (weight - before).abs().sum().item())\nassert not torch.equal(before, weight)"),
md('## 5. 检查真实实验\n\n先在终端运行 `python -m src.train`。以下只读取指标和最佳模型，不重新训练，也不使用测试集调参。'),
code("import json\nmetrics = json.loads((ROOT / 'outputs/mlp/metrics.json').read_text(encoding='utf-8'))\nfor key in ['model', 'epochs', 'train_size', 'val_size', 'test_size', 'best_epoch', 'test_loss', 'test_accuracy']:\n    print(key, ':', metrics[key])\ncheckpoint = torch.load(ROOT / 'outputs/mlp/model_best.pt', map_location='cpu', weights_only=True)\ntrained = MLP()\ntrained.load_state_dict(checkpoint['model_state'])\ntrained.eval()\nwith torch.no_grad():\n    predicted = trained(images).argmax(1)\nprint('一批训练图片的前 10 个预测:', predicted[:10].tolist())"),
code("from IPython.display import Image, display\ndisplay(Image(filename=str(ROOT / 'outputs/mlp/figures/training_curves.png')))\ndisplay(Image(filename=str(ROOT / 'outputs/mlp/predictions/correct.png')))\ndisplay(Image(filename=str(ROOT / 'outputs/mlp/predictions/incorrect.png')))"),
md('## 6. 下一步\n\n对照 README 和博客解释曲线。尝试先增加 epoch，再单独比较 CNN；一次只改变一个条件，用验证集选择设置。错误图只是测试顺序中的前 8 个错误，不能代表全部错误分布。')]
for i, cell in enumerate(cells):
    cell['id'] = f'week1-{i:02d}'
notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.13'}}, 'nbformat': 4, 'nbformat_minor': 5}
(root / 'notebooks/01_fashion_mnist.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
print('Notebook created')
