from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torchvision.transforms.v2 import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToDtype,
    ToImage,
)


# =============================
# 公共的通用数据集入口
# =============================
def get_image_dataset(
    dataset_name: str,
    root: str | Path = Path(__file__).parents[2] / "data",
    train: bool = True,
    transform: Callable | None = None,
    download: bool = True,
) -> Dataset:
    dataset_name = dataset_name.lower()
    root = Path(root).expanduser()

    if dataset_name == "mnist":
        return MNIST(root=root, train=train, transform=transform, download=download)

    elif dataset_name == "fashion_mnist":
        return FashionMNIST(
            root=root,
            train=train,
            transform=transform,
            download=download,
        )

    elif dataset_name == "cifar10":
        return CIFAR10(root, train, transform, download=download)

    elif dataset_name in ("div2k", "div2k_hr", "div2k-hr"):
        if transform is None:
            transform = get_div2k_train_transform() if train else get_div2k_test_transform()
        return DIV2KDataset(root=root, train=train, transform=transform)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# =============================
# 通用 Transform（原有）
# =============================
def get_train_transform(horizontal_flip: bool = False, normalize: bool = True) -> Callable:
    transform_list = [
        ToImage(),  # convert to torchvision.tv_tensors.Image
        ToDtype(torch.float32, scale=True),  # scale to [0, 1]
    ]
    if horizontal_flip:
        transform_list.append(RandomHorizontalFlip())
    if normalize:
        transform_list.append(Normalize((0.5,), (0.5,)))  # normalize to [-1, 1]
    return Compose(transform_list)


def get_test_transform(normalize: bool = True) -> Callable:
    transform_list = [
        ToImage(),  # convert to torchvision.tv_tensors.Image
        ToDtype(torch.float32, scale=True),  # scale to [0, 1]
    ]
    if normalize:
        transform_list.append(Normalize((0.5,), (0.5,)))  # normalize to [-1, 1]
    return Compose(transform_list)


# =============================
# DIV2K 专用 Transform（256, 256）
# =============================
def get_div2k_train_transform(
    size: tuple[int, int] = (256, 256),
    horizontal_flip: bool = False,
    normalize: bool = True,
) -> Callable:
    """
    训练用：Resize(短边=size[0]) + CenterCrop(size[1])，RGB 归一化到 [-1, 1]
    """
    H, W = size
    ops = [
        ToImage(),
        Resize(H),            # 短边缩放到 H，保持长宽比
        CenterCrop(W),        # 中心裁剪为 W×W（此处 H=W=256 时就是 256, 256）
        ToDtype(torch.float32, scale=True),  # [0,1]
    ]
    if horizontal_flip:
        ops.append(RandomHorizontalFlip())
    if normalize:
        ops.append(Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))  # [-1,1]
    return Compose(ops)


def get_div2k_test_transform(
    size: tuple[int, int] = (256, 256),
    normalize: bool = True,
) -> Callable:
    H, W = size
    ops = [
        ToImage(),
        Resize(H),
        CenterCrop(W),
        ToDtype(torch.float32, scale=True),
    ]
    if normalize:
        ops.append(Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    return Compose(ops)


# =============================
# DIV2K 自定义数据集
# =============================
class DIV2KDataset(Dataset):
    """
    单类别 DIV2K 高分辨率数据集封装。
    目录结构（根据 train 标志切换）：
        <root>/DIV2K_train_HR/*.png|jpg|jpeg
        <root>/DIV2K_valid_HR/*.png|jpg|jpeg
    返回 (img_tensor, label=0)，并暴露 .classes=["div2k"]
    """
    def __init__(self, root: str | Path, train: bool, transform: Callable | None = None):
        self.root = Path(root).expanduser()
        sub = "DIV2K_train_HR" if train else "DIV2K_valid_HR"
        self.dir = self.root / sub

        exts = (".png", ".jpg", ".jpeg")
        files: list[Path] = []
        for ext in exts:
            files.extend(self.dir.glob(f"*{ext}"))
        self.files = sorted(files)
        if not self.files:
            raise FileNotFoundError(
                f"[DIV2K] No images found under: {self.dir}. "
                "Pass --data_root (or set LTT_DATA_ROOT) to a directory containing "
                "DIV2K_train_HR/ and DIV2K_valid_HR/."
            )

        self.transform = transform
        self.classes = ["div2k"]  # 与上层代码兼容（num_classes = 1）

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        # 单类别标签
        label = 0
        return img, label
