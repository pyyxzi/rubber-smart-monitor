import copy
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def load_data(config):
    """加载数据集，进行数据增强并划分训练/验证集"""
    # 训练集数据增强
    train_transform = transforms.Compose([
        transforms.Resize((config["img_size"], config["img_size"])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomResizedCrop(config["img_size"], scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # 验证集仅做标准化
    val_transform = transforms.Compose([
        transforms.Resize((config["img_size"], config["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(root=config["data_dir"])
    class_names = full_dataset.classes
    total = len(full_dataset)
    train_size = int(total * config["train_ratio"])
    val_size = total - train_size
    print(f"类别: {class_names} | 总数: {total} | 训练: {train_size} | 验证: {val_size}")

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config["seed"])
    )
    # 为训练集和验证集分别设置不同的transform
    train_dataset.dataset = copy.copy(full_dataset)
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset = copy.copy(full_dataset)
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                              shuffle=True, num_workers=config["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"],
                            shuffle=False, num_workers=config["num_workers"])
    return train_loader, val_loader, class_names
