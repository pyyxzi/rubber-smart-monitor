import os
import copy
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .config import CONFIG
    from .dataset import load_data
    from .model import build_model
    from .visualize import plot_training_curves
    from .evaluate import evaluate_model
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from config import CONFIG
    from dataset import load_data
    from model import build_model
    from visualize import plot_training_curves
    from evaluate import evaluate_model


class EarlyStopping:
    """早停机制：验证集损失连续多轮不下降时停止训练"""

    def __init__(self, patience=3, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"  EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_model(model, train_loader, val_loader, config):
    """训练模型，包含早停机制"""
    device = config["device"]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=config["learning_rate"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=config["patience"])

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\n开始训练 | 设备: {device} | 共 {config['num_epochs']} 轮\n")

    for epoch in range(config["num_epochs"]):
        start_time = time.time()

        model.train()
        running_loss, running_corrects, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum().item()
            total += inputs.size(0)
        train_loss = running_loss / total
        train_acc = running_corrects / total

        model.eval()
        val_loss, val_corrects, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += (preds == labels).sum().item()
                val_total += inputs.size(0)
        val_loss = val_loss / val_total
        val_acc = val_corrects / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1:02d}/{config['num_epochs']}]  "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  ({elapsed:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"  * 最佳模型已更新 Val Acc: {best_val_acc:.4f}")

        scheduler.step(val_loss)
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\n早停触发，在第 {epoch+1} 轮停止训练")
            break

    model.load_state_dict(best_model_wts)
    print(f"\n训练完成，最佳验证准确率: {best_val_acc:.4f}")
    return model, history


def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    print("[1/5] 加载数据集...")
    train_loader, val_loader, class_names = load_data(CONFIG)

    print("[2/5] 构建模型...")
    model = build_model(num_classes=len(class_names), device=CONFIG["device"])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数: {total_params:,} | 可训练: {trainable:,}")

    print("[3/5] 训练模型...")
    model, history = train_model(model, train_loader, val_loader, CONFIG)

    print("[4/5] 生成可视化图表...")
    plot_training_curves(history, CONFIG["output_dir"])

    print("[5/5] 评估模型...")
    evaluate_model(model, val_loader, class_names, CONFIG)

    # 保存最佳模型
    model_path = os.path.join(CONFIG["output_dir"], "best_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "config": CONFIG,
    }, model_path)
    print(f"\n模型已保存: {model_path}")


if __name__ == "__main__":
    main()
