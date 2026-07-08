import os
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

CONFIG = {
    "data_dir": os.path.join(PROJECT_ROOT, "data_pic"),
    "output_dir": os.path.join(os.path.dirname(__file__), "output"),
    "img_size": 224,
    "batch_size": 32,
    "num_epochs": 50,
    "learning_rate": 1e-4,
    "train_ratio": 0.8,
    "patience": 3,
    "num_workers": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
