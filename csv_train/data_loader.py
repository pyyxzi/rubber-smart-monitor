"""
数据加载模块
"""

import os
import sys
import json
import pandas as pd

from config import DATA_DIR, META_PATH, CLASS_NAMES, LABEL_COL, DROP_COLS


def load_data():
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path  = os.path.join(DATA_DIR, "test.csv")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("[ERROR] 找不到处理后的数据，请先运行 data_csv/data_process.py")
        sys.exit(1)

    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    dataset_meta = {}
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            dataset_meta = json.load(f)

    feature_cols = [c for c in train.columns if c not in DROP_COLS]
    X_train = train[feature_cols].fillna(0)
    y_train = train[LABEL_COL]
    X_test  = test[feature_cols].fillna(0)
    y_test  = test[LABEL_COL]

    print(f"  训练集: {len(X_train):,} 条 | 测试集: {len(X_test):,} 条 | 特征数: {len(feature_cols)}")
    tr_dist = dict(y_train.value_counts().sort_index())
    te_dist = dict(y_test.value_counts().sort_index())
    print(f"  训练集标签分布: { {CLASS_NAMES[k]: v for k, v in tr_dist.items()} }")
    print(f"  测试集标签分布: { {CLASS_NAMES[k]: v for k, v in te_dist.items()} }")
    if dataset_meta:
        print(f"  时序拆分: {dataset_meta.get('split_strategy', '未提供')}")
        print(f"  标签说明: {dataset_meta.get('label_note', '未提供')}")
    return X_train, y_train, X_test, y_test, feature_cols, dataset_meta
