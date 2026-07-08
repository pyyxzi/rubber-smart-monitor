"""
模型定义与训练评估
"""

import time
import numpy as np

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

from config import CLASS_NAMES


def get_models():
    return {
        "逻辑回归":  (LogisticRegression(max_iter=2000, class_weight="balanced",
                                          solver="lbfgs", random_state=42), True),
        "决策树":    (DecisionTreeClassifier(max_depth=12, class_weight="balanced",
                                             random_state=42), False),
        "随机森林":  (RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                             n_jobs=1, random_state=42), False),
        "梯度提升":  (GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                 max_depth=5, random_state=42), False),
        "KNN":       (KNeighborsClassifier(n_neighbors=7, n_jobs=1), True),
        "SVM":       (SVC(kernel="rbf", C=1.0, class_weight="balanced",
                          probability=True, random_state=42), True),
    }


def train_all(model_specs, X_train, y_train, X_test, y_test):
    scaler     = StandardScaler()
    X_train_s  = scaler.fit_transform(X_train)
    X_test_s   = scaler.transform(X_test)

    results = {}
    trained = {}

    for name, (model, needs_scale) in model_specs.items():
        print(f"\n  [RUN] [{name}]  训练集: {len(X_train):,} 条  特征: {X_train.shape[1]}", flush=True)
        t0  = time.time()
        Xtr = X_train_s if needs_scale else X_train.values
        Xte = X_test_s  if needs_scale else X_test.values

        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte) if hasattr(model, "predict_proba") else None
        dt     = time.time() - t0

        acc      = accuracy_score(y_test, y_pred)
        f1_mac   = f1_score(y_test, y_pred, average="macro",    zero_division=0)
        f1_wgt   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        prec_mac = precision_score(y_test, y_pred, average="macro",    zero_division=0)
        rec_mac  = recall_score(y_test, y_pred, average="macro",     zero_division=0)
        f1_per   = f1_score(y_test, y_pred, average=None, labels=[0, 1, 2, 3], zero_division=0)

        if y_prob is not None:
            y_bin     = label_binarize(y_test, classes=[0, 1, 2, 3])
            auc_score = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
        else:
            auc_score = None

        results[name] = dict(
            accuracy=acc, f1_macro=f1_mac, f1_weighted=f1_wgt,
            precision_macro=prec_mac, recall_macro=rec_mac,
            f1_per_class=f1_per, auc_macro=auc_score,
            y_pred=y_pred, y_prob=y_prob, time=dt,
        )
        trained[name] = (model, scaler if needs_scale else None)

        auc_s = f"{auc_score:.4f}" if auc_score else " N/A "
        print(f"     准确率={acc:.4f}  宏F1={f1_mac:.4f}  宏AUC={auc_s}  [{dt:.1f}s]")
        print(f"     各类F1: {CLASS_NAMES[0]}={f1_per[0]:.3f}  {CLASS_NAMES[1]}={f1_per[1]:.3f}  "
              f"{CLASS_NAMES[2]}={f1_per[2]:.3f}  {CLASS_NAMES[3]}={f1_per[3]:.3f}")

    return results, trained
