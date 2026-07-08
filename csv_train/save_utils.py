"""
模型保存、结果导出、汇总打印
"""

import os
import pickle
import json
import numpy as np

from config import CLASS_NAMES, MODEL_DIR


def save_best(results, trained, feature_cols, dataset_meta):
    best_name = max(results, key=lambda n: results[n]["f1_macro"])
    res        = results[best_name]
    model, scaler = trained[best_name]

    payload = {
        "model_name":   best_name,
        "model":        model,
        "scaler":       scaler,
        "feature_cols": feature_cols,
        "dataset_meta": dataset_meta,
        "metrics": {k: v for k, v in res.items() if k not in ("y_pred", "y_prob")},
    }
    path = os.path.join(MODEL_DIR, "best_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(payload, f)

    auc_s = f"{res['auc_macro']:.4f}" if res["auc_macro"] else "N/A"
    print(f"\n  [BEST] 最优模型: {best_name}")
    print(f"     准确率={res['accuracy']:.4f}  宏F1={res['f1_macro']:.4f}  "
          f"加权F1={res['f1_weighted']:.4f}  宏AUC={auc_s}")
    print(f"     已保存 -> {path}")
    return best_name


def save_results_json(results, best_name, feature_cols, dataset_meta):
    def _cvt(v):
        if isinstance(v, np.ndarray):  return v.tolist()
        if isinstance(v, np.floating): return float(v)
        if isinstance(v, np.integer):  return int(v)
        return v
    data = {
        "best_model": best_name, "feature_cols": feature_cols,
        "class_names": CLASS_NAMES,
        "dataset_meta": dataset_meta,
        "evaluation_note": (
            "当前评估目标为基于流行病学规则生成的风险代理标签，"
            "结果更适合用于方法验证和风险分级，不宜直接等同于真实田间发病率预测。"
        ),
        "models": {name: {k: _cvt(v) for k, v in res.items()
                          if k not in ("y_pred", "y_prob")}
                   for name, res in results.items()},
    }
    path = os.path.join(MODEL_DIR, "results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"     结果 JSON -> {path}")


def print_summary(results, best_name, dataset_meta):
    hdr = f"{'模型':<10} {'准确率':>8} {'宏F1':>8} {'加权F1':>8} {'宏精确率':>9} {'宏召回率':>9} {'宏AUC':>8} {'耗时':>7}"
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for name, res in results.items():
        auc_s = f"{res['auc_macro']:.4f}" if res["auc_macro"] else "  N/A "
        mark  = " *" if name == best_name else ""
        print(f"{name:<10} {res['accuracy']:>8.4f} {res['f1_macro']:>8.4f} "
              f"{res['f1_weighted']:>8.4f} {res['precision_macro']:>9.4f} "
              f"{res['recall_macro']:>9.4f} {auc_s:>8} {res['time']:>6.1f}s{mark}")
    print("=" * len(hdr))
    print("* 标注为按宏F1选出的最优模型\n")
    if dataset_meta:
        print(f"标签说明: {dataset_meta.get('label_note', '未提供')}")
        print(f"拆分方式: {dataset_meta.get('split_strategy', '未提供')}\n")
