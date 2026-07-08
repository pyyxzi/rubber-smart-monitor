"""
橡胶树白粉病预警 —— 多模型训练与评估
训练6个分类模型，用8类图表评估，保存最优模型到 models/best_model.pkl
运行: python csv_train/train_models.py

模块拆分:
  config.py      - 全局配置、路径、常量
  data_loader.py - 数据加载
  models.py      - 模型定义与训练评估
  plots.py       - 评估图表生成（图1~图9）
  save_utils.py  - 模型保存、JSON导出、汇总打印
"""

import os
import sys

# 确保 train/ 目录在 sys.path 中，以便模块间互相导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import FIG_DIR, MODEL_DIR
from data_loader import load_data
from models import get_models, train_all
from plots import (
    fig1_distribution, fig2_model_comparison, fig3_confusion_matrices,
    fig4_per_class_f1, fig5_roc_curves, fig6_feature_importance,
    fig7_radar, fig8_training_time, fig9_pred_vs_actual,
)
from save_utils import save_best, save_results_json, print_summary


if __name__ == "__main__":
    print("=" * 60)
    print("  橡胶树白粉病预警 - 模型训练与评估")
    print("=" * 60)

    print("\n[1/4] 加载数据 ...")
    X_train, y_train, X_test, y_test, feature_cols, dataset_meta = load_data()

    print("\n[2/4] 训练模型 ...")
    model_specs = get_models()
    results, trained = train_all(model_specs, X_train, y_train, X_test, y_test)

    print("\n[3/4] 生成评估图表 ...")
    fig1_distribution(y_train, y_test)
    fig2_model_comparison(results)
    fig3_confusion_matrices(results, y_test)
    fig4_per_class_f1(results)
    fig5_roc_curves(results, y_test)
    fig6_feature_importance(results, trained, feature_cols)
    fig7_radar(results)
    fig8_training_time(results)
    fig9_pred_vs_actual(results, y_test)

    print("\n[4/4] 保存最优模型 ...")
    best_name = save_best(results, trained, feature_cols, dataset_meta)
    save_results_json(results, best_name, feature_cols, dataset_meta)

    print_summary(results, best_name, dataset_meta)

    print("完成！")
    print(f"   图表目录: {FIG_DIR}")
    print(f"   模型文件: {os.path.join(MODEL_DIR, 'best_model.pkl')}")
