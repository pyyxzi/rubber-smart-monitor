"""
评估图表生成模块（图1~图9）
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, roc_curve, auc

from config import FIG_DIR, CLASS_NAMES, PALETTE


def save_fig(fig, filename):
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    [OK] {filename}")


def fig1_distribution(y_train, y_test):
    """图1：训练集/测试集标签分布"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#4CAF50", "#FFC107", "#FF9800", "#F44336"]
    for ax, y, title in zip(axes, [y_train, y_test], ["训练集", "测试集"]):
        counts = y.value_counts().sort_index()
        bars = ax.bar([CLASS_NAMES[i] for i in counts.index], counts.values,
                      color=colors, edgecolor="white", linewidth=0.8)
        for bar, cnt in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + counts.max() * 0.02,
                    f"{cnt:,}\n({cnt/len(y)*100:.1f}%)",
                    ha="center", va="bottom", fontsize=10)
        ax.set_title(f"{title}标签分布（共 {len(y):,} 条）", fontsize=13, fontweight="bold")
        ax.set_ylabel("样本数")
        ax.set_ylim(0, counts.max() * 1.25)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("数据集标签分布", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "01_class_distribution.png")


def fig2_model_comparison(results):
    """图2：多模型多指标分组柱状图"""
    names   = list(results.keys())
    metrics = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro", "auc_macro"]
    labels  = ["准确率", "宏F1", "加权F1", "宏精确率", "宏召回率", "宏AUC"]

    x = np.arange(len(names))
    w = 0.12
    fig, ax = plt.subplots(figsize=(17, 7))

    for i, (metric, label, color) in enumerate(zip(metrics, labels, PALETTE)):
        vals = [results[n][metric] if results[n][metric] is not None else 0 for n in names]
        bars = ax.bar(x + i * w, vals, w, label=label, color=color, alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=90)

    ax.set_xticks(x + w * (len(metrics) - 1) / 2)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel("得分", fontsize=12)
    ax.set_title("各模型多指标综合对比", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10, ncol=3)
    ax.set_ylim(0, 1.18)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "02_model_comparison.png")


def fig3_confusion_matrices(results, y_test):
    """图3：所有模型混淆矩阵"""
    names  = list(results.keys())
    ncols  = 3
    nrows  = (len(names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6 * nrows))
    axes = axes.flatten()

    for i, name in enumerate(names):
        cm      = confusion_matrix(y_test, results[name]["y_pred"], labels=[0, 1, 2, 3])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    ax=axes[i], cbar=True, linewidths=0.5, annot_kws={"size": 11})
        axes[i].set_title(
            f"{name}  (准确率={results[name]['accuracy']:.4f})",
            fontsize=11, fontweight="bold")
        axes[i].set_xlabel("预测标签", fontsize=10)
        axes[i].set_ylabel("真实标签", fontsize=10)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("各模型混淆矩阵（颜色=行归一化比例，数字=实际样本数）",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "03_confusion_matrices.png")


def fig4_per_class_f1(results):
    """图4：各类别F1得分（柱状图 + 热力图）"""
    names = list(results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(17, 6))

    # 柱状图
    ax  = axes[0]
    x   = np.arange(4)
    w   = 0.13
    for i, (name, color) in enumerate(zip(names, PALETTE)):
        ax.bar(x + i * w, results[name]["f1_per_class"], w,
               label=name, color=color, alpha=0.88)
    ax.set_xticks(x + w * (len(names) - 1) / 2)
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_ylabel("F1 得分")
    ax.set_title("各类别 F1 得分对比", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", alpha=0.3)

    # 热力图
    ax2      = axes[1]
    matrix   = np.array([results[n]["f1_per_class"] for n in names])
    im       = ax2.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax2.set_xticks(range(4));  ax2.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax2.set_yticks(range(len(names))); ax2.set_yticklabels(names, fontsize=11)
    ax2.set_title("各类别 F1 热力图", fontsize=13, fontweight="bold")
    for r in range(len(names)):
        for c in range(4):
            ax2.text(c, r, f"{matrix[r, c]:.3f}",
                     ha="center", va="center", fontsize=10,
                     color="black" if matrix[r, c] > 0.3 else "white")
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle("各模型分类别 F1 得分分析", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "04_per_class_f1.png")


def fig5_roc_curves(results, y_test):
    """图5：各模型 ROC 曲线（One-vs-Rest）"""
    has_prob = {n: r for n, r in results.items() if r["y_prob"] is not None}
    n_models  = len(has_prob)
    ncols = 3
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6 * nrows))
    axes = axes.flatten()

    y_bin   = label_binarize(y_test, classes=[0, 1, 2, 3])
    cls_clr = ["#4CAF50", "#FFC107", "#FF9800", "#F44336"]

    for i, (name, res) in enumerate(has_prob.items()):
        ax = axes[i]
        for cls, (cname, color) in enumerate(zip(CLASS_NAMES, cls_clr)):
            fpr, tpr, _ = roc_curve(y_bin[:, cls], res["y_prob"][:, cls])
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f"{cname}  AUC={auc(fpr, tpr):.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("假阳性率 (FPR)"); ax.set_ylabel("真阳性率 (TPR)")
        ax.set_title(f"{name}\n宏均AUC = {res['auc_macro']:.4f}",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("各模型 ROC 曲线（One-vs-Rest，各发病等级）",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "05_roc_curves.png")


def fig6_feature_importance(results, trained, feature_cols):
    """图6：最优树模型的 Top-20 特征重要性"""
    for preferred in ["随机森林", "梯度提升", "决策树"]:
        if preferred in trained and trained[preferred][0] is not None:
            name = preferred
            break
    else:
        print("    [WARN] 无可用树模型，跳过特征重要性图")
        return

    model, _ = trained[name]
    imp  = model.feature_importances_
    idx  = np.argsort(imp)[::-1][:20]
    top_feat = [feature_cols[i] for i in idx]
    top_imp  = imp[idx]

    # 从高到低：绘图时逆序（barh 从下到上）
    top_feat = top_feat[::-1]
    top_imp  = top_imp[::-1]

    fig, ax = plt.subplots(figsize=(12, 8))
    colors  = plt.cm.RdYlGn(np.linspace(0.25, 0.85, len(top_feat)))
    bars = ax.barh(top_feat, top_imp, color=colors, edgecolor="white")
    for bar, v in zip(bars, top_imp):
        ax.text(bar.get_width() + top_imp.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=9)
    ax.set_xlabel("特征重要性（基尼不纯度降低）", fontsize=11)
    ax.set_title(f"Top-20 特征重要性 —— {name}", fontsize=14, fontweight="bold")
    ax.set_xlim(0, top_imp.max() * 1.18)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "06_feature_importance.png")


def fig7_radar(results):
    """图7：各模型综合指标雷达图"""
    metrics = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
    labels  = ["准确率", "宏F1", "加权F1", "宏精确率", "宏召回率"]
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw=dict(polar=True))
    for (name, res), color in zip(results.items(), PALETTE):
        vals = [res[m] if res[m] is not None else 0 for m in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", lw=2, label=name, color=color)
        ax.fill(angles, vals, alpha=0.07, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_title("各模型综合指标雷达图", fontsize=15, fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15), fontsize=11)
    ax.grid(True)
    fig.tight_layout()
    save_fig(fig, "07_radar_chart.png")


def fig8_training_time(results):
    """图8：各模型训练耗时"""
    names = list(results.keys())
    times = [results[n]["time"] for n in names]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(names)))
    bars = ax.bar(names, times, color=colors, edgecolor="white")
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(times) * 0.01,
                f"{t:.1f}s", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("训练时间（秒）", fontsize=12)
    ax.set_title("各模型训练耗时对比", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(times) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "08_training_time.png")


def fig9_pred_vs_actual(results, y_test):
    """图9：预测类别分布 vs 真实分布 + 最优模型时序对比"""
    names = list(results.keys())
    ncols = 3
    nrows = (len(names) + ncols - 1) // ncols
    bar_colors = ["#4CAF50", "#FFC107", "#FF9800", "#F44336"]

    # 图9a：各模型预测类别分布 vs 真实分布
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
    axes = axes.flatten()
    y_t_arr = y_test.values if hasattr(y_test, "values") else y_test
    for i, name in enumerate(names):
        ax    = axes[i]
        y_p   = results[name]["y_pred"]
        x     = np.arange(4)
        w     = 0.35
        actual_cnt = [np.sum(y_t_arr == c) for c in range(4)]
        pred_cnt   = [np.sum(y_p == c)     for c in range(4)]
        ax.bar(x - w / 2, actual_cnt, w, label="真实", color=bar_colors, alpha=0.65)
        ax.bar(x + w / 2, pred_cnt,   w, label="预测", color=bar_colors, alpha=1.0,
               edgecolor="white", hatch="//")
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES, fontsize=10)
        ax.set_title(f"{name}  (准确率={results[name]['accuracy']:.4f})",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("样本数")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("各模型预测类别分布 vs 真实类别分布", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "09a_pred_vs_actual_distribution.png")

    # 图9b：最优模型在测试集前 300 条上的时序对比
    best  = max(results, key=lambda n: results[n]["f1_macro"])
    y_p   = results[best]["y_pred"]
    n_show = min(300, len(y_t_arr))
    idx    = np.arange(n_show)

    fig2, ax2 = plt.subplots(figsize=(18, 5))
    ax2.plot(idx, y_t_arr[:n_show], "o-", lw=1.2, ms=3,
             color="#2196F3", alpha=0.8, label="真实等级")
    ax2.plot(idx, y_p[:n_show], "s--", lw=1.2, ms=3,
             color="#F44336", alpha=0.8, label="预测等级")
    wrong = np.where(y_t_arr[:n_show] != y_p[:n_show])[0]
    ax2.scatter(wrong, y_t_arr[wrong], c="red", s=45, zorder=5,
                marker="x", linewidths=1.5)
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(CLASS_NAMES, fontsize=10)
    ax2.set_xlabel("测试样本序号")
    ax2.set_ylabel("发病等级")
    acc_v = results[best]["accuracy"]
    ax2.set_title(
        f"最优模型 [{best}] 预测 vs 真实（前 {n_show} 条，×=预测错误，准确率={acc_v:.4f}）",
        fontsize=13, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    save_fig(fig2, "09b_pred_vs_actual_timeseries.png")
