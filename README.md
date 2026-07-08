# 橡胶树病虫害智能化监测系统

本项目面向橡胶树白粉病监测与预警，集成了气象数据处理、机器学习风险分级、叶片图片健康检测、训练结果可视化和 AI 辅助分析功能。系统采用 Flask 提供后端接口，前端使用原生 HTML/CSS/JavaScript 构建，可在浏览器中完成数据预测、图片检测和模型结果查看。

## 功能特点

- 气象数据采集：基于 Open-Meteo Archive API 获取海南橡胶主产区历史气象数据。
- 数据清洗与特征工程：生成温度、湿度、降水、叶面湿润、连续适宜天数、物候敏感度等特征。
- 白粉病风险预测：训练多种传统机器学习模型，对白粉病风险等级进行分类预测。
- 叶片图片检测：基于 ResNet-18 迁移学习识别健康叶片与疑似病害叶片。
- Grad-CAM 可视化：展示图片分类模型关注的叶片区域。
- 训练结果展示：在 Web 页面中查看模型指标、混淆矩阵、ROC 曲线、特征重要性等图表。
- AI 辅助分析：根据预测结果生成诊断分析、防治建议，并支持病虫害相关问答。

## 项目结构

```text
.
├── app.py                     # Flask 后端入口
├── index.html                 # Web 页面入口
├── static/                    # 前端样式与交互脚本
│   ├── main.js
│   └── style.css
├── data_csv/                  # 气象数据采集、清洗与处理
│   ├── data_get.py            # 获取 Open-Meteo 历史气象数据
│   ├── data_process.py        # 数据清洗、特征工程与标签生成
│   └── processed/             # 处理后的训练集、测试集和元数据
├── csv_train/                 # 气象数据风险预测模型训练
│   ├── train_models.py        # 多模型训练与评估入口
│   ├── models.py              # 传统机器学习模型定义
│   ├── plots.py               # 评估图表生成
│   ├── models/                # 已训练模型与评估结果
│   └── output/                # 训练评估可视化图表
├── data_pic/                  # 叶片图片数据集
│   ├── Healthy/
│   └── Unhealthy/
└── pic_train/                 # 图片分类模型训练
    ├── train.py               # ResNet-18 训练入口
    ├── dataset.py             # 图片数据加载与增强
    ├── model.py               # 模型结构
    ├── evaluate.py            # 模型评估
    └── visualize.py           # 训练图表与评估图表
```

## 环境要求

建议使用 Python 3.9 或以上版本。

主要依赖：

```bash
pip install flask numpy pandas requests scikit-learn matplotlib seaborn opencv-python pillow
pip install torch torchvision
```

如果需要 GPU 加速，请根据本机 CUDA 版本到 PyTorch 官网选择对应安装命令。

## 快速运行

进入项目根目录后运行：

```bash
python app.py
```

浏览器访问：

```text
http://localhost:5000
```

系统页面包含：

- 训练结果：查看气象预测模型的整体指标。
- 图表展示：查看分类分布、模型对比、混淆矩阵、ROC 曲线、特征重要性等图表。
- 数据预测：输入气温、湿度、叶面湿润时数、近 7 日降水量、连续适宜天数和监测日期，预测白粉病风险等级。
- 图片检测：上传橡胶树叶片图片，判断健康/疑似病害，并显示 Grad-CAM 热力图。
- AI 分析：基于预测结果生成诊断报告和防治建议，也可进行自由问答。

## 数据处理流程

### 1. 获取气象数据

```bash
python data_csv/data_get.py
```

脚本会显示海南橡胶主产区站点列表，可选择单个或多个区域抓取历史气象数据。数据来源为 Open-Meteo Archive API，结果保存到 `data_csv/` 目录下。

### 2. 清洗与构建数据集

```bash
python data_csv/data_process.py
```

该步骤会完成：

- 原始 CSV 合并与去重
- 日期标准化
- 按时间顺序划分训练集和测试集
- 缺失值填补
- 相对湿度推算
- 滚动窗口特征构建
- 基于气象条件和流行病学规则生成风险代理标签

输出文件位于：

```text
data_csv/processed/
├── full_dataset.csv
├── train.csv
├── test.csv
└── dataset_metadata.json
```

## 气象预测模型训练

运行：

```bash
python csv_train/train_models.py
```

当前训练流程包含以下模型：

- 逻辑回归
- 决策树
- 随机森林
- 梯度提升
- KNN
- SVM

训练完成后会保存：

```text
csv_train/models/best_model.pkl       # 最优模型
csv_train/models/results.json         # 模型评估结果
csv_train/output/*.png                # 评估图表
```

当前项目中已包含训练好的气象预测模型，Web 系统可直接读取 `csv_train/models/best_model.pkl` 进行预测。

## 图片分类模型训练

图片数据集目录格式：

```text
data_pic/
├── Healthy/
└── Unhealthy/
```

训练命令：

```bash
python pic_train/train.py
```

该模块使用 ResNet-18 迁移学习，训练完成后默认保存：

```text
pic_train/output/best_model.pth
```

如果该模型文件不存在，Web 系统仍可运行，但图片检测接口会提示需要先训练图片模型。

## 接口说明

Flask 后端提供以下主要接口：

| 接口 | 方法 | 功能 |
| --- | --- | --- |
| `/api/results` | GET | 获取气象模型训练评估结果 |
| `/api/figures` | GET | 获取气象模型评估图表列表 |
| `/api/pic-figures` | GET | 获取图片模型评估图表列表 |
| `/api/predict` | POST | 根据气象参数预测白粉病风险等级 |
| `/api/predict-image` | POST | 上传叶片图片并进行健康状态检测 |
| `/api/ai-analyze` | POST | 生成 AI 诊断分析或防治建议 |
| `/api/ai-chat` | POST | AI 问答 |

## 注意事项

- 当前白粉病等级标签是基于气象条件和流行病学规则生成的风险代理标签，不等同于真实田间病情调查结果。
- 如果用于生产或科研结论，应结合田间实测病害数据进一步校正模型。
- `app.py` 中 AI 接口需要有效的 API Key。公开上传 GitHub 前，建议不要把真实密钥写死在代码中，应改为通过环境变量读取。
- 训练图片模型时，当前示例图片数量较少，实际应用中建议扩充不同地区、不同光照、不同病害阶段的叶片图片数据。


