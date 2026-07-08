"""
橡胶树白粉病预警 + 图像检测 - 统一 Web 界面
运行: python app.py
访问: http://localhost:5000
"""

import base64
import io
import json
import math
import os
import pickle
from datetime import datetime

import cv2
import numpy as np
import requests
import torch
import torch.nn as nn
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
from torchvision import models, transforms

# ── 路径（均相对于 Project/ 根目录） ─────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

DATA_FIG_DIR = os.path.join(BASE, "csv_train", "output")
PIC_FIG_DIR  = os.path.join(BASE, "pic_train",  "output")
DATA_MODEL   = os.path.join(BASE, "csv_train", "models", "best_model.pkl")
DATA_RESULTS = os.path.join(BASE, "csv_train", "models", "results.json")
PIC_MODEL    = os.path.join(BASE, "pic_train", "output", "best_model.pth")

# ── AI 配置 ───────────────────────────────────────────────────────────────────
# AI_KEY   = "sk-vXRcE8INcXS06yPHB6B2968dEfF6483c93F61b8b84B60c4a"
# AI_URL   = "https://one.ocoolai.com/v1/chat/completions"
# AI_MODEL = "gpt-4o-mini"

AI_KEY   = "sk-28547415a6674509943af6c6698ee6b5"
AI_URL   = "https://api.deepseek.com/v1/chat/completions"
AI_MODEL = "deepseek-v4-flash"

# ── 白粉病等级 ────────────────────────────────────────────────────────────────
LABELS = ["无病(0)", "轻度(1)", "中度(2)", "重度(3)"]
PHENOLOGY_MAP = {
    1: 0.50, 2: 0.80, 3: 1.00, 4: 0.90,
    5: 0.40, 6: 0.20, 7: 0.15, 8: 0.15,
    9: 0.20, 10: 0.30, 11: 0.40, 12: 0.45,
}

# ── 图像模型 ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
CONF_THRESHOLD = 0.75
LABEL_MAP = {"Healthy": "健康叶片", "Unhealthy": "非健康叶片（疑似病害）"}

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_pic_model():
    """加载图像分类模型（ResNet-18）。若权重文件不存在则返回 (None, None)。"""
    if not os.path.exists(PIC_MODEL):
        print(f"图像模型未找到: {PIC_MODEL}")
        return None, None
    try:
        ckpt = torch.load(PIC_MODEL, map_location=DEVICE, weights_only=False)
    except TypeError:
        # 旧版 PyTorch 不支持 weights_only 参数，降级调用
        ckpt = torch.load(PIC_MODEL, map_location=DEVICE)
    cls = ckpt["class_names"]
    # 重建与训练时相同的 ResNet-18 头部结构
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(
        nn.Dropout(0.5), nn.Linear(m.fc.in_features, 256),
        nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, len(cls)),
    )
    m.load_state_dict(ckpt["model_state_dict"])
    m.to(DEVICE).eval()
    print(f"图像模型已加载，类别: {cls}")
    return m, cls


pic_model, class_names = load_pic_model()

app = Flask(__name__)


# ── 工具函数（数据预测） ──────────────────────────────────────────────────────
def clamp(v, lo, hi):
    """将 v 限制在 [lo, hi] 区间内。"""
    return max(lo, min(hi, v))


def parse_float(data, key, default=0.0):
    """从字典 data 中读取 key 对应的浮点值，缺失或空值时返回 default。"""
    raw = data.get(key, default)
    return float(default) if raw in (None, "") else float(raw)


def load_data_pickle():
    """加载 csv_train/models/best_model.pkl，返回含模型、缩放器、特征列等的 payload 字典。"""
    if not os.path.exists(DATA_MODEL):
        raise FileNotFoundError("请先运行 python csv_train/train_models.py")
    with open(DATA_MODEL, "rb") as f:
        return pickle.load(f)


def load_results_json():
    """加载 csv_train/models/results.json，返回所有模型的评估指标。"""
    if not os.path.exists(DATA_RESULTS):
        raise FileNotFoundError("请先运行 python csv_train/train_models.py")
    with open(DATA_RESULTS, "r", encoding="utf-8") as f:
        return json.load(f)


def dewpoint(temp_c, rh):
    """Magnus 公式计算露点温度（°C）。rh 为相对湿度百分比。"""
    rh = clamp(rh, 1.0, 100.0)
    a, b = 17.27, 237.7
    g = (a * temp_c) / (b + temp_c) + math.log(rh / 100.0)
    return (b * g) / (a - g)


def build_features(data, payload):
    """
    根据用户输入的 5 项关键气象观测值推导出模型所需的全量特征向量。

    用户只需提供：temperature_2m_mean、relative_humidity_mean、leaf_wetness_hours、
    precip_sum_7d、consecutive_suitable_days 和 obs_date。
    其余次级特征（温差、露点、滚动统计量等）由此推导，
    未能推导的特征用训练集 prediction_defaults 中的稳健中位数兜底。

    Returns:
        feats  (dict): 与 feature_cols 对齐的完整特征字典，供模型推断。
        derived (dict): 前端展示用的辅助信息（日期、月份、物候敏感度等）。
    """
    meta = payload.get("dataset_meta", {})
    defs = meta.get("prediction_defaults", {})  # 训练集稳健中位数，作为缺失值兜底

    raw_date = data.get("obs_date") or datetime.now().strftime("%Y-%m-%d")
    try:
        obs = datetime.strptime(raw_date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("监测日期格式应为 YYYY-MM-DD") from exc

    # ── 读取 5 项核心输入，越界时截断 ────────────────────────────────────────
    t    = parse_float(data, "temperature_2m_mean",      defs.get("temperature_2m_mean", 24.7))
    rh   = clamp(parse_float(data, "relative_humidity_mean",   defs.get("relative_humidity_mean", 80.0)), 0, 100)
    lw   = clamp(parse_float(data, "leaf_wetness_hours",       defs.get("leaf_wetness_hours", 8.0)), 0, 24)
    p7   = max(parse_float(data, "precip_sum_7d",              defs.get("precip_sum_7d", 28.0)), 0)
    csd  = clamp(parse_float(data, "consecutive_suitable_days", defs.get("consecutive_suitable_days", 3.0)), 0, 90)

    # ── 从日期推导时间特征与物候敏感度 ────────────────────────────────────────
    month = obs.month
    doy   = obs.timetuple().tm_yday       # 年积日（1-366）
    pheno = PHENOLOGY_MAP.get(month, 0.45)  # 按月份查表，反映橡胶抽叶物候周期

    # ── 推导次级气象变量 ──────────────────────────────────────────────────────
    tr    = max(float(defs.get("temp_range", 7.0)), 1.0)  # 日温差，最小 1°C
    dp    = dewpoint(t, rh)                               # 当前露点
    ddm   = float(defs.get("dewpoint_2m_mean", dp))       # 训练集露点均值（用于偏差估算）
    prec  = p7 / 7.0                                      # 7 日总降水折算为日均降水

    def _rh_gap(key, fallback, lo=True):
        """从训练集默认值估算相对湿度与均值的偏差，至少保留 2% 的区间宽度。"""
        base = float(defs.get("relative_humidity_mean", rh))
        ref  = float(defs.get(key, fallback))
        return max(2.0, abs(base - ref))

    rh_lo = _rh_gap("relative_humidity_min", max(float(defs.get("relative_humidity_mean", rh)) - 20, 0))
    rh_hi = _rh_gap("relative_humidity_max", min(float(defs.get("relative_humidity_mean", rh)) + 15, 100))

    # 由日均降水与训练集降水时数比例推算当日降水时数
    dps = float(defs.get("precipitation_sum", 4.0)) or 0.1
    dph = float(defs.get("precipitation_hours", 5.0))
    ph  = clamp(prec * (dph / dps), 0, 24)

    # ── 组装特征字典：先用训练集默认值填充所有特征列，再覆写可推导的特征 ──────
    feats = {f: float(defs.get(f, 0.0)) for f in payload["feature_cols"]}
    feats.update({
        # 温度特征
        "temperature_2m_mean": round(t, 4),
        "temperature_2m_max":  round(t + tr / 2, 4),
        "temperature_2m_min":  round(t - tr / 2, 4),
        "temp_range":          round(tr, 4),
        # 湿度特征
        "relative_humidity_mean": round(rh, 4),
        "relative_humidity_max":  round(clamp(rh + rh_hi, rh, 100), 4),
        "relative_humidity_min":  round(clamp(rh - rh_lo, 0, rh), 4),
        # 露点特征
        "dewpoint_2m_mean": round(dp, 4),
        "dewpoint_2m_min":  round(dp - max(0.5, ddm - float(defs.get("dewpoint_2m_min", ddm - 1))), 4),
        "dewpoint_2m_max":  round(dp + max(0.5, float(defs.get("dewpoint_2m_max", ddm + 1)) - ddm), 4),
        # 降水特征
        "precipitation_sum":  round(prec, 4),
        "rain_sum":           round(prec, 4),
        "precipitation_hours": round(ph, 4),
        # 时间特征
        "month":       float(month),
        "day_of_year": float(doy),
        # 叶面湿润
        "leaf_wetness_hours": round(lw, 4),
        # 滚动窗口统计（3/5/7 日），单点观测时用当日值近似
        "temp_mean_3d": round(t, 4), "temp_mean_5d": round(t, 4), "temp_mean_7d": round(t, 4),
        "rh_mean_3d":   round(rh, 4), "rh_mean_5d": round(rh, 4), "rh_mean_7d": round(rh, 4),
        "precip_sum_3d": round(p7 * 3 / 7, 4),
        "precip_sum_5d": round(p7 * 5 / 7, 4),
        "precip_sum_7d": round(p7, 4),
        "leaf_wet_3d": round(lw * 3, 4),
        "leaf_wet_5d": round(lw * 5, 4),
        "leaf_wet_7d": round(lw * 7, 4),
        # 流行病学特征
        "consecutive_suitable_days": round(csd, 4),
        "phenology_sensitivity":     round(pheno, 4),
    })
    # 返回给前端展示的辅助字段
    derived = {
        "obs_date": obs.strftime("%Y-%m-%d"), "month": month,
        "day_of_year": doy, "phenology_sensitivity": round(pheno, 4),
        "daily_precipitation": round(prec, 4),
    }
    return feats, derived


# ── GradCAM ───────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping（Grad-CAM）。
    通过对目标层的梯度加权激活图，可视化模型关注的图像区域。
    """

    def __init__(self, model_obj, layer):
        """注册前向钩子和反向钩子以捕获指定层的激活值与梯度。"""
        self.model = model_obj
        self.grad = self.act = None
        layer.register_forward_hook(lambda _m, _i, o: setattr(self, "act", o.detach()))
        layer.register_full_backward_hook(lambda _m, _gi, go: setattr(self, "grad", go[0].detach()))

    def generate(self, inp):
        """
        对输入图像 inp 进行前向+反向传播，返回归一化至 [0,1] 的 CAM 热力图（H×W numpy 数组）。
        预测类别由模型最高分数确定（argmax）。
        """
        out = self.model(inp)
        idx = out.argmax(1).item()        # 取置信度最高的预测类别
        self.model.zero_grad()
        out[0, idx].backward()            # 对该类别得分反向传播
        w = self.grad.mean(dim=(2, 3), keepdim=True)   # 全局平均池化梯度得到通道权重
        cam = torch.relu((w * self.act).sum(1, keepdim=True)).squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()              # 归一化至 [0, 1]
        return cam


def make_gradcam_b64(image, inp):
    """
    生成 Grad-CAM 热力图并与原图叠加，返回 base64 编码的 JPEG 字符串（data URI 格式）。
    叠加比例：原图 55%，热力图 45%。
    """
    if pic_model is None:
        return None
    cam = GradCAM(pic_model, pic_model.layer4).generate(inp)
    arr = np.array(image.resize((IMG_SIZE, IMG_SIZE)), dtype=np.uint8)
    # 将灰度 CAM 映射为 JET 伪彩色，再从 BGR 转为 RGB
    hm  = cv2.cvtColor(cv2.applyColorMap(np.uint8(cv2.resize(cam, (IMG_SIZE, IMG_SIZE)) * 255), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray((0.55 * arr + 0.45 * hm).astype(np.uint8)).save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """返回前端主页面 index.html。"""
    with open(os.path.join(BASE, "index.html"), "r", encoding="utf-8") as f:
        return f.read()


@app.route("/api/results")
def api_results():
    """返回 csv_train 所有模型的评估指标 JSON。"""
    try:
        return jsonify(load_results_json())
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404


@app.route("/api/figures")
def api_figures():
    """返回 csv_train/output/ 下所有 PNG 文件名列表（已排序）。"""
    if not os.path.isdir(DATA_FIG_DIR):
        return jsonify([])
    return jsonify(sorted(n for n in os.listdir(DATA_FIG_DIR) if n.endswith(".png")))


@app.route("/figures/<path:filename>")
def figures(filename):
    """提供 csv_train/output/ 下图表文件的静态服务。"""
    return send_from_directory(DATA_FIG_DIR, filename)


@app.route("/api/pic-figures")
def api_pic_figures():
    """返回 pic_train/output/ 下所有 PNG 文件名列表（已排序）。"""
    if not os.path.isdir(PIC_FIG_DIR):
        return jsonify([])
    return jsonify(sorted(n for n in os.listdir(PIC_FIG_DIR) if n.endswith(".png")))


@app.route("/pic-figures/<path:filename>")
def pic_figures(filename):
    """提供 pic_train/output/ 下图表文件的静态服务。"""
    return send_from_directory(PIC_FIG_DIR, filename)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    数据预测接口。
    接收 JSON 格式的气象参数，调用 build_features 推导全量特征后，
    使用 best_model.pkl 中的最优分类器输出白粉病等级及各级概率。
    """
    try:
        payload = load_data_pickle()
        data = request.get_json(force=True) or {}
        feats, derived = build_features(data, payload)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # 按 feature_cols 顺序组装特征矩阵，必要时进行标准化
    X = np.array([[float(feats.get(f, 0.0)) for f in payload["feature_cols"]]])
    if payload["scaler"] is not None:
        X = payload["scaler"].transform(X)
    probs = payload["model"].predict_proba(X)[0]
    level = int(np.argmax(probs))
    dm = payload.get("dataset_meta", {})
    return jsonify({
        "level": level, "label": LABELS[level],
        "probs": {LABELS[i]: round(float(v), 4) for i, v in enumerate(probs)},
        "model": payload["model_name"], "derived": derived,
        "note": dm.get("label_note", ""),
    })


@app.route("/api/predict-image", methods=["POST"])
def api_predict_image():
    """
    图像检测接口。
    接收上传的叶片图片（multipart/form-data），返回：
    - prediction: 中文类别标签
    - confidence: 置信度（%）
    - probabilities: 各类别概率
    - gradcam: Grad-CAM 热力图的 base64 data URI（可选）
    - image: 原图 base64 data URI
    置信度低于 CONF_THRESHOLD(0.75) 时判定为其他类别。
    """
    if pic_model is None:
        return jsonify({"error": "图像模型未加载，请先运行 pic_train/train.py"}), 503
    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"error": "请上传图片文件"}), 400
    file = request.files["file"]
    ext  = os.path.splitext(file.filename)[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return jsonify({"error": f"不支持的格式: {ext}"}), 400

    image_bytes = file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return jsonify({"error": "无法解析图片"}), 400

    # 预处理并启用梯度（Grad-CAM 需要反向传播）
    inp = transform(image).unsqueeze(0).to(DEVICE)
    inp.requires_grad_(True)
    cam_b64 = make_gradcam_b64(image, inp)

    with torch.no_grad():
        probs = torch.softmax(pic_model(inp), dim=1)[0]
        max_p, pred_i = torch.max(probs, 0)

    pred_cls = class_names[pred_i.item()]
    # 置信度不足时归为"其他"，避免错误的强制分类
    if max_p.item() < CONF_THRESHOLD:
        label, cls_key = "其他（非目标叶片或其他病害）", "other"
    else:
        label, cls_key = LABEL_MAP.get(pred_cls, pred_cls), pred_cls

    result = {
        "prediction": label, "result_class": cls_key,
        "confidence": round(max_p.item() * 100, 2),
        "probabilities": {LABEL_MAP.get(n, n): round(probs[i].item() * 100, 2) for i, n in enumerate(class_names)},
        "image": f"data:image/{ext.lstrip('.')};base64,{base64.b64encode(image_bytes).decode()}",
    }
    if cam_b64:
        result["gradcam"] = cam_b64
    return jsonify(result)


def call_ai(messages):
    """
    向 AI 接口发送 messages 列表并返回模型回复文本。
    messages 格式遵循 OpenAI Chat Completions 规范（role/content 字典列表）。
    超时设置为 60 秒；HTTP 错误时抛出异常。
    """
    resp = requests.post(
        AI_URL,
        headers={"Authorization": f"Bearer {AI_KEY}", "Content-Type": "application/json"},
        json={"model": AI_MODEL, "messages": messages},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


@app.route("/api/ai-analyze", methods=["POST"])
def api_ai_analyze():
    """
    AI 分析接口，根据 mode 参数生成不同类型的报告：
      mode='analyze'    （默认）：【诊断报告】+ 【风险评估】
      mode='prevention'         ：【化学防治】+ 【农业防治】+ 【预警建议】
    请求体须包含 lastPrediction 字段（气象参数 + 预测结果）。
    """
    d = request.get_json(force=True) or {}
    mode = d.get("mode", "analyze")
    probs_str = "  ".join(f"{k}: {v*100:.1f}%" for k, v in d.get("probs", {}).items())
    context = (
        f"气象参数：\n"
        f"  平均气温：{d.get('temperature', '-')}°C\n"
        f"  平均相对湿度：{d.get('humidity', '-')}%\n"
        f"  叶面湿润时数：{d.get('leaf_wetness', '-')} h/天\n"
        f"  近7日总降水：{d.get('precip', '-')} mm\n"
        f"  连续适宜天数：{d.get('consecutive_days', '-')} 天\n"
        f"  监测日期：{d.get('obs_date', '-')}（{d.get('month', '-')}月，物候敏感度：{d.get('phenology', '-')}）\n\n"
        f"预测结果：{d.get('label', '-')}\n"
        f"各等级概率：{probs_str}\n\n"
    )
    if mode == "prevention":
        prompt = (
            "你是橡胶树白粉病防治专家。根据以下气象监测数据和预测结果，"
            "制定详细的防治方案。\n\n" + context +
            "请按以下格式输出（不超过400字）：\n"
            "【化学防治】\n（推荐药剂、施药时机与用量）\n\n"
            "【农业防治】\n（栽培管理与减少侵染源的措施）\n\n"
            "【预警建议】\n（监测频率与应急响应建议）"
        )
    else:
        prompt = (
            "你是橡胶树白粉病防治专家。根据以下气象监测数据和预测结果，"
            "提供简洁的诊断报告。\n\n" + context +
            "请按以下格式输出（每部分不超过150字）：\n"
            "【诊断报告】\n（分析当前发病风险等级及主要气象驱动因素）\n\n"
            "【风险评估】\n（近期趋势研判与重点关注区域）"
        )
    try:
        reply = call_ai([{"role": "user", "content": prompt}])
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ai-chat", methods=["POST"])
def api_ai_chat():
    """
    AI 自由问答接口。
    接收 messages 历史列表（role/content），在前端追加系统 prompt 后调用 AI。
    系统 prompt 将 AI 设定为橡胶树病虫害防治专家角色。
    """
    d = request.get_json(force=True) or {}
    messages = d.get("messages", [])
    system = {
        "role": "system",
        "content": (
            "你是橡胶树种植与病虫害防治专家，精通橡胶树白粉病、炭疽病、根病等病害的识别与防治，"
            "以及橡胶树栽培管理技术。请用中文回答，回答专业、简洁、实用。"
        ),
    }
    try:
        reply = call_ai([system] + messages)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("访问 http://localhost:5000")
    app.run(port=5000, debug=True, use_reloader=False)
