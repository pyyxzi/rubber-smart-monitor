"""
步骤2：数据清洗整合

自动读取 data_csv/ 目录中的站点 CSV，完成：
1. 去重与时间标准化
2. 按唯一日期做时序拆分，避免测试集信息泄漏到训练集
3. 基于站点历史做顺序填补（前向填补 + 训练集统计量兜底）
4. 相对湿度推算与特征工程
5. 基于流行病学规则生成风险代理标签
6. 保存训练集、测试集、完整数据集与数据集元信息
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd


# ============================================================
# 配置
# ============================================================
TRAIN_RATIO = 0.8
LABEL_NAME_MAP = {0: "无病", 1: "轻度", 2: "中度", 3: "重度"}
PHENOLOGY_MAP = {
    1: 0.50, 2: 0.80, 3: 1.00, 4: 0.90,
    5: 0.40, 6: 0.20, 7: 0.15, 8: 0.15,
    9: 0.20, 10: 0.30, 11: 0.40, 12: 0.45,
}
IDENTIFIER_COLS = {"station", "date"}
LABEL_COLS = {"disease_risk_score", "disease_level", "disease_level_name"}


def load_raw_data(data_dir):
    """从 data_csv/ 目录加载所有站点 CSV 文件。"""
    files = [
        f for f in os.listdir(data_dir)
        if f.endswith(".csv") and not f.startswith("_")
        and os.path.isfile(os.path.join(data_dir, f))
    ]

    if not files:
        print("[ERROR] data_csv/ 目录中没有找到 CSV 数据文件。")
        print("  请先运行 data_get.py 抓取数据。")
        return pd.DataFrame()

    print(f"找到 {len(files)} 个数据文件")
    frames = []
    for filename in sorted(files):
        path = os.path.join(data_dir, filename)
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
            frames.append(df)
            print(f"  [OK] {filename}: {len(df)} 条记录")
        except Exception as exc:
            print(f"  [FAIL] {filename}: 读取失败 ({exc})")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def standardize_data(df):
    """去重并标准化日期。"""
    print("\n--- 数据清洗 ---")
    n_before = len(df)

    df = df.drop_duplicates(subset=["station", "date"], keep="last").copy()
    n_dedup = n_before - len(df)
    if n_dedup > 0:
        print(f"  去除重复记录: {n_dedup} 条")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    invalid_dates = df["date"].isna().sum()
    if invalid_dates > 0:
        raise ValueError(f"存在 {invalid_dates} 条无法解析的日期，请先检查原始数据。")

    df = df.sort_values(["station", "date"]).reset_index(drop=True)
    print(f"  清洗后记录数: {len(df)}")
    print(f"  时间范围: {df['date'].min().date()} ~ {df['date'].max().date()}")
    return df


def split_by_unique_dates(df, train_ratio=TRAIN_RATIO):
    """按照唯一日期进行时序拆分，避免同一天被切到训练集和测试集两侧。"""
    unique_dates = np.array(sorted(df["date"].dt.normalize().unique()))
    if len(unique_dates) < 2:
        raise ValueError("唯一日期数量不足，无法进行训练/测试拆分。")

    split_idx = int(len(unique_dates) * train_ratio)
    split_idx = min(max(split_idx, 1), len(unique_dates) - 1)
    split_date = pd.Timestamp(unique_dates[split_idx])

    train_df = df[df["date"] < split_date].copy()
    test_df = df[df["date"] >= split_date].copy()

    print("\n--- 时序拆分 ---")
    print(f"  拆分日期: {split_date.date()}")
    print(f"  训练日期范围: {train_df['date'].min().date()} ~ {train_df['date'].max().date()}")
    print(f"  测试日期范围: {test_df['date'].min().date()} ~ {test_df['date'].max().date()}")
    print(f"  训练集: {len(train_df)} 条")
    print(f"  测试集: {len(test_df)} 条")
    return train_df, test_df, split_date


def get_numeric_cols(df):
    """获取数值列，排除标识列。"""
    return [
        col for col in df.columns
        if col not in IDENTIFIER_COLS and pd.api.types.is_numeric_dtype(df[col])
    ]


def fill_missing_sequential(train_df, test_df):
    """
    仅用过去信息做填补：
    1. 先按站点时间顺序前向填补
    2. 再用训练集站点中位数兜底
    3. 最后用训练集全局中位数兜底
    """
    numeric_cols = get_numeric_cols(train_df)
    print("\n--- 缺失值处理（顺序填补） ---")

    before_train_missing = int(train_df[numeric_cols].isna().sum().sum())
    before_test_missing = int(test_df[numeric_cols].isna().sum().sum())
    print(f"  训练集缺失值: {before_train_missing}")
    print(f"  测试集缺失值: {before_test_missing}")

    station_medians = train_df.groupby("station")[numeric_cols].median(numeric_only=True)
    global_medians = train_df[numeric_cols].median(numeric_only=True)

    combined = pd.concat(
        [train_df.assign(_split="train"), test_df.assign(_split="test")],
        ignore_index=True,
    ).sort_values(["station", "date"]).reset_index(drop=True)

    combined[numeric_cols] = combined.groupby("station")[numeric_cols].ffill()

    for station, idx in combined.groupby("station").groups.items():
        idx = list(idx)
        fill_values = station_medians.loc[station] if station in station_medians.index else global_medians
        combined.loc[idx, numeric_cols] = combined.loc[idx, numeric_cols].fillna(fill_values)

    combined[numeric_cols] = combined[numeric_cols].fillna(global_medians)

    after_missing = int(combined[numeric_cols].isna().sum().sum())
    print(f"  填补后剩余缺失值: {after_missing}")

    train_filled = combined[combined["_split"] == "train"].drop(columns="_split").copy()
    test_filled = combined[combined["_split"] == "test"].drop(columns="_split").copy()
    return train_filled, test_filled


def compute_humidity(df):
    """通过 Magnus 公式从露点温度推算相对湿度。"""
    print("\n--- 推算相对湿度 (Magnus公式) ---")
    a, b = 17.27, 237.7

    def magnus_rh(temp, dewpoint):
        gamma_t = (a * temp) / (b + temp)
        gamma_d = (a * dewpoint) / (b + dewpoint)
        rh = 100 * np.exp(gamma_d - gamma_t)
        return np.clip(rh, 0, 100)

    df = df.copy()
    df["relative_humidity_mean"] = magnus_rh(
        df["temperature_2m_mean"].values,
        df["dewpoint_2m_mean"].values,
    ).round(1)
    df["relative_humidity_max"] = magnus_rh(
        df["temperature_2m_min"].values,
        df["dewpoint_2m_max"].values,
    ).round(1)
    df["relative_humidity_min"] = magnus_rh(
        df["temperature_2m_max"].values,
        df["dewpoint_2m_min"].values,
    ).round(1)

    print(f"  relative_humidity_mean: 均值={df['relative_humidity_mean'].mean():.1f}%")
    print(f"  relative_humidity_max:  均值={df['relative_humidity_max'].mean():.1f}%")
    print(f"  relative_humidity_min:  均值={df['relative_humidity_min'].mean():.1f}%")
    return df


def feature_engineering(df):
    """构造时间、滚动窗口与物候等特征。"""
    print("\n--- 特征工程 ---")
    df = df.sort_values(["station", "date"]).reset_index(drop=True).copy()

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["leaf_wetness_hours"] = np.clip(
        ((df["relative_humidity_mean"] - 70) * 0.3 + df["precipitation_hours"].fillna(0) * 0.8).round(0),
        0,
        24,
    ).astype(int)
    df["temp_range"] = (df["temperature_2m_max"] - df["temperature_2m_min"]).round(1)

    feature_frames = []
    for station, station_df in df.groupby("station", sort=False):
        station_df = station_df.sort_values("date").copy()

        for window in [3, 5, 7]:
            station_df[f"temp_mean_{window}d"] = (
                station_df["temperature_2m_mean"].rolling(window, min_periods=1).mean().round(1)
            )
            station_df[f"rh_mean_{window}d"] = (
                station_df["relative_humidity_mean"].rolling(window, min_periods=1).mean().round(1)
            )
            station_df[f"precip_sum_{window}d"] = (
                station_df["precipitation_sum"].rolling(window, min_periods=1).sum().round(1)
            )
            station_df[f"leaf_wet_{window}d"] = (
                station_df["leaf_wetness_hours"].rolling(window, min_periods=1).sum()
            )

        suitable = (
            station_df["temperature_2m_mean"].between(18, 27)
            & station_df["relative_humidity_mean"].between(70, 95)
            & (station_df["precipitation_sum"] < 20)
        ).astype(int)

        consecutive_days = []
        counter = 0
        for flag in suitable.values:
            counter = counter + 1 if flag else 0
            consecutive_days.append(counter)
        station_df["consecutive_suitable_days"] = consecutive_days
        station_df["phenology_sensitivity"] = station_df["month"].map(PHENOLOGY_MAP)

        feature_frames.append(station_df)

    df = pd.concat(feature_frames, ignore_index=True)
    print("  [OK] 时间特征: year, month, day_of_year")
    print("  [OK] 叶面湿润时数: leaf_wetness_hours")
    print("  [OK] 滚动窗口: temp/rh/precip/leaf_wet x 3d/5d/7d")
    print("  [OK] 连续适宜天数: consecutive_suitable_days")
    print("  [OK] 物候敏感性: phenology_sensitivity")
    return df


def generate_labels(df):
    """基于流行病学规则生成风险代理标签。"""
    print("\n--- 生成风险代理标签 ---")
    df = df.copy()
    scores = np.zeros(len(df), dtype=float)

    temperature = df["temperature_2m_mean"].values
    scores += np.exp(-0.5 * ((temperature - 22.5) / 4.0) ** 2) * 25

    humidity = df["relative_humidity_mean"].values
    scores += np.exp(-0.5 * ((humidity - 82) / 8.0) ** 2) * 20

    precipitation = df["precipitation_sum"].fillna(0).values
    precip_score = np.where(
        precipitation == 0, 0.3,
        np.where(
            precipitation < 5, 0.8,
            np.where(precipitation < 15, 0.5, np.where(precipitation < 30, 0.2, 0.05)),
        ),
    )
    scores += precip_score * 10

    windspeed = df["windspeed_10m_max"].fillna(8).values
    scores += np.exp(-0.5 * ((windspeed - 8) / 5.0) ** 2) * 8

    scores += np.clip(df["leaf_wet_7d"].fillna(0).values / 80.0, 0, 1) * 15
    scores += np.clip(df["consecutive_suitable_days"].values / 7.0, 0, 1) * 15
    scores += df["phenology_sensitivity"].values * 20
    scores += np.clip((df["temp_range"].fillna(5).values - 5) / 10.0, 0, 1) * 5

    scores = np.clip(scores, 0, 120)
    labels = np.zeros(len(df), dtype=int)
    labels[scores >= 40] = 1
    labels[scores >= 60] = 2
    labels[scores >= 78] = 3

    df["disease_risk_score"] = np.round(scores, 2)
    df["disease_level"] = labels
    df["disease_level_name"] = pd.Series(labels).map(LABEL_NAME_MAP).values

    print("  标签性质: 基于气象与规则模型生成的风险代理标签，不是真实田间病情调查值")
    print("  分级阈值: <40=无病 | 40-60=轻度 | 60-78=中度 | >=78=重度")
    for level, name in LABEL_NAME_MAP.items():
        count = int((df["disease_level"] == level).sum())
        print(f"  {name}: {count:>6} ({count / len(df) * 100:.1f}%)")

    return df


def get_model_feature_cols(df):
    """与训练脚本保持一致的模型特征列。"""
    drop_cols = {"station", "date", "year"} | LABEL_COLS
    return [col for col in df.columns if col not in drop_cols]


def build_dataset_metadata(df, train_df, test_df, split_date):
    """构建供训练脚本和 Web 页面使用的数据集元信息。"""
    feature_cols = get_model_feature_cols(df)
    prediction_defaults = {}
    for col, value in train_df[feature_cols].median(numeric_only=True).to_dict().items():
        prediction_defaults[col] = round(float(value), 4)

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "train_ratio": TRAIN_RATIO,
        "split_strategy": "按唯一日期做时序拆分，避免同一天同时进入训练集和测试集",
        "split_date": split_date.strftime("%Y-%m-%d"),
        "missing_value_strategy": (
            "按站点时间顺序前向填补；若仍缺失，则使用训练集站点中位数；"
            "最后使用训练集全局中位数兜底"
        ),
        "label_strategy": "rule_based_proxy_label",
        "label_note": "当前发病等级标签由气象因子和流行病学规则自动生成，属于风险代理标签。",
        "label_limitations": [
            "不是田间实测病情指数或叶片调查值",
            "更适合用于方法验证和风险分级，不宜直接宣称为真实发病率预测",
            "后续应使用田间病害调查数据对规则标签进行替换或校正",
        ],
        "prediction_defaults": prediction_defaults,
        "prediction_form": {
            "required_inputs": [
                "obs_date",
                "temperature_2m_mean",
                "relative_humidity_mean",
                "leaf_wetness_hours",
                "precip_sum_7d",
                "consecutive_suitable_days",
            ],
            "server_side_derived_fields": [
                "month",
                "day_of_year",
                "phenology_sensitivity",
                "temp_mean_3d/5d/7d",
                "rh_mean_3d/5d/7d",
                "precip_sum_3d/5d",
                "leaf_wet_3d/5d/7d",
            ],
        },
        "station_count": int(df["station"].nunique()),
        "record_count": int(len(df)),
        "train_count": int(len(train_df)),
        "test_count": int(len(test_df)),
        "date_range": {
            "start": df["date"].min().strftime("%Y-%m-%d"),
            "end": df["date"].max().strftime("%Y-%m-%d"),
        },
        "model_feature_count": len(feature_cols),
        "stations": sorted(df["station"].unique().tolist()),
    }
    return metadata


def save_splits(df, train_df, test_df, output_dir, metadata):
    """保存完整数据集、训练集、测试集与元信息。"""
    print("\n--- 保存数据集 ---")
    os.makedirs(output_dir, exist_ok=True)

    full_path = os.path.join(output_dir, "full_dataset.csv")
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    meta_path = os.path.join(output_dir, "dataset_metadata.json")

    def to_output_frame(source_df):
        output_df = source_df.copy()
        output_df["date"] = output_df["date"].dt.strftime("%Y-%m-%d")
        return output_df

    to_output_frame(df).to_csv(full_path, index=False, encoding="utf-8-sig")
    to_output_frame(train_df).to_csv(train_path, index=False, encoding="utf-8-sig")
    to_output_frame(test_df).to_csv(test_path, index=False, encoding="utf-8-sig")
    with open(meta_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"  完整数据集: {full_path} ({len(df)} 条)")
    print(f"  训练集:     {train_path} ({len(train_df)} 条)")
    print(f"  测试集:     {test_path} ({len(test_df)} 条)")
    print(f"  元信息:     {meta_path}")


def print_summary(df, train_df, test_df, metadata):
    """打印最终概览。"""
    print("\n" + "=" * 60)
    print("  数据集概览")
    print("=" * 60)
    print(f"  站点数:       {metadata['station_count']}")
    print(f"  时间范围:     {metadata['date_range']['start']} ~ {metadata['date_range']['end']}")
    print(f"  总记录数:     {metadata['record_count']}")
    print(f"  模型特征数:   {metadata['model_feature_count']}")
    print(f"  训练集:       {len(train_df)} 条")
    print(f"  测试集:       {len(test_df)} 条")
    print(f"  拆分日期:     {metadata['split_date']}")
    print(f"  标签说明:     {metadata['label_note']}")
    print("=" * 60)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = base_dir
    output_dir = os.path.join(data_dir, "processed")

    print("=" * 60)
    print("  橡胶树白粉病 - 数据清洗整合")
    print("=" * 60)

    print("\n--- 加载原始数据 ---")
    df = load_raw_data(data_dir)
    if df.empty:
        return

    df = standardize_data(df)
    train_raw, test_raw, split_date = split_by_unique_dates(df)
    train_filled, test_filled = fill_missing_sequential(train_raw, test_raw)

    df = pd.concat([train_filled, test_filled], ignore_index=True).sort_values(["station", "date"]).reset_index(drop=True)
    df = compute_humidity(df)
    df = feature_engineering(df)
    df = generate_labels(df)

    train_df = df[df["date"] < split_date].copy()
    test_df = df[df["date"] >= split_date].copy()
    metadata = build_dataset_metadata(df, train_df, test_df, split_date)

    save_splits(df, train_df, test_df, output_dir, metadata)
    print_summary(df, train_df, test_df, metadata)


if __name__ == "__main__":
    main()
