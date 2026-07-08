"""
步骤1：爬取海南橡胶产区真实气象数据

运行后显示站点列表，输入数字选择要爬取的区域。
每个区域的数据保存为单独的CSV文件到 data_csv/ 文件夹中。
可多次运行，逐步积累不同区域的数据。

数据源: Open-Meteo Archive API
"""

import requests
import pandas as pd
import time
import os
import sys

# ============================================================
# 海南橡胶主产区站点配置
# ============================================================
STATIONS = [
    {"id": 1,  "name": "海南_儋州", "lat": 19.52, "lon": 109.58, "desc": "西部 | 海南最大橡胶产区"},
    {"id": 2,  "name": "海南_琼中", "lat": 19.03, "lon": 109.84, "desc": "中部 | 中部山区，海拔较高"},
    {"id": 3,  "name": "海南_万宁", "lat": 18.80, "lon": 110.39, "desc": "东南 | 东南沿海"},
    {"id": 4,  "name": "海南_澄迈", "lat": 19.74, "lon": 110.00, "desc": "北部 | 冬季冷空气影响较大"},
    {"id": 5,  "name": "海南_白沙", "lat": 19.23, "lon": 109.45, "desc": "中西 | 中西部山区"},
    {"id": 6,  "name": "海南_乐东", "lat": 18.75, "lon": 109.17, "desc": "西南 | 西南干旱区"},
    {"id": 7,  "name": "海南_琼海", "lat": 19.26, "lon": 110.47, "desc": "东部 | 东部沿海，降水丰富"},
    {"id": 8,  "name": "海南_定安", "lat": 19.68, "lon": 110.36, "desc": "中北 | 中北部丘陵"},
    {"id": 9,  "name": "海南_屯昌", "lat": 19.35, "lon": 110.10, "desc": "中部 | 中部内陆"},
    {"id": 10, "name": "海南_临高", "lat": 19.91, "lon": 109.69, "desc": "西北 | 西北沿海"},
]

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
DELAY = 10  # 请求间隔(秒)
START_YEAR = 2019
END_YEAR = 2024

DAILY_VARS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "precipitation_sum", "rain_sum", "precipitation_hours",
    "windspeed_10m_max", "windgusts_10m_max", "winddirection_10m_dominant",
    "shortwave_radiation_sum", "et0_fao_evapotranspiration",
    "dewpoint_2m_min", "dewpoint_2m_max", "dewpoint_2m_mean",
]


def request_with_retry(params, max_retries=5):
    """带重试和退避的API请求"""
    for attempt in range(max_retries):
        try:
            r = requests.get(BASE_URL, params=params, timeout=60)
            if r.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"  (API限流, 等待{wait}秒...)", end=" ")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            if "429" in str(e):
                wait = 30 * (attempt + 1)
                print(f"  (API限流, 等待{wait}秒...)", end=" ")
                time.sleep(wait)
                continue
            print(f"  (HTTP错误: {e})")
            return None
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            wait = 15 * (attempt + 1)
            print(f"  (网络异常, 第{attempt+1}次重试...)", end=" ")
            time.sleep(wait)
    print("  (多次重试失败)")
    return None


def fetch_station_data(station):
    """爬取单个站点的全部年份日数据"""
    name = station["name"]
    lat, lon = station["lat"], station["lon"]
    frames = []

    for year in range(START_YEAR, END_YEAR + 1):
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": f"{year}-01-01",
            "end_date": f"{year}-12-31",
            "daily": ",".join(DAILY_VARS),
            "timezone": "Asia/Shanghai",
        }
        print(f"  {year}年 ...", end=" ")
        j = request_with_retry(params)
        if j and "daily" in j:
            df = pd.DataFrame(j["daily"])
            df.rename(columns={"time": "date"}, inplace=True)
            df.insert(0, "station", name)
            frames.append(df)
            print(f"[OK] {len(df)}条")
            # 打印前3条数据预览
            preview = df.head(3)
            for _, row in preview.iterrows():
                print(f"    {row['date']} | 温度:{row['temperature_2m_min']:.1f}~{row['temperature_2m_max']:.1f}°C "
                      f"均温:{row['temperature_2m_mean']:.1f}°C | 降水:{row['precipitation_sum']:.1f}mm "
                      f"| 风速:{row['windspeed_10m_max']:.1f}km/h | 露点:{row['dewpoint_2m_mean']:.1f}°C")
            if len(df) > 3:
                print(f"    ... 共{len(df)}条, 至 {df.iloc[-1]['date']}")
        else:
            print("[FAIL] 失败")
        time.sleep(DELAY)

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def show_menu(data_dir):
    """显示站点选择菜单"""
    # 检查已有数据
    existing = set()
    if os.path.isdir(data_dir):
        for f in os.listdir(data_dir):
            if f.endswith(".csv") and not f.startswith("_"):
                existing.add(f.replace(".csv", ""))

    print("\n" + "=" * 60)
    print("  橡胶树白粉病监测 —— 海南气象数据爬取")
    print("  数据源: Open-Meteo Archive API (免费)")
    print(f"  时间范围: {START_YEAR}-{END_YEAR}")
    print("=" * 60)
    print("\n可选站点:")
    print("-" * 60)

    for s in STATIONS:
        status = " [已有]" if s["name"] in existing else ""
        print(f"  {s['id']:>2}. {s['name']}  {s['desc']}{status}")

    print(f"\n  0. 爬取全部站点")
    print(f"  q. 退出")
    print("-" * 60)


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = base
    os.makedirs(data_dir, exist_ok=True)

    while True:
        show_menu(data_dir)
        choice = input("\n请输入编号 (多个用逗号分隔, 如 1,3,5): ").strip()

        if choice.lower() == "q":
            print("退出。")
            break

        # 解析选择
        if choice == "0":
            selected = STATIONS[:]
        else:
            try:
                ids = [int(x.strip()) for x in choice.split(",")]
                selected = [s for s in STATIONS if s["id"] in ids]
                if not selected:
                    print("[FAIL] 无效编号，请重新输入。")
                    continue
            except ValueError:
                print("[FAIL] 输入格式错误，请输入数字。")
                continue

        # 确认
        print(f"\n将爬取以下 {len(selected)} 个站点:")
        for s in selected:
            print(f"  - {s['name']} ({s['desc']})")
        confirm = input("\n确认开始? (y/n): ").strip().lower()
        if confirm != "y":
            print("已取消。")
            continue

        # 逐站爬取
        success_count = 0
        for s in selected:
            print(f"\n{'='*40}")
            print(f"正在爬取: {s['name']} ({s['desc']})")
            print(f"{'='*40}")

            df = fetch_station_data(s)
            if df.empty:
                print(f"[FAIL] {s['name']} 未获取到数据")
                continue

            # 保存为独立CSV
            filename = f"{s['name']}.csv"
            filepath = os.path.join(data_dir, filename)
            df.to_csv(filepath, index=False, encoding="utf-8-sig")
            print(f"[OK] 已保存: {filepath} ({len(df)}条记录)")
            success_count += 1

        print(f"\n{'='*40}")
        print(f"本次完成: {success_count}/{len(selected)} 个站点")

        # 列出data目录现有文件
        files = [f for f in os.listdir(data_dir)
                 if f.endswith(".csv") and not f.startswith("_")]
        print(f"data_csv/ 目录已有 {len(files)} 个数据文件:")
        for f in sorted(files):
            size = os.path.getsize(os.path.join(data_dir, f))
            print(f"  {f} ({size/1024:.0f} KB)")
        print(f"{'='*40}")

        # 是否继续
        again = input("\n继续爬取其他站点? (y/n): ").strip().lower()
        if again != "y":
            break

    print("\n提示: 爬取完成后，运行 data_process.py 对数据进行清洗整合。")


if __name__ == "__main__":
    main()
