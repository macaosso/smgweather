import os
import io
import sys
import time
import sqlite3
import warnings
from datetime import datetime

import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.interpolate as si
from scipy.ndimage import maximum_filter
import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings("ignore")

# 建立輸出目錄
OUT_DIR = 'rain_nowcast/images'
DB_PATH = 'rain_nowcast/nowcast_meta.db'
os.makedirs(OUT_DIR, exist_ok=True)


# ==========================================
# 資料庫管理 (記錄最新預報時間以避免重複繪圖)
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sys_meta
                 (key TEXT PRIMARY KEY, val TEXT, updated_at REAL)''')
    conn.commit()
    conn.close()


def get_last_nowcast_key():
    if not os.path.exists(DB_PATH):
        return ""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    c = conn.cursor()
    try:
        c.execute("SELECT val FROM sys_meta WHERE key = 'last_nowcast_key'")
        row = c.fetchone()
        return row[0] if row else ""
    except Exception:
        return ""
    finally:
        conn.close()


def save_nowcast_key(key_val):
    conn = sqlite3.connect(DB_PATH, timeout=10)
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO sys_meta (key, val, updated_at)
                 VALUES ('last_nowcast_key', ?, ?)''', (key_val, time.time()))
    conn.commit()
    conn.close()


# ==========================================
# CSV 擷取 (加入時間戳與 no-cache 標頭穿透快取)
# ==========================================
def fetch_hko_rainfall_csv():
    ts_param = int(time.time())
    url = f"https://data.weather.gov.hk/weatherAPI/hko_data/F3/Gridded_rainfall_nowcast_tc.csv?_nocache={ts_param}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
    }
    try:
        res = requests.get(url, headers=headers, timeout=12)
        if res.status_code != 200:
            print(f"❌ 請求失敗，狀態碼：{res.status_code}")
            return None
        df = pd.read_csv(io.BytesIO(res.content), encoding='utf-8-sig')
        df.columns = [col.strip() for col in df.columns]
        return df
    except Exception as e:
        print(f"❌ 請求 HKO 網格降雨預報 CSV 異常: {e}")
        return None


# ==========================================
# 稀疏篩選點位函式，防止文字重疊
# ==========================================
def sample_valid_points(df_sample, lon_col, lat_col, rain_col, min_dist_deg=0.03, min_rain=1.0):
    df_filter = df_sample[df_sample[rain_col] >= min_rain].copy()
    if df_filter.empty:
        return pd.DataFrame()

    selected = []
    lons = df_filter[lon_col].values
    lats = df_filter[lat_col].values
    rains = df_filter[rain_col].values

    for lon, lat, rain in zip(lons, lats, rains):
        too_close = False
        for slon, slat, _ in selected:
            dist = np.sqrt((lon - slon) ** 2 + (lat - slat) ** 2)
            if dist < min_dist_deg:
                too_close = True
                break
        if not too_close:
            selected.append([lon, lat, rain])
    return pd.DataFrame(selected, columns=[lon_col, lat_col, rain_col])


# ==========================================
# 雨量文字顏色映射
# ==========================================
def get_text_color(val):
    if val < 1:
        return "#15438e"
    elif val < 5:
        return "#2fdbeb"
    elif val < 10:
        return "#0F9C5D"
    elif val < 20:
        return "#52ee27"
    elif val < 50:
        return "#ffdd00"
    elif val < 80:
        return "#ff0000"
    elif val < 100:
        return "#4a4747"
    else:
        return "#6C074E"


# ==========================================
# 封裝單張圖繪圖邏輯
# ==========================================
def plot_rain_map(ax, df_frame, lon_col, lat_col, rain_col, valid_dt_obj):
    # 地圖範圍
    lon_min, lon_max = 113.0, 114.4
    lat_min, lat_max = 21.5, 22.75
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # 底圖要素
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1, edgecolor="dimgray")
    ax.add_feature(cfeature.LAND, edgecolor="#959a9f", facecolor="#2d363f")
    ax.add_feature(cfeature.OCEAN, facecolor="#222a35")

    # 網格線
    grid_1deg_lon = np.arange(lon_min, lon_max + 1, 0.02)
    grid_1deg_lat = np.arange(lat_min, lat_max + 1, 0.02)
    major_5deg_lon = np.arange(lon_min, lon_max + 1, 0.1)
    major_5deg_lat = np.arange(lat_min, lat_max + 1, 0.1)

    ax.gridlines(xlocs=grid_1deg_lon, ylocs=grid_1deg_lat, draw_labels=False, linewidth=0.15, color='gray', alpha=0.25, linestyle='--', zorder=-9)
    ax.gridlines(xlocs=major_5deg_lon, ylocs=major_5deg_lat, draw_labels=False, linewidth=0.6, color='gray', alpha=0.4, linestyle='--', zorder=-8)

    gl_label = ax.gridlines(
        xlocs=major_5deg_lon, ylocs=major_5deg_lat,
        draw_labels={"bottom": "x", "left": "y"}, linewidth=0,
        xlabel_style={"size": 6, "color": "white", "alpha": 0.8},
        ylabel_style={"size": 6, "color": "white", "alpha": 0.8}
    )
    gl_label.xpadding = -6
    gl_label.ypadding = -6

    # 澳門雷達圈
    MACAU_LAT = 22.1595
    MACAU_LON = 113.5685
    KM_PER_DEG = 111.32
    labels = ['10 km', '25 km', '50 km', '100 km']
    for km, label in zip([10, 25, 50, 100], labels):
        radius_deg = km / KM_PER_DEG
        circle = plt.Circle(
            (MACAU_LON, MACAU_LAT),
            radius_deg,
            color="#949494",
            fill=False,
            linewidth=0.5,
            alpha=0.5,
            linestyle='--',
            transform=ccrs.PlateCarree()
        )
        ax.add_patch(circle)
        ax.text(
            MACAU_LON,
            MACAU_LAT - radius_deg - 0.025,
            label,
            color='white',
            alpha=0.5,
            fontsize=6,
            ha='center',
            va='top',
            transform=ccrs.PlateCarree()
        )
    ax.plot(MACAU_LON, MACAU_LAT, 'o', color='white', markersize=5)

    cf = None
    # 至少需要 4 個點才能進行 cubic 插值
    if len(df_frame) >= 4:
        xi = np.linspace(df_frame[lon_col].min(), df_frame[lon_col].max(), 400)
        yi = np.linspace(df_frame[lat_col].min(), df_frame[lat_col].max(), 400)
        Xi, Yi = np.meshgrid(xi, yi)

        Zi = si.griddata(
            (df_frame[lon_col], df_frame[lat_col]),
            df_frame[rain_col],
            (Xi, Yi),
            method='cubic'
        )
        Zi[Zi < 0.1] = np.nan

        # 局部峰值檢測
        window_size = 8
        peak_min_threshold = 10.0
        max_filtered = maximum_filter(Zi, size=window_size)
        peak_mask = (Zi == max_filtered) & (~np.isnan(Zi))

        peak_lon_arr = Xi[peak_mask]
        peak_lat_arr = Yi[peak_mask]
        peak_val_arr = Zi[peak_mask]

        valid_peak_idx = peak_val_arr >= peak_min_threshold
        peak_lon_arr = peak_lon_arr[valid_peak_idx]
        peak_lat_arr = peak_lat_arr[valid_peak_idx]
        peak_val_arr = peak_val_arr[valid_peak_idx]

        # 填色雨量場
        levels = [0.1, 1, 5, 10, 20, 50, 80, 100]
        colors = [
            "#15438e", "#2fdbeb", "#0F9C5D", "#52ee27",
            "#ffdd00", "#ff0000", "#4a4747", "#6C074E",
        ]
        cf = ax.contourf(
            Xi, Yi, Zi,
            levels=levels,
            colors=colors,
            extend='max',
            alpha=0.6,
            transform=ccrs.PlateCarree()
        )

        # 稀疏站點雨量標註
        label_df = sample_valid_points(df_frame, lon_col, lat_col, rain_col, min_dist_deg=0.02, min_rain=2.0)
        if not label_df.empty:
            for _, row in label_df.iterrows():
                lon = row[lon_col]
                lat = row[lat_col]
                if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
                    continue
                rain_val = round(row[rain_col])
                txt_color = get_text_color(rain_val)
                ax.text(
                    lon + 0.003, lat + 0.003,
                    f"{rain_val}",
                    fontsize=6,
                    color=txt_color,
                    weight="bold",
                    alpha=0.95,
                    transform=ccrs.PlateCarree()
                )

        # 峰值紅點+標註
        ax.scatter(
            peak_lon_arr, peak_lat_arr,
            c="#ff2222", s=14, zorder=12,
            transform=ccrs.PlateCarree()
        )
        for plon, plat, pval in zip(peak_lon_arr, peak_lat_arr, peak_val_arr):
            if not (lon_min <= plon <= lon_max and lat_min <= plat <= lat_max):
                continue
            p_round = round(pval)
            peak_txt_clr = get_text_color(p_round)
            ax.text(
                plon + 0.005, plat + 0.005,
                f"{p_round}",
                fontsize=7, weight="bold", color=peak_txt_clr,
                alpha=1.0, zorder=13, transform=ccrs.PlateCarree()
            )
    else:
        ax.text(
            0.5, 0.5, "No Significant Rainfall",
            transform=ax.transAxes, ha='center', va='center',
            fontsize=12, color='white', alpha=0.6, zorder=10
        )

    # 左上角 MST 時間方塊
    date_time = valid_dt_obj.strftime("%Y-%m-%d  %H:%M") + " MST"
    ax.text(
        0.05, 0.95, date_time,
        transform=ax.transAxes, ha='left', va='top',
        fontsize=10, color='white', alpha=0.9,
        bbox=dict(facecolor='black', alpha=0.3, pad=3, edgecolor='none'),
        zorder=10
    )

    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    return cf


# ==========================================
# 主流程：擷取 CSV 並獨立繪製 4 張預報圖
# ==========================================
def main():
    init_db()
    last_key = get_last_nowcast_key()
    print(f"🚀 開始執行 HKO 網格降雨預報同步，資料庫上記錄之期數: '{last_key}'")

    df = fetch_hko_rainfall_csv()
    if df is None or df.empty:
        print("❌ 無法取得有效的網格降雨 CSV 數據。")
        sys.exit(1)

    # 自動偵測欄位名稱
    lat_col = next((col for col in df.columns if "緯度" in col), None)
    lon_col = next((col for col in df.columns if "經度" in col), None)
    rain_col = next((col for col in df.columns if "雨量" in col), None)
    time_col = next((c for c in df.columns if "時間" in c and ("完結" in c or "結束" in c)), None)

    if not all([lat_col, lon_col, rain_col, time_col]):
        print(f"⚠️ CSV 欄位缺失，現有欄位：{list(df.columns)}")
        sys.exit(1)

    # 清洗數值欄位
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
    df[rain_col] = pd.to_numeric(df[rain_col], errors='coerce')
    df = df.dropna(subset=[lat_col, lon_col, rain_col, time_col])
    df[time_col] = df[time_col].astype(str).str.strip()

    # 取得排序後的 4 個預報時段
    timestamps = sorted(df[time_col].unique())[:4]
    if not timestamps:
        print("⚠️ CSV 未包含任何有效的時間戳。")
        sys.exit(1)

    current_key = f"{timestamps[0]}_{timestamps[-1]}"
    if current_key == last_key and last_key != "":
        print(f"ℹ️ HKO 預報期數 ({current_key}) 尚未改變，跳過重新繪圖。")
        sys.exit(2)

    print(f"🔔 檢測到新預報期數 ({len(timestamps)} 個時段): {timestamps}")

    for idx, ts_str in enumerate(timestamps, start=1):
        valid_dt = datetime.strptime(ts_str, "%Y%m%d%H%M")
        df_frame = df[(df[time_col] == ts_str) & (df[rain_col] > 0)].copy()

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": ccrs.PlateCarree()})
        plot_rain_map(ax, df_frame, lon_col, lat_col, rain_col, valid_dt)
        plt.subplots_adjust(top=0.94, bottom=0.06, left=0.06, right=0.94)

        img_out = os.path.join(OUT_DIR, f"rain_nowcast_{idx}.png")
        plt.savefig(img_out, dpi=150, bbox_inches='tight', facecolor='#222a35')
        plt.close(fig)
        print(f"✅ 已儲存第 {idx} 張圖：{img_out} ({ts_str})")

    save_nowcast_key(current_key)
    print("🎉 4 張網格降雨預報圖繪製完畢。")
    sys.exit(0)


if __name__ == '__main__':
    main()
