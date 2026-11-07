import os
import sys
import json
import warnings
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import xml.etree.ElementTree as ET

import requests
import numpy as np
import pandas as pd
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# 建立輸出目錄
os.makedirs('flood/images', exist_ok=True)

DB_PATH = 'flood/floodhistory.db'
JSON_PATH = 'flood/flooddata.json'

MACAU_TZ = ZoneInfo('Asia/Macau')

plt.rcParams['font.sans-serif'] = [
    'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK TC', 
    'Noto Sans CJK SC', 'PingFang SC', 'Microsoft YaHei', 'sans-serif'
]
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 水位站警戒區間與顏色配置
# 0.5 - 1.0 yellow
# 1.0 - 1.5 orange
# 1.5 - 2.5 red
# 2.5 +     black
# ==========================================
FLOOD_SPANS = [
    (2.5, 6.0,  "#CBD5E1", "#0F172A", "黑色風暴潮 (≥2.5m)"),
    (1.5, 2.5,  "#FEE2E2", "#DC2626", "紅色風暴潮 (1.5-2.5m)"),
    (1.0, 1.5,  "#FFEDD5", "#EA580C", "橙色風暴潮 (1.0-1.5m)"),
    (0.5, 1.0,  "#FEF9C3", "#CA8A04", "黃色風暴潮 (0.5-1.0m)")
]

GRID_BORDER_COLOR = '#8DA59A'
TEXT_COLOR = '#334155'
LINE_COLOR = '#0284C7'

# ==========================================
# 資料庫管理
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS flood_history
                 (time_str TEXT, timestamp REAL, station_code TEXT, 
                  station_name TEXT, water_level REAL,
                  PRIMARY KEY (time_str, station_code))''')
    conn.commit()
    conn.close()

def get_last_pubdate_from_db():
    if not os.path.exists(DB_PATH):
        return ""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    c = conn.cursor()
    try:
        c.execute("SELECT time_str FROM flood_history ORDER BY timestamp DESC LIMIT 1")
        row = c.fetchone()
        return row[0] if row else ""
    except Exception:
        return ""
    finally:
        conn.close()

def save_to_db(stations_data, pub_time_str, pub_timestamp_val):
    conn = sqlite3.connect(DB_PATH, timeout=10)
    c = conn.cursor()
    cutoff_ts = pub_timestamp_val - (7 * 86400)
    c.execute("DELETE FROM flood_history WHERE timestamp < ?", (cutoff_ts,))
    for code, data in stations_data.items():
        row = (pub_time_str, pub_timestamp_val, code, 
               data.get('station_name', code), data.get('water_level'))
        c.execute('''INSERT OR REPLACE INTO flood_history 
                     (time_str, timestamp, station_code, station_name, water_level) 
                     VALUES (?, ?, ?, ?, ?)''', row)
    conn.commit()
    conn.close()

# ==========================================
# XML 資料擷取與解析 (依據實際 XML 結構)
# ==========================================
def fetch_smg_flood_xml():
    url = "https://xml.dsmg.gov.mo/c_ftgms.xml"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers, timeout=12)
        res.encoding = 'utf-8'
        return ET.fromstring(res.content)
    except Exception as e:
        print(f"❌ 請求 SMG 水位 XML 失敗: {e}")
        return None

def parse_xml_timestamp(root):
    pub_node = root.find('.//System/SysPubdate')
    if pub_node is None:
        pub_node = root.find('.//SysPubdate') or root.find('.//SysPubDate')
        
    if pub_node is not None and pub_node.text:
        raw_time = pub_node.text.strip()
        for fmt in ["%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"]:
            try:
                dt = datetime.strptime(raw_time, fmt).replace(tzinfo=MACAU_TZ)
                clean_str = dt.strftime("%Y-%m-%d %H:%M")
                return clean_str, dt.timestamp()
            except ValueError:
                continue
    fallback_dt = datetime.now(MACAU_TZ)
    return fallback_dt.strftime("%Y-%m-%d %H:%M"), fallback_dt.timestamp()

def parse_flood_xml(root):
    if root is None:
        return "", 0.0, {}

    pub_time_str, pub_timestamp = parse_xml_timestamp(root)
    stations_data = {}

    def parse_float(val_str):
        if val_str is None:
            return None
        try:
            return float(str(val_str).strip())
        except (ValueError, TypeError):
            return None

    # 針對 <WLReport><station code="..."> 結構抓取
    for st in root.findall('.//station'):
        # 1. 優先從屬性 code 取出代碼 (如 code="LSI")，若無則降級找子標籤
        code = st.get('code')
        if not code:
            code_node = st.find('stationcode') or st.find('StationCode') or st.find('code')
            if code_node is not None and code_node.text:
                code = code_node.text.strip()

        if not code:
            continue
        code = code.strip()

        # 2. 獲取站名
        name_node = st.find('stationname') or st.find('StationName') or st.find('name')
        name = name_node.text.strip() if name_node is not None and name_node.text else code

        # 3. 獲取站點記錄時間 RecordTime
        rec_time_node = st.find('RecordTime')
        rec_time = rec_time_node.text.strip() if rec_time_node is not None and rec_time_node.text else pub_time_str

        # 4. 提取水位數值 (Waterlevel/Value)
        water_val = None
        wl_node = st.find('Waterlevel') or st.find('waterlevel')
        if wl_node is not None:
            val_node = wl_node.find('Value') or wl_node.find('value') or wl_node.find('dValue')
            if val_node is not None and val_node.text:
                water_val = parse_float(val_node.text)

        # 容錯降級搜尋
        if water_val is None:
            for tag in ['Value', 'value', 'waterlevel']:
                node = st.find(f'.//{tag}')
                if node is not None and node.text:
                    parsed = parse_float(node.text)
                    if parsed is not None:
                        water_val = parsed
                        break

        stations_data[code] = {
            "station_code": code,
            "station_name": name,
            "record_time": rec_time,
            "water_level": water_val
        }

    return pub_time_str, pub_timestamp, stations_data

# ==========================================
# 繪圖斷線輔助函式
# ==========================================
def insert_nans_on_gap(df, col, max_gap_minutes=25):
    valid_df = df.dropna(subset=[col]).sort_values('datetime').reset_index(drop=True)
    if len(valid_df) < 2:
        return valid_df

    new_rows = []
    for i in range(len(valid_df) - 1):
        curr_row = valid_df.iloc[i]
        next_row = valid_df.iloc[i + 1]
        new_rows.append(curr_row)

        diff = (next_row['datetime'] - curr_row['datetime']).total_seconds() / 60.0
        if diff > max_gap_minutes:
            gap_dt = curr_row['datetime'] + timedelta(minutes=1)
            dummy_row = curr_row.copy()
            dummy_row['datetime'] = gap_dt
            dummy_row[col] = np.nan
            new_rows.append(dummy_row)

    new_rows.append(valid_df.iloc[-1])
    return pd.DataFrame(new_rows).reset_index(drop=True)

# ==========================================
# 繪製各水位站趨勢圖
# ==========================================
def render_flood_charts(pub_time_str, pub_timestamp):
    conn = sqlite3.connect(DB_PATH, timeout=10)
    df = pd.read_sql_query("SELECT * FROM flood_history ORDER BY timestamp ASC", conn)
    conn.close()

    if df.empty:
        print("⚠️ flood_history 為空，略過圖表渲染。")
        return

    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Macau').dt.tz_localize(None)

    x_end = datetime.fromtimestamp(pub_timestamp, tz=MACAU_TZ).replace(tzinfo=None)
    x_start = x_end - timedelta(hours=24)
    header_update_text = f"更新時間 Update Time:{pub_time_str.replace('-', '/')}"

    unique_stations = df[['station_code', 'station_name']].drop_duplicates().to_dict('records')

    # 計算全站過去 24 小時動態 Y 軸刻度
    df_24h = df[df['datetime'] >= x_start]
    valid_levels = df_24h['water_level'].dropna() if not df_24h.empty else df['water_level'].dropna()
    max_level = valid_levels.max() if not valid_levels.empty else 1.0
    min_level = valid_levels.min() if not valid_levels.empty else 0.0

    plot_ymin = min(0.0, np.floor(min_level * 2.0) / 2.0)
    calc_ymax = max(1.5, np.ceil(max_level * 2.0) / 2.0 + 0.2)
    step = 0.2 if calc_ymax <= 2.0 else 0.5
    plot_ymax = np.ceil(calc_ymax / step) * step

    for st in unique_stations:
        st_code = st['station_code']
        st_name = st['station_name']
        st_df = df[df['station_code'] == st_code].copy()
        if st_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(15, 6.5), dpi=120)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        # 1. 著色背景警告區間
        for low, high, bg_color, text_c, label_text in FLOOD_SPANS:
            if low < plot_ymax and high > plot_ymin:
                eff_low = max(low, plot_ymin)
                eff_high = min(high, plot_ymax)
                ax.axhspan(eff_low, eff_high, facecolor=bg_color, alpha=0.35, zorder=1)
                
                # 門檻標籤與輔助虛線
                if eff_low > plot_ymin:
                    ax.axhline(eff_low, color=text_c, linestyle='--', linewidth=0.9, alpha=0.6, zorder=2)
                    ax.text(x_end, eff_low, f" {eff_low:.1f}m", color=text_c, va='center', ha='left', fontsize=9.5, fontweight='bold')
                
                mid_y = (eff_low + eff_high) / 2.0
                ax.text(x_start + timedelta(minutes=15), mid_y, label_text,
                        color=text_c, fontsize=9.5, fontweight='bold', alpha=0.8, va='center', zorder=2)

        # 2. 繪製折線
        plot_df = insert_nans_on_gap(st_df, 'water_level', max_gap_minutes=25)
        if not plot_df['water_level'].dropna().empty:
            ax.plot(plot_df['datetime'], plot_df['water_level'], color=LINE_COLOR, linewidth=2.0, antialiased=True, zorder=4, label='水位 Water Level')
            ax.scatter(st_df['datetime'], st_df['water_level'], color='#0369A1', s=16, alpha=0.8, edgecolors='none', zorder=5)
        else:
            ax.text(0.5, 0.5, '暫無水位數據 No Data', transform=ax.transAxes, fontsize=18, color='gray', ha='center', va='center')

        # 3. 坐標軸與刻度
        ax.set_xlim(x_start, x_end)
        ax.set_ylim(plot_ymin, plot_ymax)
        y_ticks = np.arange(plot_ymin, plot_ymax + step * 0.1, step)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{y:.1f}m" for y in y_ticks], fontsize=13, color=LINE_COLOR, family='monospace')

        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[15, 30, 45]))

        def format_hour_ticks(x, pos=None):
            dt = mdates.num2date(x)
            hour_str = f"{dt.strftime('%H')}H"
            return f"{hour_str}\n{dt.strftime('%d %b')}" if dt.hour == 0 else hour_str

        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_hour_ticks))
        plt.setp(ax.get_xticklabels(), fontsize=13, color=TEXT_COLOR, family='monospace')

        ax.grid(True, which='major', linestyle=':', linewidth=1.4, color=GRID_BORDER_COLOR, alpha=0.85)
        ax.tick_params(axis='both', which='major', length=6, width=1.1, color=GRID_BORDER_COLOR, direction='out')
        ax.tick_params(axis='x', which='minor', length=3, width=0.8, color=GRID_BORDER_COLOR, direction='out')

        for spine in ax.spines.values():
            spine.set_color(GRID_BORDER_COLOR)
            spine.set_linewidth(1.3)

        ax.set_ylabel('水位 Water Level (m)', fontsize=15, color=LINE_COLOR, labelpad=12)
        ax.set_xlabel('澳門時間(時) Local Time (Hour)', fontsize=15, color=TEXT_COLOR, labelpad=14)

        header_title = f"{st_name} ({st_code}) 水位趨勢"
        ax.text(0.0, 1.02, header_title, transform=ax.transAxes, fontsize=15, va='bottom', ha='left', color=TEXT_COLOR, fontweight='bold')
        ax.text(1.0, 1.02, header_update_text, transform=ax.transAxes, fontsize=15, va='bottom', ha='right', color=TEXT_COLOR)

        plt.tight_layout()
        img_out = f"flood/images/{st_code}.png"
        plt.savefig(img_out, dpi=120, bbox_inches='tight')
        plt.close(fig)

    print("✅ 全部水位站趨勢圖繪製完成！")

# ==========================================
# 主流程
# ==========================================
def main():
    init_db()
    last_pubdate = get_last_pubdate_from_db()
    print(f"🚀 開始執行 DSMG 水位監測同步，資料庫最新記錄：'{last_pubdate}'")

    root = fetch_smg_flood_xml()
    if root is None:
        print("❌ 無法取得水位 XML 數據。")
        sys.exit(1)

    pub_time_str, pub_timestamp, stations_data = parse_flood_xml(root)

    if not stations_data:
        print("⚠️ XML 未包含有效的站點水位數據。")
        sys.exit(1)

    if pub_time_str == last_pubdate:
        print(f"ℹ️ DSMG 水位 XML 尚未更新 (發布時間: {pub_time_str})，跳過後續步驟。")
        sys.exit(2)

    print(f"🔔 檢測到新水位發布數據: {pub_time_str}，共解析到 {len(stations_data)} 個站點，正在處理...")
    
    # 儲存至 SQLite
    save_to_db(stations_data, pub_time_str, pub_timestamp)

    # 儲存至 JSON
    payload = {
        "update_time": pub_time_str,
        "timestamp": pub_timestamp,
        "stations": stations_data
    }
    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 繪製各站圖片 (儲存至 flood/images/<station code>.png)
    render_flood_charts(pub_time_str, pub_timestamp)
    print("🎉 水位同步與圖片儲存完畢。")
    sys.exit(0)

if __name__ == '__main__':
    main()
