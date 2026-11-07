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

# 確保輸出目錄存在
os.makedirs('flood/images', exist_ok=True)

DB_PATH = 'flood/tidehistory.db'
JSON_PATH = 'flood/tidedata.json'
IMG_PATH = 'flood/images/tide.png'

MACAU_TZ = ZoneInfo('Asia/Macau')

plt.rcParams['font.sans-serif'] = [
    'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK TC', 
    'Noto Sans CJK SC', 'PingFang SC', 'Microsoft YaHei', 'sans-serif'
]
plt.rcParams['axes.unicode_minus'] = False

API_URL = "https://dsama.apigateway.data.gov.mo/currentTideXmlApi"
API_HEADERS = {
    "Authorization": "APPCODE 09d43a591fba407fb862412970667de4",
    "User-Agent": "Mozilla/5.0"
}

GRID_BORDER_COLOR = '#8DA59A'
TEXT_COLOR = '#334155'
LINE_COLOR = '#0284C7'
ALERT_LINE_COLOR = '#EF4444'

# ==========================================
# 資料庫管理
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS tide_history
                 (record_time TEXT PRIMARY KEY, 
                  timestamp REAL, 
                  record_tide_cm REAL,
                  record_tide_m REAL)''')
    conn.commit()
    conn.close()

def get_last_record_from_db():
    if not os.path.exists(DB_PATH):
        return ""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    c = conn.cursor()
    try:
        c.execute("SELECT record_time FROM tide_history ORDER BY timestamp DESC LIMIT 1")
        row = c.fetchone()
        return row[0] if row else ""
    except Exception:
        return ""
    finally:
        conn.close()

def save_to_db(record_time, timestamp_val, tide_cm, tide_m):
    conn = sqlite3.connect(DB_PATH, timeout=10)
    c = conn.cursor()
    cutoff_ts = timestamp_val - (7 * 86400)
    c.execute("DELETE FROM tide_history WHERE timestamp < ?", (cutoff_ts,))
    c.execute('''INSERT OR REPLACE INTO tide_history 
                 (record_time, timestamp, record_tide_cm, record_tide_m) 
                 VALUES (?, ?, ?, ?)''', (record_time, timestamp_val, tide_cm, tide_m))
    conn.commit()
    conn.close()

# ==========================================
# API 擷取與 XML 解析
# ==========================================
def fetch_tide_xml():
    try:
        res = requests.get(API_URL, headers=API_HEADERS, timeout=12)
        res.encoding = 'utf-8'
        if res.status_code != 200:
            print(f"❌ 請求 DSAMA 潮汐 API 失敗，HTTP Status: {res.status_code}")
            return None
        return ET.fromstring(res.content)
    except Exception as e:
        print(f"❌ 請求 DSAMA 潮汐 API 異常: {e}")
        return None

def parse_tide_xml(root):
    if root is None:
        return None

    item = root.find('.//item')
    if item is None:
        print("⚠️ XML 中未找到 <item> 節點")
        return None

    date_str = item.findtext('date', '').strip()
    time_str = item.findtext('time', '').strip()
    raw_tide = item.findtext('recordTide', '').strip()

    if not date_str or not time_str or not raw_tide:
        print("⚠️ 潮汐數據欄位不完整")
        return None

    dt_str = f"{date_str.replace('/', '-')} {time_str}"
    try:
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M").replace(tzinfo=MACAU_TZ)
        ts = dt.timestamp()
        clean_time_str = dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        dt = datetime.now(MACAU_TZ)
        ts = dt.timestamp()
        clean_time_str = dt.strftime("%Y-%m-%d %H:%M")

    try:
        tide_cm = float(raw_tide)
        tide_m = round(tide_cm / 100.0, 2)
    except ValueError:
        tide_cm = None
        tide_m = None

    return {
        "record_time": clean_time_str,
        "timestamp": ts,
        "record_tide_cm": tide_cm,
        "record_tide_m": tide_m
    }

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
# 繪製 24 小時潮位時序趨勢圖 (含 3.1m 紅線)
# ==========================================
def render_tide_chart(latest_info):
    conn = sqlite3.connect(DB_PATH, timeout=10)
    df = pd.read_sql_query("SELECT * FROM tide_history ORDER BY timestamp ASC", conn)
    conn.close()

    if df.empty:
        print("⚠️ tide_history 資料表為空，略過圖表生成。")
        return

    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Macau').dt.tz_localize(None)

    st_record_time = latest_info['record_time']
    st_timestamp = latest_info['timestamp']

    x_end = datetime.fromtimestamp(st_timestamp, tz=MACAU_TZ).replace(tzinfo=None)
    x_start = x_end - timedelta(hours=24)

    df_24h = df[df['datetime'] >= x_start]
    valid_series = df_24h['record_tide_m'].dropna() if not df_24h.empty else df['record_tide_m'].dropna()

    # 計算動態 Y 軸刻度，並確保 plot_ymax 至少容納 3.1m 紅線
    if not valid_series.empty:
        data_min = valid_series.min()
        data_max = max(valid_series.max(), 3.1)
        v_range = data_max - data_min

        if v_range < 0.8:
            step = 0.1
        elif v_range < 2.0:
            step = 0.2
        else:
            step = 0.5

        plot_ymin = max(0.0, np.floor((data_min - 0.05) / step) * step)
        plot_ymax = np.ceil((data_max + 0.1) / step) * step
        if plot_ymin >= plot_ymax:
            plot_ymax = plot_ymin + step * 2
    else:
        plot_ymin, plot_ymax, step = 0.0, 3.5, 0.5

    fig, ax = plt.subplots(figsize=(15, 6.5), dpi=120)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # 1. 繪製 3.1m 紅色警戒線
    if plot_ymin <= 3.1 <= plot_ymax:
        ax.axhline(3.1, color=ALERT_LINE_COLOR, linestyle='--', linewidth=1.4, alpha=0.9, zorder=2)
        ax.text(x_end, 3.1, " 3.10m", color=ALERT_LINE_COLOR, va='center', ha='left', fontsize=10.5, fontweight='bold')
        ax.text(x_start + timedelta(minutes=15), 3.13, "警戒潮位 (3.1m)", color=ALERT_LINE_COLOR, 
                fontsize=9.5, fontweight='bold', alpha=0.9, va='bottom', zorder=2)

    # 2. 潮位折線與陰影填充
    plot_df = insert_nans_on_gap(df, 'record_tide_m', max_gap_minutes=25)
    if not plot_df['record_tide_m'].dropna().empty:
        ax.plot(plot_df['datetime'], plot_df['record_tide_m'], color=LINE_COLOR, linewidth=2.2, antialiased=True, zorder=4, label='實時潮高 Real-time Tide')
        ax.fill_between(plot_df['datetime'], plot_df['record_tide_m'], plot_ymin, color=LINE_COLOR, alpha=0.15, zorder=3)
        ax.scatter(df['datetime'], df['record_tide_m'], color='#0369A1', s=16, alpha=0.85, edgecolors='none', zorder=5)
    else:
        ax.text(0.5, 0.5, '暫無潮位數據 No Data', transform=ax.transAxes, fontsize=18, color='gray', ha='center', va='center')

    # 3. 坐標軸與動態刻度設置
    ax.set_xlim(x_start, x_end)
    ax.set_ylim(bottom=plot_ymin, top=plot_ymax)
    y_ticks = np.arange(plot_ymin, plot_ymax + step * 0.1, step)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y:.2f}m" for y in y_ticks], fontsize=13, color=LINE_COLOR, family='monospace')

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

    ax.set_ylabel('實時潮高 Tide Height (m)', fontsize=15, color=LINE_COLOR, labelpad=12)
    ax.set_xlabel('澳門時間(時) Local Time (Hour)', fontsize=15, color=TEXT_COLOR, labelpad=14)

    header_title = "澳門實時潮汐"
    header_update_text = f"更新時間 Update Time:{st_record_time.replace('-', '/')}"

    ax.text(0.0, 1.02, header_title, transform=ax.transAxes, fontsize=15, va='bottom', ha='left', color=TEXT_COLOR, fontweight='bold')
    ax.text(1.0, 1.02, header_update_text, transform=ax.transAxes, fontsize=15, va='bottom', ha='right', color=TEXT_COLOR)

    plt.tight_layout()
    plt.savefig(IMG_PATH, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ 實時潮汐折線圖已儲存至：{IMG_PATH}")

# ==========================================
# 主流程
# ==========================================
def main():
    init_db()
    last_record_time = get_last_record_from_db()
    print(f"🚀 開始執行 DSAMA 實時潮汐同步，資料庫最新記錄：'{last_record_time}'")

    root = fetch_tide_xml()
    if root is None:
        sys.exit(1)

    tide_data = parse_tide_xml(root)
    if not tide_data or tide_data.get('record_tide_cm') is None:
        print("⚠️ 解析潮汐 XML 失敗或無有效數值。")
        sys.exit(1)

    current_rec_time = tide_data['record_time']
    if current_rec_time == last_record_time and last_record_time != "":
        print(f"ℹ️ 潮汐數據尚未更新 (當前記錄: {current_rec_time})，跳過發布與繪圖。")
        sys.exit(2)

    print(f"🔔 檢測到新潮位數據: {current_rec_time} -> {tide_data['record_tide_cm']} cm ({tide_data['record_tide_m']} m)")

    # 1. 寫入 SQLite
    save_to_db(
        tide_data['record_time'],
        tide_data['timestamp'],
        tide_data['record_tide_cm'],
        tide_data['record_tide_m']
    )

    # 2. 輸出 JSON
    payload = {
        "update_time": tide_data['record_time'],
        "timestamp": tide_data['timestamp'],
        "tide_cm": tide_data['record_tide_cm'],
        "tide_m": tide_data['record_tide_m'],
        "source": "DSAMA"
    }
    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"✅ 潮位 JSON 已輸出至：{JSON_PATH}")

    # 3. 繪製並儲存圖表至 flood/images/tide.png
    render_tide_chart(tide_data)
    print("🎉 潮位同步與繪圖流程順利完成！")
    sys.exit(0)

if __name__ == '__main__':
    main()
