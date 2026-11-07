import os
import sys
import json
import warnings
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

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
os.makedirs('AQI/images', exist_ok=True)

DB_PATH = 'AQI/AQIhistory.db'
JS_PATH = 'AQI/AQIdata.js'

MACAU_TZ = ZoneInfo('Asia/Macau')

plt.rcParams['font.sans-serif'] = [
    'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK TC', 
    'Noto Sans CJK SC', 'PingFang SC', 'Microsoft YaHei', 'sans-serif'
]
plt.rcParams['axes.unicode_minus'] = False

JSON_URL = "https://www.smg.gov.mo/smg/airQuality/latestAirConcentration.json"

GRID_BORDER_COLOR = '#8DA59A'
TEXT_COLOR = '#334155'
LINE_COLOR = '#0284C7'

# ==========================================
# 澳門 AQI 斷點表與等級著色標準
# ==========================================
AQI_LEVELS = [
    (0,   50,  "#DCFCE7", "#16A34A", "良好 (0-50)"),
    (50,  100, "#FEF9C3", "#CA8A04", "普通 (51-100)"),
    (100, 200, "#FEF08A", "#D97706", "不良 (101-200)"),
    (200, 300, "#FFEDD5", "#EA580C", "非常不良 (201-300)"),
    (300, 400, "#FEE2E2", "#DC2626", "嚴重 (301-400)"),
    (400, 500, "#FCE7F3", "#991B1B", "有害 (401-500)")
]

BREAKPOINTS = {
    'I':     [0, 50, 100, 200, 300, 400, 500],
    'PM10':  [0, 50, 100, 250, 350, 420, 500],
    'PM2_5': [0, 25,  50, 115, 150, 250, 350],
    'SO2':   [0, 20,  50, 150, 475, 800, 1600],
    'NO2':   [0, 100, 200, 700, 1200, 2000, 2500],
    'O3':    [0, 80,  160, 240, 400, 600, 800],
    'CO':    [0, 5,   10,  17,  34,  46,  57]
}

# ==========================================
# 副指數片段式線性內插計算
# ==========================================
def calculate_sub_index(c_val, pollutant_key):
    if c_val is None or np.isnan(c_val) or c_val < 0:
        return None
    
    c_bps = BREAKPOINTS.get(pollutant_key)
    i_bps = BREAKPOINTS['I']
    
    if c_val <= 0:
        return 0.0

    for k in range(len(c_bps) - 1):
        c_low, c_high = c_bps[k], c_bps[k+1]
        i_low, i_high = i_bps[k], i_bps[k+1]

        if c_low <= c_val <= c_high:
            if c_high == c_low:
                return float(i_low)
            i_val = ((i_high - i_low) / (c_high - c_low)) * (c_val - c_low) + i_low
            return round(i_val, 1)

    c_low, c_high = c_bps[-2], c_bps[-1]
    i_low, i_high = i_bps[-2], i_bps[-1]
    i_val = ((i_high - i_low) / (c_high - c_low)) * (c_val - c_low) + i_low
    return round(i_val, 1)

# ==========================================
# 資料庫管理
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS aqi_history
                 (record_time TEXT, timestamp REAL, station_code TEXT, 
                  station_name TEXT, aqi REAL, primary_pollutant TEXT,
                  i_pm10 REAL, i_pm25 REAL, i_so2 REAL, i_no2 REAL, i_o3 REAL, i_co REAL,
                  PRIMARY KEY (record_time, station_code))''')
    conn.commit()
    conn.close()

def get_last_record_from_db():
    if not os.path.exists(DB_PATH):
        return ""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    c = conn.cursor()
    try:
        c.execute("SELECT record_time FROM aqi_history ORDER BY timestamp DESC LIMIT 1")
        row = c.fetchone()
        return row[0] if row else ""
    except Exception:
        return ""
    finally:
        conn.close()

def save_to_db(stations_data, timestamp_val):
    conn = sqlite3.connect(DB_PATH, timeout=10)
    c = conn.cursor()
    cutoff_ts = timestamp_val - (7 * 86400)
    c.execute("DELETE FROM aqi_history WHERE timestamp < ?", (cutoff_ts,))
    for code, d in stations_data.items():
        row = (
            d['record_time'], d['timestamp'], code, d['station_name'],
            d.get('aqi'), d.get('primary_pollutant'),
            d.get('i_pm10'), d.get('i_pm25'), d.get('i_so2'),
            d.get('i_no2'), d.get('i_o3'), d.get('i_co')
        )
        c.execute('''INSERT OR REPLACE INTO aqi_history 
                     (record_time, timestamp, station_code, station_name, aqi, 
                      primary_pollutant, i_pm10, i_pm25, i_so2, i_no2, i_o3, i_co) 
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', row)
    conn.commit()
    conn.close()

# ==========================================
# API 擷取與 AQI 計算
# ==========================================
def fetch_smg_air_json():
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(JSON_URL, headers=headers, timeout=12)
        res.encoding = 'utf-8'
        if res.status_code != 200:
            print(f"❌ 請求 SMG 空氣質量 JSON 失敗，HTTP Status: {res.status_code}")
            return None
        return res.json()
    except Exception as e:
        print(f"❌ 請求 SMG 空氣質量 JSON 異常: {e}")
        return None

def parse_and_compute_aqi(data_json):
    if not data_json:
        return "", 0.0, {}

    dt_str = data_json.get('datetime', '').strip()
    try:
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M").replace(tzinfo=MACAU_TZ)
        clean_time_str = dt.strftime("%Y-%m-%d %H:%M")
        ts = dt.timestamp()
    except Exception:
        dt = datetime.now(MACAU_TZ)
        clean_time_str = dt.strftime("%Y-%m-%d %H:%M")
        ts = dt.timestamp()

    stations_result = {}

    def parse_float(val):
        if val is None:
            return None
        try:
            return float(str(val).strip())
        except (ValueError, TypeError):
            return None

    for key, val in data_json.items():
        if not isinstance(val, dict):
            continue

        st_name = val.get('Chinese', '').strip() or key
        st_code = val.get('CODE', '').strip() or key

        c_pm10 = parse_float(val.get('HE_PM10'))
        c_pm25 = parse_float(val.get('HE_PM2_5'))
        c_so2  = parse_float(val.get('HE_SO2'))
        c_no2  = parse_float(val.get('HE_NO2'))
        c_o3   = parse_float(val.get('HE_O3'))
        c_co   = parse_float(val.get('HE_CO'))

        i_pm10 = calculate_sub_index(c_pm10, 'PM10')
        i_pm25 = calculate_sub_index(c_pm25, 'PM2_5')
        i_so2  = calculate_sub_index(c_so2,  'SO2')
        i_no2  = calculate_sub_index(c_no2,  'NO2')
        i_o3   = calculate_sub_index(c_o3,   'O3')
        i_co   = calculate_sub_index(c_co,   'CO')

        sub_indices = {
            'PM10': i_pm10,
            'PM2.5': i_pm25,
            'SO2': i_so2,
            'NO2': i_no2,
            'O3': i_o3,
            'CO': i_co
        }

        valid_indices = {k: v for k, v in sub_indices.items() if v is not None and not np.isnan(v)}
        if valid_indices:
            aqi_val = round(max(valid_indices.values()), 1)
            primary_pollutant = max(valid_indices, key=valid_indices.get)
        else:
            aqi_val = None
            primary_pollutant = None

        stations_result[st_code] = {
            "station_code": st_code,
            "station_name": st_name,
            "record_time": clean_time_str,
            "timestamp": ts,
            "aqi": aqi_val,
            "primary_pollutant": primary_pollutant,
            "i_pm10": i_pm10,
            "i_pm25": i_pm25,
            "i_so2": i_so2,
            "i_no2": i_no2,
            "i_o3": i_o3,
            "i_co": i_co,
            "concentrations": {
                "PM10": c_pm10,
                "PM2_5": c_pm25,
                "SO2": c_so2,
                "NO2": c_no2,
                "O3": c_o3,
                "CO": c_co
            }
        }

    return clean_time_str, ts, stations_result

# ==========================================
# 繪圖斷線輔助函式
# ==========================================
def insert_nans_on_gap(df, col, max_gap_minutes=70):
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
# 獨立繪製 6 個測站的 AQI 時序圖
# ==========================================
def render_aqi_charts(current_time_str, current_timestamp, current_stations_data):
    conn = sqlite3.connect(DB_PATH, timeout=10)
    df = pd.read_sql_query("SELECT * FROM aqi_history ORDER BY timestamp ASC", conn)
    conn.close()

    if df.empty:
        print("⚠️ aqi_history 資料表為空，跳過繪圖。")
        return

    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Macau').dt.tz_localize(None)

    x_end = datetime.fromtimestamp(current_timestamp, tz=MACAU_TZ).replace(tzinfo=None)
    x_start = x_end - timedelta(hours=24)
    header_update_text = f"更新時間 Update Time:{current_time_str.replace('-', '/')}"

    for st_code, cur_info in current_stations_data.items():
        st_df = df[df['station_code'] == st_code].copy()
        if st_df.empty:
            continue

        st_name = cur_info.get('station_name')
        if not st_name or st_name == st_code:
            valid_names = st_df['station_name'].dropna()
            valid_names = valid_names[valid_names != st_code]
            st_name = valid_names.iloc[-1] if not valid_names.empty else st_code

        # 動態計算該站專屬 Y 軸刻度（不低於 0，預設至少涵蓋良好/普通區間至 100）
        st_df_24h = st_df[st_df['datetime'] >= x_start]
        valid_series = st_df_24h['aqi'].dropna() if not st_df_24h.empty else st_df['aqi'].dropna()

        if not valid_series.empty:
            max_aqi = valid_series.max()
            calc_ymax = max(100.0, max_aqi + 15.0)
            step = 25.0 if calc_ymax <= 150 else (50.0 if calc_ymax <= 300 else 100.0)
            plot_ymax = np.ceil(calc_ymax / step) * step
        else:
            plot_ymax, step = 100.0, 25.0

        plot_ymin = 0.0

        fig, ax = plt.subplots(figsize=(15, 6.5), dpi=120)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        # 1. 繪製 AQI 等級背景區間與分割線
        for low, high, bg_c, text_c, label_txt in AQI_LEVELS:
            if low < plot_ymax and high > plot_ymin:
                eff_low = max(low, plot_ymin)
                eff_high = min(high, plot_ymax)
                ax.axhspan(eff_low, eff_high, facecolor=bg_c, alpha=0.35, zorder=1)

                if eff_low > plot_ymin:
                    ax.axhline(eff_low, color=text_c, linestyle='--', linewidth=0.9, alpha=0.55, zorder=2)
                    ax.text(x_end, eff_low, f" {int(eff_low)}", color=text_c, va='center', ha='left', fontsize=9.5, fontweight='bold')

                mid_y = (eff_low + eff_high) / 2.0
                ax.text(x_start + timedelta(minutes=15), mid_y, label_txt,
                        color=text_c, fontsize=10, fontweight='bold', alpha=0.85, va='center', zorder=2)

        # 2. 繪製該站 AQI 折線
        plot_df = insert_nans_on_gap(st_df, 'aqi', max_gap_minutes=70)
        if not plot_df['aqi'].dropna().empty:
            ax.plot(plot_df['datetime'], plot_df['aqi'], color=LINE_COLOR, linewidth=2.2, antialiased=True, zorder=4, label='空氣質量指數 AQI')
            ax.scatter(st_df['datetime'], st_df['aqi'], color='#0369A1', s=18, alpha=0.85, edgecolors='none', zorder=5)
        else:
            ax.text(0.5, 0.5, '暫無數據 No Data', transform=ax.transAxes, fontsize=18, color='gray', ha='center', va='center')

        # 3. 坐標軸與動態刻度設置
        ax.set_xlim(x_start, x_end)
        ax.set_ylim(bottom=plot_ymin, top=plot_ymax)
        y_ticks = np.arange(plot_ymin, plot_ymax + step * 0.1, step)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(int(y)) for y in y_ticks], fontsize=13, color='#1E293B', family='monospace')

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

        ax.set_ylabel('空氣質量指數 AQI', fontsize=15, color='#1E293B', labelpad=12)
        ax.set_xlabel('澳門時間(時) Local Time (Hour)', fontsize=15, color=TEXT_COLOR, labelpad=14)

        header_title = f"{st_name} ({st_code}) 空氣質量指數趨勢"
        ax.text(0.0, 1.02, header_title, transform=ax.transAxes, fontsize=15, va='bottom', ha='left', color=TEXT_COLOR, fontweight='bold')
        ax.text(1.0, 1.02, header_update_text, transform=ax.transAxes, fontsize=15, va='bottom', ha='right', color=TEXT_COLOR)

        plt.tight_layout()
        img_out = f"AQI/images/{st_code}.png"
        plt.savefig(img_out, dpi=120, bbox_inches='tight')
        plt.close(fig)

    print("✅ 全部 6 個測站 AQI 趨勢圖繪製完成！")

# ==========================================
# 輸出前端 JavaScript 變數檔案 (AQI/AQIdata.js)
# ==========================================
def export_aqi_js(current_time_str, current_timestamp, stations_data):
    payload = {
        "datetime": current_time_str,
        "timestamp": current_timestamp,
        "stations": stations_data
    }
    js_content = f"window.AQIData = {json.dumps(payload, ensure_ascii=False, indent=2)};\n"
    with open(JS_PATH, 'w', encoding='utf-8') as f:
        f.write(js_content)
    print(f"✅ AQI 數據已輸出至 JavaScript 檔案：{JS_PATH}")

# ==========================================
# 主流程
# ==========================================
def main():
    init_db()
    last_record_time = get_last_record_from_db()
    print(f"🚀 開始執行 SMG 空氣質量指數 (AQI) 同步，資料庫最新記錄：'{last_record_time}'")

    raw_json = fetch_smg_air_json()
    if not raw_json:
        sys.exit(1)

    current_time_str, current_timestamp, stations_data = parse_and_compute_aqi(raw_json)

    if not stations_data:
        print("⚠️ 未解析到有效的測站空氣濃度數據。")
        sys.exit(1)

    if current_time_str == last_record_time and last_record_time != "":
        print(f"ℹ️ 空氣質量數據尚未推進 (當前記錄: {current_time_str})，跳過發布與繪圖。")
        sys.exit(2)

    print(f"🔔 檢測到新發布數據: {current_time_str}，共解析到 {len(stations_data)} 個空氣監測站，正在計算並處理...")

    # 1. 寫入 SQLite (保留 7 天紀錄)
    save_to_db(stations_data, current_timestamp)

    # 2. 輸出 AQI/AQIdata.js
    export_aqi_js(current_time_str, current_timestamp, stations_data)

    # 3. 獨立繪製 6 個測站圖表至 AQI/images/{station_code}.png
    render_aqi_charts(current_time_str, current_timestamp, stations_data)
    print("🎉 空氣質量指數同步與 6 站繪圖流程順利完成！")
    sys.exit(0)

if __name__ == '__main__':
    main()
