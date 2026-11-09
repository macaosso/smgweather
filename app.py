import os
import re
import sys
import time
import json
import math
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
from matplotlib.collections import LineCollection

warnings.filterwarnings("ignore")

# 建立輸出目錄
os.makedirs('static/images', exist_ok=True)

DB_PATH = 'history.db'
JSON_PATH = 'weather_data.json'
CSV_PATH = 'static/weather_data.csv'
WIND_INDEX_DB = 'windindex.db'
WIND_INDEX_JSON = 'windindexdata.json'

MACAU_TZ = ZoneInfo('Asia/Macau')

# 全域時間變數：統一嚴格由 c_actualweather.xml 中的 <SysPubdate> 初始化
update_time_str = ""
date_part = ""
time_part = ""
pub_timestamp = 0.0

plt.rcParams['font.sans-serif'] = [
    'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK TC', 
    'Noto Sans CJK SC', 'PingFang SC', 'Microsoft YaHei', 'sans-serif'
]
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 氣象站點設定
# ==========================================
STATION_COORDS = {
    "紀念孫中山公園": {"lat": 22.216088, "lon": 113.544967, "area": "澳門半島", "code": "DP", "disp": "孫中山公園"},
    "黑沙環": {"lat": 22.207104, "lon": 113.560261, "area": "澳門半島", "code": "EM", "disp": "黑沙環"},
    "大炮台": {"lat": 22.196739, "lon": 113.542226, "area": "澳門半島", "code": "FM", "disp": "大炮台"},
    "外港碼頭": {"lat": 22.196783, "lon": 113.558233, "area": "澳門半島", "code": "PE", "disp": "外港碼頭"},
    "媽閣": {"lat": 22.185970, "lon": 113.530564, "area": "澳門半島", "code": "MM", "disp": "媽閣"},
    "大潭山": {"lat": 22.158936, "lon": 113.568968, "area": "氹仔", "code": "TG", "disp": "大潭山"},
    "東亞運大馬路": {"lat": 22.155145, "lon": 113.542488, "area": "氹仔", "code": "JA", "disp": "東亞運"},
    "澳門大學": {"lat": 22.131157, "lon": 113.546909, "area": "橫琴澳大校區", "code": "UM", "disp": "澳門大學"},
    "九澳": {"lat": 22.134651, "lon": 113.580201, "area": "路環", "code": "KV", "disp": "九澳"},
    "路環": {"lat": 22.117241, "lon": 113.551101, "area": "路環", "code": "DC", "disp": "路環"},
    "友誼大橋北": {"lat": 22.190346, "lon": 113.563335, "area": "跨海大橋", "code": "PN", "disp": "友誼橋北"},
    "友誼大橋南": {"lat": 22.180513, "lon": 113.563346, "area": "跨海大橋", "code": "PS", "disp": "友誼橋南"},
    "澳門大橋北": {"lat": 22.188002, "lon": 113.573184, "area": "跨海大橋", "code": "MN", "disp": "澳門橋北"},
    "澳門大橋南": {"lat": 22.173380, "lon": 113.573100, "area": "跨海大橋", "code": "MS", "disp": "澳門橋南"},
    "嘉樂庇總督大橋": {"lat": 22.179965, "lon": 113.545396, "area": "跨海大橋", "code": "PG", "disp": "嘉樂庇橋"},
    "西灣大橋": {"lat": 22.175653, "lon": 113.534625, "area": "跨海大橋", "code": "PV", "disp": "西灣大橋"},
    "蓮花大橋": {"lat": 22.139996, "lon": 113.549312, "area": "跨海大橋", "code": "PL", "disp": "蓮花大橋"}
}

BRIDGE_STATIONS = [
    "友誼大橋北", "友誼大橋南", "澳門大橋北", "澳門大橋南", 
    "嘉樂庇總督大橋", "西灣大橋", "蓮花大橋"
]

PRESSURE_STATIONS = ["外港碼頭", "媽閣", "大潭山", "澳門大學", "九澳", "東亞運大馬路"]

COMPASS_DIRS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'N']
DIR_DEG_MAP = {
    'N': 0.0, 'NNE': 22.5, 'NE': 45.0, 'ENE': 67.5,
    'E': 90.0, 'ESE': 112.5, 'SE': 135.0, 'SSE': 157.5,
    'S': 180.0, 'SSW': 202.5, 'SW': 225.0, 'WSW': 247.5,
    'W': 270.0, 'WNW': 292.5, 'NW': 315.0, 'NNW': 337.5
}

def get_wind_index_color(wind_idx):
    if wind_idx is None or np.isnan(wind_idx):
        return "#64748B"
    if wind_idx <= 13:
        return "#15803D"
    elif wind_idx <= 24:
        return "#1D4ED8"
    elif wind_idx <= 44:
        return "#DC2626"
    elif wind_idx <= 74:
        return "#7E22CE"
    else:
        return "#92400E"

# ==========================================
# 氣象指數數學公式
# ==========================================
def estimate_wbgt_kmh(ta, rh, w, uvi=0.0):
    if any(v is None or np.isnan(v) for v in [ta, rh, w]):
        return None
    try:
        uvi_val = 0.0 if (uvi is None or np.isnan(uvi)) else float(uvi)
        rh_sqrt = np.sqrt(rh + 8.313659)
        twb = (ta * np.arctan(0.151977 * rh_sqrt)
               + np.arctan(ta + rh)
               - np.arctan(rh - 1.676331)
               + 0.00391838 * (rh ** 1.5) * np.arctan(0.023101 * rh)
               - 4.686035)
        tg = ta + (1.404 * uvi_val) - (0.111 * w)
        wbgt = 0.7 * twb + 0.2 * tg + 0.1 * ta
        return round(float(wbgt), 1)
    except:
        return None

def calculate_apparent_temperature(ta, rh, w):
    if any(v is None or np.isnan(v) for v in [ta, rh, w]):
        return None
    try:
        wind_speed_ms = w / 3.6
        e = (rh / 100.0) * 6.105 * math.exp((17.27 * ta) / (237.7 + ta))
        apparent_temp = (1.04 * ta) + (0.2 * e) - (0.65 * wind_speed_ms) - 2.7
        return round(float(apparent_temp), 1)
    except:
        return None

# ==========================================
# 資料庫管理與 CSV 存儲
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS weather_history
                 (time_str TEXT, timestamp REAL, station TEXT, 
                  temp REAL, humidity REAL, wind_speed REAL, gust_speed REAL, 
                  wind_dir TEXT, rain REAL, pressure REAL,
                  PRIMARY KEY (time_str, station))''')
    c.execute('''CREATE TABLE IF NOT EXISTS uv_history
                 (time_str TEXT, timestamp REAL, station TEXT, uvi REAL,
                  PRIMARY KEY (time_str, station))''')
    conn.commit()
    conn.close()

def init_wind_index_db():
    conn = sqlite3.connect(WIND_INDEX_DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS wind_index_history
                 (time_str TEXT PRIMARY KEY, timestamp REAL,
                  land_u REAL, land_ug REAL,
                  bridge_v REAL, bridge_vg REAL,
                  wind_index REAL, grade TEXT,
                  wind_dirs_json TEXT)''')
    
    c.execute("PRAGMA table_info(wind_index_history)")
    columns = [row[1] for row in c.fetchall()]
    if 'wind_dirs_json' not in columns:
        try:
            c.execute("ALTER TABLE wind_index_history ADD COLUMN wind_dirs_json TEXT")
            print("🔧 已自動為舊版 wind_index_history 資料庫補充缺失欄位：wind_dirs_json")
        except Exception as e:
            print(f"⚠️ 自動補充欄位失敗: {e}")

    conn.commit()
    conn.close()

def get_last_pubdate_from_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    c = conn.cursor()
    try:
        c.execute("SELECT time_str FROM weather_history ORDER BY timestamp DESC LIMIT 1")
        row = c.fetchone()
        return row[0] if row else ""
    except Exception:
        return ""
    finally:
        conn.close()

def save_to_db(stations, pub_time_str, pub_timestamp_val):
    conn = sqlite3.connect(DB_PATH, timeout=10)
    c = conn.cursor()
    cutoff_ts = pub_timestamp_val - (7 * 86400)
    c.execute("DELETE FROM weather_history WHERE timestamp < ?", (cutoff_ts,))
    for name, data in stations.items():
        row = (pub_time_str, pub_timestamp_val, name, 
               data.get('temp_now'), data.get('humidity'), 
               data.get('wind_speed'), data.get('gust_speed'),
               data.get('wind_dir'), data.get('rain_h'), data.get('pressure'))
        c.execute('''INSERT OR REPLACE INTO weather_history 
                     (time_str, timestamp, station, temp, humidity, wind_speed, gust_speed, wind_dir, rain, pressure) 
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', row)
    conn.commit()
    conn.close()
    export_csv(pub_time_str, stations)

def export_csv(pub_time_str, stations):
    csv_rows = []
    for name, d in stations.items():
        csv_rows.append({
            "SysPubdate": pub_time_str,
            "station": name,
            "code": d.get('code', ''),
            "area": d.get('area', ''),
            "temp": d.get('temp_now'),
            "temp_max": d.get('temp_max'),
            "temp_min": d.get('temp_min'),
            "temp_diff_24h": d.get('temp_diff_24h'),
            "apparent": d.get('apparent'),
            "wbgt": d.get('wbgt'),
            "humidity": d.get('humidity'),
            "wind_speed": d.get('wind_speed'),
            "gust_speed": d.get('gust_speed'),
            "wind_dir": d.get('wind_dir'),
            "rain_1h": d.get('rain_h'),
            "rain_day": d.get('rain_d'),
            "pressure": d.get('pressure')
        })
    csv_df = pd.DataFrame(csv_rows)
    csv_df.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')

def save_uv_to_db(uv_data, pub_time_str, pub_timestamp_val):
    if not uv_data or uv_data.get('uvi') is None: return
    conn = sqlite3.connect(DB_PATH, timeout=10)
    c = conn.cursor()
    cutoff_ts = pub_timestamp_val - (7 * 86400)
    c.execute("DELETE FROM uv_history WHERE timestamp < ?", (cutoff_ts,))
    c.execute('''INSERT OR REPLACE INTO uv_history (time_str, timestamp, station, uvi) 
                 VALUES (?, ?, ?, ?)''',
              (pub_time_str, pub_timestamp_val, uv_data.get('station', '大潭山'), uv_data.get('uvi')))
    conn.commit()
    conn.close()

# ==========================================
# 風力指數與風向計算與儲存
# ==========================================
def compute_and_save_wind_index(stations_data, pub_time_str, pub_timestamp_val):
    land_ws, land_gust = [], []
    bridge_ws, bridge_gust = [], []
    station_wind_dirs = {}

    for name, d in stations_data.items():
        ws = d.get('wind_speed')
        g = d.get('gust_speed')
        wdir = d.get('wind_dir')
        is_bridge = name in BRIDGE_STATIONS

        station_wind_dirs[name] = wdir

        if ws is not None and not np.isnan(ws):
            (bridge_ws if is_bridge else land_ws).append(float(ws))
        if g is not None and not np.isnan(g):
            (bridge_gust if is_bridge else land_gust).append(float(g))

    U = float(np.mean(land_ws)) if land_ws else 0.0
    Ug = float(np.mean(land_gust)) if land_gust else 0.0
    V = float(np.mean(bridge_ws)) if bridge_ws else 0.0
    Vg = float(np.mean(bridge_gust)) if bridge_gust else 0.0

    def get_r(a):
        r = -0.00005 * (a ** 2) + 0.0081 * a - 0.0005
        return max(r, 0.0)

    Ur = get_r(U)
    Vr = get_r(V)
    wind_idx = 0.56 * ((1.0 - Ur) * U + Ur * Ug) + 0.44 * ((1.0 - Vr) * V + Vr * Vg)
    idx_rounded = round(wind_idx, 1)

    if idx_rounded <= 13: grade = "輕微"
    elif idx_rounded <= 24: grade = "清勁"
    elif idx_rounded <= 44: grade = "強勁"
    elif idx_rounded <= 74: grade = "強烈"
    else: grade = "極強"

    dirs_json_str = json.dumps(station_wind_dirs, ensure_ascii=False)

    conn = sqlite3.connect(WIND_INDEX_DB, timeout=10)
    c = conn.cursor()
    c.execute("DELETE FROM wind_index_history WHERE timestamp < ?", (pub_timestamp_val - 7 * 86400,))
    c.execute('''INSERT OR REPLACE INTO wind_index_history 
                 (time_str, timestamp, land_u, land_ug, bridge_v, bridge_vg, wind_index, grade, wind_dirs_json)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (pub_time_str, pub_timestamp_val, round(U, 2), round(Ug, 2), round(V, 2), round(Vg, 2), idx_rounded, grade, dirs_json_str))
    conn.commit()
    conn.close()

    json_stations = {}
    for name, d in stations_data.items():
        json_stations[name] = {
            "wind_speed": d.get('wind_speed'),
            "gust_speed": d.get('gust_speed'),
            "wind_dir": d.get('wind_dir'),
            "is_bridge": name in BRIDGE_STATIONS,
            "code": d.get('code', '')
        }

    payload = {
        "update_time": pub_time_str,
        "timestamp": pub_timestamp_val,
        "wind_index": idx_rounded,
        "grade": grade,
        "land": {"avg_wind_speed": round(U, 2), "avg_wind_gust": round(Ug, 2)},
        "bridge": {"avg_wind_speed": round(V, 2), "avg_wind_gust": round(Vg, 2)},
        "stations": json_stations
    }
    with open(WIND_INDEX_JSON, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return idx_rounded, grade

# ==========================================
# 數據獲取與時間同步
# ==========================================
def parse_xml_timestamp(root):
    pub_node = root.find('.//System/SysPubdate')
    if pub_node is None:
        pub_node = root.find('.//SysPubdate') or root.find('.//SysPubDate')
        
    if pub_node is not None and pub_node.text:
        raw_time = pub_node.text.strip()
        for fmt in ["%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M:%S"]:
            try:
                dt = datetime.strptime(raw_time, fmt).replace(tzinfo=MACAU_TZ)
                clean_str = dt.strftime("%Y-%m-%d %H:%M")
                return clean_str, dt.timestamp()
            except ValueError:
                continue
    fallback_dt = datetime.now(MACAU_TZ)
    return fallback_dt.strftime("%Y-%m-%d %H:%M"), fallback_dt.timestamp()

def compute_24h_temp_diff(stations, current_timestamp):
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        target_ts_24h = current_timestamp - 86400
        for name, data in stations.items():
            cur_temp = data.get('temp_now')
            if cur_temp is None or np.isnan(cur_temp):
                data['temp_diff_24h'] = None
                continue
            cursor = conn.cursor()
            cursor.execute("""
                SELECT temp, timestamp FROM weather_history
                WHERE station = ? AND temp IS NOT NULL
                  AND timestamp BETWEEN ? AND ?
                ORDER BY ABS(timestamp - ?) ASC LIMIT 1
            """, (name, target_ts_24h - 2100, target_ts_24h + 2100, target_ts_24h))
            row = cursor.fetchone()
            if row and row[0] is not None:
                data['temp_diff_24h'] = round(cur_temp - float(row[0]), 1)
            else:
                cursor.execute("""
                    SELECT temp, timestamp FROM weather_history
                    WHERE station = ? AND temp IS NOT NULL
                    ORDER BY timestamp ASC LIMIT 1
                """, (name,))
                base_row = cursor.fetchone()
                if base_row and base_row[0] is not None and (current_timestamp - base_row[1]) >= 1800:
                    data['temp_diff_24h'] = round(cur_temp - float(base_row[0]), 1)
                else:
                    data['temp_diff_24h'] = None
        conn.close()
    except Exception as e:
        print(f"⚠️ 計算 24 小時溫差異常: {e}")

def fetch_smg_weather_xml():
    url = "https://xml.smg.gov.mo/c_actualweather.xml"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers, timeout=12)
        res.encoding = 'utf-8'
        return ET.fromstring(res.content)
    except Exception as e:
        print(f"❌ 請求 SMG XML 失敗: {e}")
        return None

def parse_smg_weather(root, current_uvi=0.0):
    global update_time_str, date_part, time_part, pub_timestamp
    if root is None:
        return None

    update_time_str, pub_timestamp = parse_xml_timestamp(root)
    dt_obj = datetime.strptime(update_time_str, '%Y-%m-%d %H:%M')
    date_part = dt_obj.strftime('%m/%d')
    time_part = dt_obj.strftime('%H:%M')

    stations = {name: dict(info) for name, info in STATION_COORDS.items()}

    def clean_name(n):
        return (n.replace("市政", "").replace("氣象站", "").replace("站", "")
                 .replace('纪','紀').replace('孙','孫').replace('园','園')
                 .replace('臺','台').replace('妈','媽').replace('阁','閣')
                 .replace('东','東').replace('运','運').replace('学','學')
                 .replace('莲','蓮').replace('桥','橋').strip())

    def parse_float(el):
        if el is None: return None
        for tag in ['dValue', 'Value']:
            v = el.find(f'.//{tag}')
            if v is not None and v.text:
                try: return float(v.text.strip())
                except: pass
        if el.text and el.text.strip():
            try: return float(el.text.strip())
            except: pass
        return None

    for st in root.findall('.//WeatherReport/station'):
        name_node = st.find('stationname')
        if name_node is None or not name_node.text:
            continue
        st_clean = clean_name(name_node.text)
        target = next((k for k in STATION_COORDS if clean_name(k) in st_clean or st_clean in clean_name(k)), None)
        if not target:
            continue

        item = stations[target]
        item['temp_now'] = parse_float(st.find('Temperature'))
        item['temp_max'] = parse_float(st.find('Temperature_daily_max'))
        item['temp_min'] = parse_float(st.find('Temperature_daily_min'))
        item['humidity'] = parse_float(st.find('Humidity'))

        press_node = None
        for tag_cand in ['MeanSeaLevelPressure', 'meansealevelpressure', 'Pressure', 'SeaLevelPressure', 'AirPressure']:
            press_node = st.find(tag_cand)
            if press_node is not None:
                break
        item['pressure'] = parse_float(press_node)

        h_rain, d_rain = 0.0, 0.0
        for rf in st.findall('Rainfall'):
            type_node = rf.find('Type')
            if type_node is not None and type_node.text:
                t_val = type_node.text.strip()
                val = parse_float(rf)
                if val is not None:
                    if t_val == '4': h_rain = val
                    elif t_val == '5': d_rain = val
        item['rain_h'] = h_rain
        item['rain_d'] = d_rain

        for el in st.iter():
            tag = el.tag.split('}')[-1].lower()
            if tag in ['windspeed', 'wind']:
                v = parse_float(el)
                if v is not None: item['wind_speed'] = v
            elif tag in ['windgust', 'gust']:
                v = parse_float(el)
                if v is not None: item['gust_speed'] = v
            elif 'direction' in tag or tag == 'dir':
                if el.text and el.text.strip():
                    item['wind_dir'] = el.text.strip().upper()
                else:
                    v_node = el.find('.//Value')
                    if v_node is not None and v_node.text:
                        item['wind_dir'] = v_node.text.strip().upper()

    compute_24h_temp_diff(stations, pub_timestamp)

    for name, item in stations.items():
        t = item.get('temp_now')
        h = item.get('humidity')
        w = item.get('wind_speed', 0.0)
        item['apparent'] = calculate_apparent_temperature(t, h, w) if (t is not None and h is not None) else None
        item['wbgt'] = estimate_wbgt_kmh(t, h, w, current_uvi) if (t is not None and h is not None) else None
        if name == "大潭山":
            item['uv_index'] = current_uvi

    save_to_db(stations, update_time_str, pub_timestamp)

    output_json_payload = {
        "update_time": update_time_str,
        "timestamp": pub_timestamp,
        "uv_index": current_uvi,
        "stations": stations
    }
    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_json_payload, f, ensure_ascii=False, indent=2)

    return stations

def fetch_smg_uvi():
    url = "https://xml.smg.gov.mo/c_actualUVI.xml"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers, timeout=12)
        res.encoding = 'utf-8'
        root = ET.fromstring(res.content)
    except Exception as e:
        print(f"❌ 請求 UVI XML 失敗: {e}")
        return None

    uv_time_str, uv_pub_timestamp = parse_xml_timestamp(root)

    uv_val = None
    val_node = root.find('.//UVI') or root.find('.//Value')
    if val_node is not None and val_node.text:
        try: uv_val = float(val_node.text.strip())
        except ValueError: pass

    st_node = root.find('.//StationName')
    station = st_node.text.strip() if st_node is not None and st_node.text else "大潭山"
    uv_data = {'station': station, 'uvi': uv_val, 'time_str': uv_time_str, 'timestamp': uv_pub_timestamp}
    return uv_data

# ==========================================
# 輔助函數：斷線處理與多色折線繪製
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

def plot_two_color_diff_line(ax, dts, values):
    x_nums = mdates.date2num(dts)
    y_vals = np.array(values, dtype=float)

    segments, colors = [], []
    for i in range(len(x_nums) - 1):
        x1, x2 = x_nums[i], x_nums[i + 1]
        y1, y2 = y_vals[i], y_vals[i + 1]

        if np.isnan(y1) or np.isnan(y2):
            continue

        if (y1 >= 0 and y2 >= 0):
            segments.append([(x1, y1), (x2, y2)])
            colors.append('#EF4444')
        elif (y1 < 0 and y2 < 0):
            segments.append([(x1, y1), (x2, y2)])
            colors.append('#3B82F6')
        else:
            t = (0.0 - y1) / (y2 - y1)
            x_cross = x1 + t * (x2 - x1)
            segments.append([(x1, y1), (x_cross, 0.0)])
            colors.append('#EF4444' if y1 >= 0 else '#3B82F6')
            segments.append([(x_cross, 0.0), (x2, y2)])
            colors.append('#EF4444' if y2 >= 0 else '#3B82F6')

    if segments:
        lc = LineCollection(segments, colors=colors, linewidths=1.8, zorder=4)
        ax.add_collection(lc)

def compute_clean_step(data_range, target_ticks=7):
    if data_range <= 0:
        return 1.0
    raw_step = data_range / target_ticks
    power = 10 ** math.floor(math.log10(raw_step))
    rel_step = raw_step / power
    if rel_step <= 1.5:
        chosen = 1.0
    elif rel_step <= 3.0:
        chosen = 2.0
    elif rel_step <= 7.0:
        chosen = 5.0
    else:
        chosen = 10.0
    return chosen * power

# ==========================================
# 繪製各測站歷史時序折線圖
# ==========================================
def render_trend_charts():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    df = pd.read_sql_query("SELECT * FROM weather_history ORDER BY timestamp ASC", conn)
    uv_df = pd.read_sql_query("SELECT * FROM uv_history ORDER BY timestamp ASC", conn)
    conn.close()

    if df.empty:
        print("⚠️ weather_history 資料庫為空，跳過時序圖生成。")
        return

    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Macau').dt.tz_localize(None)

    uv_latest = 0.0
    if not uv_df.empty:
        uv_latest = uv_df['uvi'].iloc[-1] or 0.0

    df['wbgt'] = [estimate_wbgt_kmh(t, h, w, uv_latest) for t, h, w in zip(df['temp'], df['humidity'], df['wind_speed'])]
    df['apparent'] = [calculate_apparent_temperature(t, h, w) for t, h, w in zip(df['temp'], df['humidity'], df['wind_speed'])]

    temp_diff_series = []
    for st_name in df['station'].unique():
        sub_indices = df[df['station'] == st_name].index
        st_records = df.loc[sub_indices]
        base_temp = st_records['temp'].dropna().iloc[0] if not st_records['temp'].dropna().empty else None

        for idx in sub_indices:
            row = df.loc[idx]
            t_curr = row['temp']
            ts_curr = row['timestamp']
            if pd.isna(t_curr):
                temp_diff_series.append((idx, np.nan))
                continue
            ts_24h = ts_curr - 86400
            candidates = st_records[(st_records['timestamp'] >= ts_24h - 2100) & (st_records['timestamp'] <= ts_24h + 2100) & st_records['temp'].notna()]
            if not candidates.empty:
                closest_idx = (candidates['timestamp'] - ts_24h).abs().idxmin()
                past_t = candidates.loc[closest_idx, 'temp']
                temp_diff_series.append((idx, round(t_curr - past_t, 1)))
            elif base_temp is not None and len(st_records) >= 2:
                temp_diff_series.append((idx, round(t_curr - base_temp, 1)))
            else:
                temp_diff_series.append((idx, np.nan))

    temp_diff_dict = dict(temp_diff_series)
    df['temp_diff_24h'] = df.index.map(temp_diff_dict)

    GRID_BORDER_COLOR = '#8DA59A'
    TEXT_COLOR = '#334155'
    header_update_text = f"更新時間 Update Time:{update_time_str.replace('-', '/')}"

    # 更鮮明可辨識的分級背景色 (提升飽和度) 與高對比度分類標準線顏色
    TEMP_SPANS = [
        (37.5, 60.0, "#FECACA", "極為酷熱 (≥37.5)", "#991B1B"),
        (35.5, 37.5, "#FED7AA", "非常酷熱 (35.5-37.4)", "#C2410C"),
        (32.5, 35.5, "#FDE68A", "酷熱 (32.5-35.4)", "#B45309"),
        (7.5,  12.5, "#BAE6FD", "寒冷 (7.5-12.4)", "#0369A1"),
        (3.5,   7.5, "#BFDBFE", "非常寒冷 (3.5-7.4)", "#1D4ED8"),
        (-20.0, 3.5, "#DDD6FE", "極為寒冷 (<3.5)", "#6D28D9")
    ]
    WBGT_SPANS = [
        (32.5, 50.0, "#FECACA", "極危 (≥32.5)", "#991B1B"),     
        (30.5, 32.5, "#FED7AA", "危險 (30.5-32.4)", "#DC2626"), 
        (29.5, 30.5, "#FDE68A", "警戒 (29.5-30.4)", "#D97706")  
    ]
    WIND_SPANS = [
        (117.5, 250.0, "#FECACA", "颶風 (≥117.5)", "#DC2626"),     
        (87.5,  117.5, "#FED7AA", "暴風 (87.5-117.4)", "#EA580C"), 
        (62.5,   87.5, "#FDE68A", "烈風 (62.5-87.4)", "#D97706"),  
        (40.5,   62.5, "#BAE6FD", "強風 (40.5-62.4)", "#0284C7")   
    ]
    RAIN_SPANS = [
        (100.0, 300.0, "#DDD6FE", "大暴雨 (≥100)", "#6D28D9"),    
        (50.0,  100.0, "#FECACA", "暴雨 (50-99.9)", "#DC2626"),   
        (20.0,   50.0, "#BFDBFE", "大雨 (20-49.9)", "#2563EB")    
    ]
    PRESSURE_SPANS = [
        (1025.0, 1080.0, "#FECACA", "高壓脊 (≥1025)", "#DC2626"), 
        (800.0,  1000.0, "#BAE6FD", "低壓 (≤1000)", "#0284C7")    
    ]
    HUMIDITY_SPANS = [
        (95.0, 100.0, "#93C5FD", "極潮濕 (95-100%)", "#1D4ED8"), 
        (85.0,  95.0, "#BAE6FD", "潮濕 (85-94%)", "#0284C7"),    
        (31.0,  50.0, "#FED7AA", "乾燥 (31-50%)", "#EA580C"),    
        (0.0,   31.0, "#FECDD3", "極乾燥 (0-30%)", "#E11D48")     
    ]

    METRIC_DEFS = {
        'temp': {
            'label': '溫度 Temperature(°C)', 'color': '#A23B5A', 'is_bar': False,
            'spans': TEMP_SPANS,
            'lines': [(37.5, '極為酷熱 37.5', '#991B1B'), (35.5, '非常酷熱 35.5', '#C2410C'), (32.5, '酷熱 32.5', '#B45309'), (7.5, '寒冷 7.5', '#0369A1'), (3.5, '非常寒冷 3.5', '#1D4ED8')]
        },
        'temp_diff_24h': {
            'label': '24小時溫差 24H Temp Change(°C)', 'color': '#EF4444', 'is_bar': False,
            'spans': [],
            'lines': [(0.0, '0℃ 氣溫持平', '#475569')]
        },
        'apparent': {
            'label': '體感溫度 Apparent Temp(°C)', 'color': '#EA580C', 'is_bar': False,
            'spans': TEMP_SPANS,
            'lines': [(37.5, '極為酷熱 37.5', '#991B1B'), (35.5, '非常酷熱 35.5', '#C2410C'), (32.5, '酷熱 32.5', '#B45309'), (7.5, '寒冷 7.5', '#0369A1'), (3.5, '非常寒冷 3.5', '#1D4ED8')]
        },
        'wbgt': {
            'label': '暑熱指數 WBGT', 'color': '#D97706', 'is_bar': False,
            'spans': WBGT_SPANS,
            'lines': [(32.5, '極危 32.5', '#991B1B'), (30.5, '危險 30.5', '#DC2626'), (29.5, '警戒 29.5', '#D97706')]
        },
        'humidity': {
            'label': '相對濕度 Humidity(%)', 'color': '#2A7B9B', 'is_bar': False,
            'spans': HUMIDITY_SPANS,
            'lines': [(95.0, '95%', '#1D4ED8'), (85.0, '85%', '#0284C7'), (50.0, '50%', '#EA580C'), (30.0, '30%', '#E11D48')]
        },
        'wind_speed': {
            'label': '風速 Wind Speed(km/h)', 'color': '#8B3A62', 'is_bar': False,
            'spans': WIND_SPANS,
            'lines': [(117.5, '颶風 117.5', '#DC2626'), (87.5, '暴風 87.5', '#EA580C'), (62.5, '烈風 62.5', '#D97706'), (40.5, '強風 40.5', '#0284C7')]
        },
        'rain': {
            'label': '降水量 Rainfall(mm)', 'color': '#1D4ED8', 'is_bar': True,
            'spans': RAIN_SPANS,
            'lines': [(20.0, '大雨 20mm', '#2563EB'), (50.0, '暴雨 50mm', '#DC2626'), (100.0, '大暴雨 100mm', '#6D28D9')]
        },
        'pressure': {
            'label': '海平面氣壓 Pressure(hPa)', 'color': '#1E3A8A', 'is_bar': False,
            'spans': PRESSURE_SPANS,
            'lines': [(1025.0, '高壓 1025', '#DC2626'), (1000.0, '低壓 1000', '#0284C7'), (1013.25, '標準氣壓 1013.25', '#475569')]
        }
    }

    x_end = datetime.fromtimestamp(pub_timestamp, tz=MACAU_TZ).replace(tzinfo=None)
    x_start = x_end - timedelta(hours=24)

    # 依據全體觀測數據動態調整 Y 軸（相同要素維持全站統一 Y 軸，無固定寫死間距）
    UNIFIED_Y_LIMITS = {}
    for col, cfg in METRIC_DEFS.items():
        sub_df = df.copy()
        if col == 'wind_speed':
            all_v = pd.concat([sub_df['wind_speed'].dropna(), sub_df['gust_speed'].dropna()])
        else:
            all_v = sub_df[col].dropna()

        if all_v.empty:
            UNIFIED_Y_LIMITS[col] = (-2.0, 2.0, 1.0) if col == 'temp_diff_24h' else (0.0, 10.0, 2.0)
            continue

        y_min = float(all_v.min())
        y_max = float(all_v.max())

        if col == 'temp_diff_24h':
            abs_max = max(abs(y_min), abs(y_max), 1.0)
            step = compute_clean_step(abs_max * 2, target_ticks=7)
            plot_ymax = math.ceil(abs_max / step) * step
            plot_ymin = -plot_ymax
        elif col == 'wind_speed':
            data_max = max(y_max, 5.0)
            step = compute_clean_step(data_max, target_ticks=6)
            plot_ymin = -1.0 * step
            plot_ymax = math.ceil(data_max / step) * step
            if plot_ymax - plot_ymin < step * 3:
                plot_ymax = plot_ymin + step * 4
        elif col == 'rain':
            data_max = max(y_max, 5.0)
            step = compute_clean_step(data_max, target_ticks=6)
            plot_ymin = 0.0
            plot_ymax = math.ceil(data_max / step) * step
        elif col == 'humidity':
            data_range = max(y_max - y_min, 10.0)
            step = compute_clean_step(data_range, target_ticks=6)
            plot_ymin = max(0.0, math.floor(y_min / step) * step)
            plot_ymax = min(100.0, math.ceil(y_max / step) * step)
            if plot_ymin >= plot_ymax:
                plot_ymin = max(0.0, plot_ymin - step)
                plot_ymax = min(100.0, plot_ymax + step)
        else:
            data_range = max(y_max - y_min, 2.0)
            step = compute_clean_step(data_range, target_ticks=6)
            plot_ymin = math.floor(y_min / step) * step
            plot_ymax = math.ceil(y_max / step) * step
            if plot_ymin == plot_ymax:
                plot_ymin -= step
                plot_ymax += step

        UNIFIED_Y_LIMITS[col] = (plot_ymin, plot_ymax, step)

    for station_name in STATION_COORDS.keys():
        st_df = df[df['station'] == station_name].copy()
        if st_df.empty:
            continue

        is_bridge = station_name in BRIDGE_STATIONS
        code = STATION_COORDS[station_name].get("code", "")
        disp_name = STATION_COORDS[station_name].get("disp", station_name)
        header_station_text = f"{disp_name}({code})" if code else disp_name

        metrics = ['wind_speed'] if is_bridge else ['temp', 'temp_diff_24h', 'apparent', 'wbgt', 'humidity', 'wind_speed', 'rain']
        if not is_bridge and station_name in PRESSURE_STATIONS:
            metrics.append('pressure')

        for col in metrics:
            cfg = METRIC_DEFS[col]
            fig, ax = plt.subplots(figsize=(15, 6.5), dpi=120)
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')

            valid_pts = st_df.dropna(subset=[col])
            plot_ymin, plot_ymax, step = UNIFIED_Y_LIMITS[col]

            if valid_pts.empty:
                ax.text(0.5, 0.5, '暫無數據 No Data', transform=ax.transAxes, fontsize=20, color='gray', ha='center', va='center')
                ax.text(0.0, 1.02, header_station_text, transform=ax.transAxes, fontsize=15, va='bottom', ha='left', color=TEXT_COLOR)
                ax.text(1.0, 1.02, header_update_text, transform=ax.transAxes, fontsize=15, va='bottom', ha='right', color=TEXT_COLOR)
                ax.set_axis_off()
                plt.tight_layout()
                plt.savefig(f"static/images/line_{station_name}_{col}.png", dpi=120, bbox_inches='tight')
                plt.close(fig)
                continue

            # 背景色塊：提高清晰度與透明度
            for span_item in cfg.get('spans', []):
                low, high, span_color, span_lbl = span_item[0], span_item[1], span_item[2], span_item[3]
                text_col = span_item[4] if len(span_item) > 4 else '#334155'
                if low < plot_ymax and high > plot_ymin:
                    eff_low = max(low, plot_ymin)
                    eff_high = min(high, plot_ymax)
                    ax.axhspan(eff_low, eff_high, facecolor=span_color, alpha=0.34, zorder=1)
                    if (eff_high - eff_low) >= (step * 0.45):
                        mid_y = (eff_low + eff_high) / 2.0
                        ax.text(x_start + pd.Timedelta(minutes=15), mid_y, span_lbl,
                                color=text_col, fontsize=9.5, fontweight='bold', alpha=0.9, va='center', zorder=2)

            plot_df = insert_nans_on_gap(st_df, col, max_gap_minutes=25)

            if cfg['is_bar']:
                ax.bar(plot_df['datetime'], plot_df[col], width=pd.Timedelta(minutes=8), color=cfg['color'], alpha=0.85, edgecolor='none', zorder=4)
            elif col == 'temp_diff_24h':
                plot_two_color_diff_line(ax, plot_df['datetime'], plot_df[col])
                ax.fill_between(plot_df['datetime'], plot_df[col], 0, where=(plot_df[col] >= 0), color='#EF4444', alpha=0.30, interpolate=True, zorder=3)
                ax.fill_between(plot_df['datetime'], plot_df[col], 0, where=(plot_df[col] < 0), color='#3B82F6', alpha=0.30, interpolate=True, zorder=3)
            else:
                ax.plot(plot_df['datetime'], plot_df[col], color=cfg['color'], linewidth=1.8, antialiased=True, zorder=4, label=col.capitalize() if col != 'wind_speed' else 'Wind Speed')

            if col == 'wind_speed':
                gust_valid = st_df.dropna(subset=['gust_speed'])
                if not gust_valid.empty:
                    ax.scatter(gust_valid['datetime'], gust_valid['gust_speed'], color='#C0567A', s=16, alpha=0.8, edgecolors='none', label='Gust', zorder=5)
                leg = ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor=GRID_BORDER_COLOR, fontsize=11)
                leg.get_frame().set_linewidth(1.0)

            ax.set_xlim(x_start, x_end)
            y_ticks = np.arange(plot_ymin, plot_ymax + step * 0.1, step)
            ax.set_ylim(plot_ymin, plot_ymax)
            ax.set_yticks(y_ticks)

            def y_fmt(v):
                if col == 'wind_speed' and v < 0: return ""
                if col == 'temp_diff_24h' and v > 0: return f"+{int(v)}" if float(v).is_integer() else f"+{v:.1f}"
                return f"{int(v)}" if float(v).is_integer() else f"{v:.1f}"

            ax.set_yticklabels([y_fmt(y) for y in y_ticks], fontsize=13, color=cfg['color'], family='monospace')

            # 標準指標線：依背景色對應之主題色加深繪製，並加上加粗標註
            if 'lines' in cfg:
                for threshold_val, line_label, line_color in cfg['lines']:
                    if plot_ymin < threshold_val < plot_ymax:
                        ax.axhline(y=threshold_val, color=line_color, linestyle='--', linewidth=1.4, alpha=0.9, zorder=3)
                        ax.text(x_end, threshold_val, f" {line_label}", color=line_color, 
                                va='center', ha='left', fontsize=9.5, fontweight='bold', alpha=0.95, clip_on=False)

            if col == 'wind_speed':
                arrow_df = st_df.dropna(subset=['wind_dir']).copy()
                arrow_df['minute'] = arrow_df['datetime'].dt.minute
                whole_hour_candidates = arrow_df[(arrow_df['minute'] <= 10) | (arrow_df['minute'] >= 50)].copy()
                
                if not whole_hour_candidates.empty:
                    whole_hour_candidates['diff_hour'] = np.minimum(whole_hour_candidates['minute'], (60 - whole_hour_candidates['minute']).abs())
                    whole_hour_candidates['hour_str'] = whole_hour_candidates['datetime'].dt.strftime('%Y-%m-%d %H')
                    hourly_barbs = whole_hour_candidates.sort_values('diff_hour').groupby('hour_str').first().reset_index()

                    barb_x, barb_y, barb_u, barb_v = [], [], [], []
                    barb_base_y = plot_ymin + 0.5 * step

                    for _, r in hourly_barbs.iterrows():
                        d_t = r['datetime']
                        if pd.isnull(d_t) or d_t < x_start or d_t > x_end:
                            continue
                        w_dir = r.get('wind_dir')
                        w_spd = r.get('wind_speed', 0.0) or 0.0
                        try: deg = float(w_dir)
                        except: deg = DIR_DEG_MAP.get(str(w_dir).upper(), None)

                        if deg is not None:
                            spd_kt = max(2.0, float(w_spd) / 1.852)
                            rad = np.radians(deg)
                            barb_x.append(mdates.date2num(d_t))
                            barb_y.append(barb_base_y)
                            barb_u.append(-spd_kt * np.sin(rad))
                            barb_v.append(-spd_kt * np.cos(rad))

                    if barb_x:
                        ax.barbs(barb_x, barb_y, barb_u, barb_v, length=5.5, pivot='middle',
                                 color='#2563EB', barbcolor='#1E40AF', flagcolor='#1E40AF', linewidth=1.2, zorder=6)

            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[15, 30, 45]))

            def format_hour_ticks(x, pos=None):
                dt = mdates.num2date(x)
                hour_str = f"{dt.strftime('%H')}H"
                return f"{hour_str}\n{dt.strftime('%d %b')}" if dt.hour == 0 else hour_str

            ax.xaxis.set_major_formatter(plt.FuncFormatter(format_hour_ticks))
            plt.setp(ax.get_xticklabels(), fontsize=13, color=TEXT_COLOR, family='monospace')

            ax.grid(True, which='major', linestyle=':', linewidth=1.4, color=GRID_BORDER_COLOR, alpha=0.9)
            ax.tick_params(axis='both', which='major', length=6, width=1.1, color=GRID_BORDER_COLOR, direction='out')
            ax.tick_params(axis='x', which='minor', length=3, width=0.8, color=GRID_BORDER_COLOR, direction='out')

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(GRID_BORDER_COLOR)
                spine.set_linewidth(1.3)

            ax.set_ylabel(cfg['label'], fontsize=15, color=cfg['color'], labelpad=12)
            ax.set_xlabel('澳門時間(時) Local Time (Hour)', fontsize=15, color=TEXT_COLOR, labelpad=14)

            ax.text(0.0, 1.02, header_station_text, transform=ax.transAxes, fontsize=15, va='bottom', ha='left', color=TEXT_COLOR)
            ax.text(1.0, 1.02, header_update_text, transform=ax.transAxes, fontsize=15, va='bottom', ha='right', color=TEXT_COLOR)

            plt.tight_layout()
            plt.savefig(f"static/images/line_{station_name}_{col}.png", dpi=120, bbox_inches='tight')
            plt.close(fig)

    # 跨海大橋專用風速趨勢圖
    bridge_colors = {
        "友誼大橋北": "#1E40AF", "友誼大橋南": "#3B82F6",
        "澳門大橋北": "#047857", "澳門大橋南": "#10B981",
        "嘉樂庇總督大橋": "#B45309", "西灣大橋": "#DC2626", "蓮花大橋": "#7C3AED"
    }

    for b_type in ['wind_speed', 'gust_speed']:
        fig, ax = plt.subplots(figsize=(16, 7), dpi=120)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        type_label = '平均風速 Wind Speed(km/h)' if b_type == 'wind_speed' else '最高陣風 Wind Gust(km/h)'
        title_prefix = '大橋風力趨勢圖 (各跨海大橋平均風速)' if b_type == 'wind_speed' else '大橋陣風趨勢圖 (各跨海大橋最高陣風)'
        filename = 'line_bridges_wind_speed.png' if b_type == 'wind_speed' else 'line_bridges_gust_speed.png'

        plot_ymin, plot_ymax, step = UNIFIED_Y_LIMITS['wind_speed']

        for span_item in WIND_SPANS:
            low, high, span_color, span_lbl = span_item[0], span_item[1], span_item[2], span_item[3]
            text_col = span_item[4] if len(span_item) > 4 else '#334155'
            if low < plot_ymax and high > 0:
                eff_low = max(low, 0)
                eff_high = min(high, plot_ymax)
                ax.axhspan(eff_low, eff_high, facecolor=span_color, alpha=0.34, zorder=1)
                mid_y = (eff_low + eff_high) / 2.0
                ax.text(x_start + pd.Timedelta(minutes=15), mid_y, span_lbl,
                        color=text_col, fontsize=9.5, fontweight='bold', alpha=0.9, va='center', zorder=2)

        for b_st in BRIDGE_STATIONS:
            b_df = df[df['station'] == b_st].dropna(subset=[b_type])
            if not b_df.empty:
                plot_b_df = insert_nans_on_gap(b_df, b_type, max_gap_minutes=25)
                ax.plot(plot_b_df['datetime'], plot_b_df[b_type], 
                        label=b_st, color=bridge_colors.get(b_st, '#000000'), 
                        linewidth=1.8, antialiased=True, zorder=4)

        ax.set_xlim(x_start, x_end)
        y_ticks = np.arange(0, plot_ymax + step * 0.1, step)
        ax.set_ylim(0, plot_ymax)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(int(y)) for y in y_ticks], fontsize=13, color='#8B3A62', family='monospace')

        for threshold_val, line_label, line_color in METRIC_DEFS['wind_speed']['lines']:
            if 0 < threshold_val < plot_ymax:
                ax.axhline(y=threshold_val, color=line_color, linestyle='--', linewidth=1.4, alpha=0.9, zorder=3)
                ax.text(x_end, threshold_val, f" {line_label}", color=line_color, va='center', ha='left', fontsize=9.5, fontweight='bold', alpha=0.95)

        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[15, 30, 45]))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_hour_ticks))
        plt.setp(ax.get_xticklabels(), fontsize=13, color=TEXT_COLOR, family='monospace')

        ax.grid(True, which='major', linestyle=':', linewidth=1.4, color=GRID_BORDER_COLOR, alpha=0.9)
        ax.set_ylabel(type_label, fontsize=15, color='#8B3A62', labelpad=12)
        ax.set_xlabel('澳門時間(時) Local Time (Hour)', fontsize=15, color=TEXT_COLOR, labelpad=14)

        ax.legend(loc='upper left', ncol=4, frameon=True, facecolor='white', edgecolor=GRID_BORDER_COLOR, fontsize=11)
        ax.text(0.0, 1.02, title_prefix, transform=ax.transAxes, fontsize=15, va='bottom', ha='left', color=TEXT_COLOR, fontweight='bold')
        ax.text(1.0, 1.02, header_update_text, transform=ax.transAxes, fontsize=15, va='bottom', ha='right', color=TEXT_COLOR)

        plt.tight_layout()
        plt.savefig(f"static/images/{filename}", dpi=120, bbox_inches='tight')
        plt.close(fig)

    print("✅ 時序趨勢圖全部渲染完成！")

# ==========================================
# 繪製 24 小時紫外線折線圖
# ==========================================
def render_uv_trend_chart():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    df = pd.read_sql_query("SELECT * FROM uv_history ORDER BY timestamp ASC", conn)
    conn.close()

    if df.empty: return

    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Macau').dt.tz_localize(None)

    GRID_BORDER_COLOR = '#8DA59A'
    TEXT_COLOR = '#334155'
    UV_COLOR = '#EA580C'

    fig, ax = plt.subplots(figsize=(15, 6.5), dpi=120)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    valid_df = df.dropna(subset=['uvi'])
    if valid_df.empty:
        ax.text(0.5, 0.5, '暫無紫外線數據 No Data', transform=ax.transAxes, fontsize=20, color='gray', ha='center', va='center')
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig("static/images/line_uv_index.png", dpi=120, bbox_inches='tight')
        plt.close(fig)
        return

    x_end = datetime.fromtimestamp(pub_timestamp, tz=MACAU_TZ).replace(tzinfo=None)
    x_start = x_end - timedelta(hours=24)
    ax.set_xlim(x_start, x_end)

    max_uv = float(valid_df['uvi'].max())
    step = compute_clean_step(max(max_uv, 3.0), target_ticks=6)
    plot_ymax = math.ceil(max(max_uv, 3.0) / step) * step
    if plot_ymax < 6.0:
        plot_ymax = 6.0

    y_ticks = np.arange(0, plot_ymax + step * 0.1, step)
    ax.set_ylim(0, plot_ymax)

    uv_levels = [
        (11.0, 50.0, "#DDD6FE", "極高 11+", "#6D28D9"),
        (8.0,  11.0, "#FECACA", "甚高 8-10", "#DC2626"),
        (6.0,  8.0,  "#FED7AA", "高 6-7", "#EA580C"),
        (3.0,  6.0,  "#FEF08A", "中 3-5", "#CA8A04"),
        (0.0,  3.0,  "#BBF7D0", "低 0-2", "#16A34A")
    ]
    for low, high, uv_bg, uv_lbl, uv_line_color in uv_levels:
        if low < plot_ymax:
            eff_high = min(high, plot_ymax)
            ax.axhspan(low, eff_high, facecolor=uv_bg, alpha=0.35, zorder=1)
            if 0 < low < plot_ymax:
                ax.axhline(low, color=uv_line_color, linestyle='--', linewidth=1.4, alpha=0.9, zorder=2)
                ax.text(x_end, low, f" {low}", color=uv_line_color, va='center', ha='left', fontsize=9.5, fontweight='bold')
            mid_y = (low + eff_high) / 2.0
            ax.text(x_start + pd.Timedelta(minutes=15), mid_y, uv_lbl,
                    color=uv_line_color, fontsize=10, fontweight='bold', alpha=0.9, va='center', zorder=2)

    plot_df = insert_nans_on_gap(valid_df, 'uvi', max_gap_minutes=25)
    ax.plot(plot_df['datetime'], plot_df['uvi'], color=UV_COLOR, linewidth=1.8, marker='o', markersize=3.5, antialiased=True, zorder=4)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(int(y)) if float(y).is_integer() else f"{y:.1f}" for y in y_ticks], fontsize=13, color=UV_COLOR, family='monospace')

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[15, 30, 45]))

    def format_hour_ticks(x, pos=None):
        dt = mdates.num2date(x)
        hour_str = f"{dt.strftime('%H')}H"
        return f"{hour_str}\n{dt.strftime('%d %b')}" if dt.hour == 0 else hour_str

    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_hour_ticks))
    plt.setp(ax.get_xticklabels(), fontsize=13, color=TEXT_COLOR, family='monospace')

    ax.grid(True, which='major', linestyle=':', linewidth=1.4, color=GRID_BORDER_COLOR, alpha=0.9)
    ax.tick_params(axis='both', which='major', length=6, width=1.1, color=GRID_BORDER_COLOR, direction='out')
    ax.tick_params(axis='x', which='minor', length=3, width=0.8, color=GRID_BORDER_COLOR, direction='out')

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(GRID_BORDER_COLOR)
        spine.set_linewidth(1.3)

    ax.set_ylabel('紫外線指數 UV Index', fontsize=15, color=UV_COLOR, labelpad=12)
    ax.set_xlabel('澳門時間(時) Local Time (Hour)', fontsize=15, color=TEXT_COLOR, labelpad=14)

    header_update_text = f"更新時間 Update Time:{update_time_str.replace('-', '/')}"
    ax.text(0.0, 1.02, "大潭山(TG) 紫外線指數 UV Index", transform=ax.transAxes, fontsize=15, va='bottom', ha='left', color=TEXT_COLOR)
    ax.text(1.0, 1.02, header_update_text, transform=ax.transAxes, fontsize=15, va='bottom', ha='right', color=TEXT_COLOR)

    plt.tight_layout()
    plt.savefig("static/images/line_uv_index.png", dpi=120, bbox_inches='tight')
    plt.close(fig)
    print("✅ 紫外線折線圖生成完成！")

# ==========================================
# 繪製風力指數歷史折線圖
# ==========================================
def render_wind_index_chart():
    conn = sqlite3.connect(WIND_INDEX_DB, timeout=10)
    df = pd.read_sql_query("SELECT * FROM wind_index_history ORDER BY timestamp ASC", conn)
    conn.close()

    if df.empty:
        print("⚠️ windindex.db 為空，略過風力指數折線圖繪製。")
        return

    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Macau').dt.tz_localize(None)

    fig, ax = plt.subplots(figsize=(15, 6.5), dpi=120)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    plot_df = insert_nans_on_gap(df, 'wind_index', max_gap_minutes=25)

    x_end = datetime.fromtimestamp(pub_timestamp, tz=MACAU_TZ).replace(tzinfo=None)
    x_start = x_end - timedelta(hours=24)
    ax.set_xlim(x_start, x_end)

    df_24h = df[df['datetime'] >= x_start]
    valid_vals = df_24h['wind_index'].dropna() if not df_24h.empty else df['wind_index'].dropna()
    max_val = float(valid_vals.max()) if not valid_vals.empty else 10.0

    step = compute_clean_step(max(max_val, 15.0), target_ticks=6)
    plot_ymax = math.ceil(max(max_val, 15.0) / step) * step
    ax.set_ylim(0, plot_ymax)

    grades = [
        (0, 13, "輕微 (≤13)", "#BBF7D0", "#15803D"),
        (13, 24, "清勁 (14-24)", "#BFDBFE", "#1D4ED8"),
        (24, 44, "強勁 (25-44)", "#FECACA", "#DC2626"),
        (44, 74, "強烈 (45-74)", "#DDD6FE", "#7E22CE"),
        (74, 150, "極強 (≥75)", "#FED7AA", "#92400E")
    ]

    for low, high, g_label, bg_color, line_color in grades:
        if low >= plot_ymax:
            continue
        eff_high = min(high, plot_ymax)
        ax.axhspan(low, eff_high, facecolor=bg_color, alpha=0.38, zorder=1)
        if 0 < low < plot_ymax:
            ax.axhline(low, color=line_color, linestyle='--', linewidth=1.5, alpha=0.9, zorder=2)
            ax.text(x_end, low, f" {low}", color=line_color, va='center', ha='left', fontsize=10, fontweight='bold')
        mid_y = (low + eff_high) / 2.0
        if mid_y < plot_ymax:
            ax.text(x_start + pd.Timedelta(minutes=20), mid_y, g_label, 
                    color=line_color, fontsize=11, fontweight='bold', alpha=0.9, va='center', zorder=2)

    ax.plot(plot_df['datetime'], plot_df['wind_index'], color='#1E293B', linewidth=2.2, label='風力指數 Wind Index', zorder=4)
    ax.scatter(df['datetime'], df['wind_index'], color='#0F172A', s=20, alpha=0.85, zorder=5)

    y_ticks = np.arange(0, plot_ymax + step * 0.1, step)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(int(y)) if float(y).is_integer() else f"{y:.1f}" for y in y_ticks], fontsize=13, color='#1E293B', family='monospace')

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[15, 30, 45]))

    def format_hour_ticks(x, pos=None):
        dt = mdates.num2date(x)
        hour_str = f"{dt.strftime('%H')}H"
        return f"{hour_str}\n{dt.strftime('%d %b')}" if dt.hour == 0 else hour_str

    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_hour_ticks))
    plt.setp(ax.get_xticklabels(), fontsize=13, color='#334155', family='monospace')

    GRID_BORDER_COLOR = '#8DA59A'
    TEXT_COLOR = '#334155'
    ax.grid(True, which='major', linestyle=':', linewidth=1.4, color=GRID_BORDER_COLOR, alpha=0.8)
    ax.tick_params(axis='both', which='major', length=6, width=1.1, color=GRID_BORDER_COLOR, direction='out')

    for spine in ax.spines.values():
        spine.set_color(GRID_BORDER_COLOR)
        spine.set_linewidth(1.3)

    ax.set_ylabel('風力指數 Wind Index', fontsize=15, color='#0F172A', labelpad=12)
    ax.set_xlabel('澳門時間(時) Local Time (Hour)', fontsize=15, color=TEXT_COLOR, labelpad=14)

    title_left = "澳門風力指數趨勢"
    header_update_text = f"更新時間 Update Time:{update_time_str.replace('-', '/')}"

    ax.text(0.0, 1.02, title_left, transform=ax.transAxes, fontsize=15, va='bottom', ha='left', color=TEXT_COLOR, fontweight='bold')
    ax.text(1.0, 1.02, header_update_text, transform=ax.transAxes, fontsize=15, va='bottom', ha='right', color=TEXT_COLOR)
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor=GRID_BORDER_COLOR, fontsize=11)

    plt.tight_layout()
    output_img = "static/images/line_wind_index.png"
    plt.savefig(output_img, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ 風力指數折線圖渲染完成：{output_img}")

# ==========================================
# 繪製風向散點/全站平均趨勢圖 (Y 軸從 N 到 NW, N)
# ==========================================
def render_wind_direction_chart():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    w_df = pd.read_sql_query("SELECT timestamp, station, wind_dir FROM weather_history WHERE wind_dir IS NOT NULL ORDER BY timestamp ASC", conn)
    conn.close()

    conn_idx = sqlite3.connect(WIND_INDEX_DB, timeout=10)
    idx_df = pd.read_sql_query("SELECT timestamp, wind_index FROM wind_index_history ORDER BY timestamp ASC", conn_idx)
    conn_idx.close()

    if w_df.empty:
        print("⚠️ 暫無風向歷史紀錄，跳過風向圖繪製。")
        return

    w_df['datetime'] = pd.to_datetime(w_df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Macau').dt.tz_localize(None)

    idx_map = dict(zip(idx_df['timestamp'], idx_df['wind_index'])) if not idx_df.empty else {}

    def get_deg(dir_val):
        if pd.isna(dir_val): return np.nan
        val_str = str(dir_val).strip().upper()
        try:
            return float(val_str)
        except ValueError:
            return DIR_DEG_MAP.get(val_str, np.nan)

    w_df['deg'] = w_df['wind_dir'].apply(get_deg)
    w_df = w_df.dropna(subset=['deg'])

    if w_df.empty:
        return

    # 計算每個時間戳的全站向量平均風向 (Vector Mean)
    avg_records = []
    for ts, group in w_df.groupby('timestamp'):
        degs = group['deg'].values
        rads = np.radians(degs)
        sin_sum = np.sum(np.sin(rads))
        cos_sum = np.sum(np.cos(rads))
        if len(degs) > 0 and not (sin_sum == 0 and cos_sum == 0):
            avg_rad = np.arctan2(sin_sum, cos_sum)
            avg_deg = (np.degrees(avg_rad) + 360.0) % 360.0
            avg_records.append({
                'timestamp': ts,
                'datetime': group['datetime'].iloc[0],
                'avg_deg': avg_deg,
                'wind_index': idx_map.get(ts)
            })

    avg_df = pd.DataFrame(avg_records).sort_values('datetime').reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(16, 7.5), dpi=120)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    x_end = datetime.fromtimestamp(pub_timestamp, tz=MACAU_TZ).replace(tzinfo=None)
    x_start = x_end - timedelta(hours=24)
    ax.set_xlim(x_start, x_end)

    y_ticks = np.linspace(0, 360, len(COMPASS_DIRS))
    ax.set_ylim(-10, 370)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(COMPASS_DIRS, fontsize=12, fontweight='bold', color='#1E293B', family='monospace')

    # 背景畫出各站離散點 (淡灰色)
    ax.scatter(w_df['datetime'], w_df['deg'], color='#94A3B8', s=16, alpha=0.35, edgecolors='none', label='各站觀測點', zorder=3)

    # 繪製全站平均風向折線 (避免 0°/360° 跨界飛線與斷線)
    if not avg_df.empty:
        line_segments = []
        for i in range(len(avg_df) - 1):
            row1 = avg_df.iloc[i]
            row2 = avg_df.iloc[i + 1]
            diff_min = (row2['datetime'] - row1['datetime']).total_seconds() / 60.0
            
            # 若時間間隔超過 25 分鐘或跨越 0°/360° (角度突變 > 180°)，不相連
            if diff_min <= 25 and abs(row2['avg_deg'] - row1['avg_deg']) <= 180:
                x1 = mdates.date2num(row1['datetime'])
                x2 = mdates.date2num(row2['datetime'])
                line_segments.append([(x1, row1['avg_deg']), (x2, row2['avg_deg'])])

        if line_segments:
            lc = LineCollection(line_segments, colors='#475569', linewidths=2.2, zorder=4)
            ax.add_collection(lc)

        # 依照風力指數著色平均風向點
        point_colors = [get_wind_index_color(wi) for wi in avg_df['wind_index']]
        ax.scatter(avg_df['datetime'], avg_df['avg_deg'], c=point_colors, s=42, alpha=0.95, edgecolors='white', linewidths=1.0, label='全站平均風向', zorder=5)

    for y_val in y_ticks:
        ax.axhline(y_val, color='#CBD5E1', linestyle=':', linewidth=0.9, alpha=0.6, zorder=2)

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[15, 30, 45]))

    def format_hour_ticks(x, pos=None):
        dt = mdates.num2date(x)
        hour_str = f"{dt.strftime('%H')}H"
        return f"{hour_str}\n{dt.strftime('%d %b')}" if dt.hour == 0 else hour_str

    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_hour_ticks))
    plt.setp(ax.get_xticklabels(), fontsize=13, color='#334155', family='monospace')

    GRID_BORDER_COLOR = '#8DA59A'
    TEXT_COLOR = '#334155'
    ax.grid(True, which='major', linestyle=':', linewidth=1.4, color=GRID_BORDER_COLOR, alpha=0.8)
    ax.tick_params(axis='both', which='major', length=6, width=1.1, color=GRID_BORDER_COLOR, direction='out')

    for spine in ax.spines.values():
        spine.set_color(GRID_BORDER_COLOR)
        spine.set_linewidth(1.3)

    ax.set_ylabel('風向 Wind Direction', fontsize=15, color='#1E293B', labelpad=12)
    ax.set_xlabel('澳門時間(時) Local Time (Hour)', fontsize=15, color=TEXT_COLOR, labelpad=14)

    title_left = "澳門各站平均風向趨勢 (風力指數著色)"
    header_update_text = f"更新時間 Update Time:{update_time_str.replace('-', '/')}"
    ax.text(0.0, 1.02, title_left, transform=ax.transAxes, fontsize=15, va='bottom', ha='left', color=TEXT_COLOR, fontweight='bold')
    ax.text(1.0, 1.02, header_update_text, transform=ax.transAxes, fontsize=15, va='bottom', ha='right', color=TEXT_COLOR)

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='輕微 (≤13)', markerfacecolor='#15803D', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='清勁 (14-24)', markerfacecolor='#1D4ED8', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='強勁 (25-44)', markerfacecolor='#DC2626', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='強烈 (45-74)', markerfacecolor='#7E22CE', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='極強 (≥75)', markerfacecolor='#92400E', markersize=8),
    ]
    ax.legend(handles=legend_handles, loc='upper right', frameon=True, facecolor='white', edgecolor=GRID_BORDER_COLOR, fontsize=10, ncol=5)

    plt.tight_layout()
    output_img = "static/images/line_wind_direction.png"
    plt.savefig(output_img, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ 風向圖渲染完成：{output_img}")

# ==========================================
# 24 小時降雨量統計表
# ==========================================
def generate_24h_rainfall_summary(stations_data=None):
    output_path = "static/images/rainfall_24h_summary.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    records = []

    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        df = pd.read_sql_query("SELECT * FROM weather_history ORDER BY timestamp ASC", conn)
        conn.close()
    except Exception as e:
        print(f"⚠️ 讀取資料庫異常: {e}")
        df = pd.DataFrame()

    if not df.empty:
        try:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Macau')

            for station, info in STATION_COORDS.items():
                if station in BRIDGE_STATIONS:
                    continue

                st_data = df[df['station'] == station].copy()
                total_24h_rain = 0.0
                current_h_rain = 0.0

                if not st_data.empty:
                    st_data = st_data.sort_values('datetime')
                    latest_row = st_data.iloc[-1]
                    latest_time = latest_row['datetime']
                    current_h_rain = latest_row['rain'] if pd.notnull(latest_row['rain']) else 0.0

                    target_times = [latest_time - timedelta(hours=i) for i in range(24)]
                    
                    hourly_readings = []
                    for target in target_times:
                        window = st_data[
                            (st_data['datetime'] >= target - timedelta(minutes=7)) &
                            (st_data['datetime'] <= target + timedelta(minutes=7))
                        ]
                        if not window.empty:
                            closest_idx = (window['datetime'] - target).abs().idxmin()
                            val = window.loc[closest_idx, 'rain']
                            if pd.notnull(val):
                                hourly_readings.append(val)

                    total_24h_rain = sum(hourly_readings)

                records.append({
                    "Station": station,
                    "Code": info.get('code', ''),
                    "Area": info.get('area', ''),
                    "Current_Hourly_Rain (mm)": round(float(current_h_rain), 1),
                    "Past_24H_Total_Rain (mm)": round(float(total_24h_rain), 1)
                })

        except Exception as e:
            print(f"⚠️ 分析 24 小時累積雨量錯誤: {e}")
            records = []

    if not records:
        for station, info in STATION_COORDS.items():
            if station in BRIDGE_STATIONS:
                continue
            d = (stations_data or {}).get(station, {})
            current_h_rain = d.get('rain_h') or 0.0
            day_rain = d.get('rain_d') or current_h_rain
            records.append({
                "Station": station,
                "Code": info.get('code', ''),
                "Area": info.get('area', ''),
                "Current_Hourly_Rain (mm)": round(float(current_h_rain), 1),
                "Past_24H_Total_Rain (mm)": round(float(day_rain), 1)
            })

    summary_df = pd.DataFrame(records)
    if not summary_df.empty and "Past_24H_Total_Rain (mm)" in summary_df.columns:
        summary_df.sort_values(by="Past_24H_Total_Rain (mm)", ascending=False, inplace=True)

    summary_df.to_json(output_path, orient="records", force_ascii=False, indent=2)
    print(f"✅ 成功生成並輸出 {output_path}")

# ==========================================
# 完整處理管線
# ==========================================
def process_pipeline(root):
    uv_data = fetch_smg_uvi()
    current_uvi = uv_data.get('uvi', 0.0) if uv_data else 0.0

    stations_data = parse_smg_weather(root, current_uvi=current_uvi)

    if stations_data:
        if uv_data:
            save_uv_to_db(uv_data, update_time_str, pub_timestamp)

        compute_and_save_wind_index(stations_data, update_time_str, pub_timestamp)
        generate_24h_rainfall_summary(stations_data)
        
        # 繪製各項趨勢圖表
        render_trend_charts()
        render_uv_trend_chart()
        render_wind_index_chart()
        render_wind_direction_chart()
        print(f"🎉 全部圖表生成完成！發布時間：{update_time_str}")
    else:
        generate_24h_rainfall_summary(None)
        print("❌ 解析 SMG XML 數據失敗。")

# ==========================================
# 程式進入點 (單次執行模式，由外部 CI 循環排程)
# ==========================================
if __name__ == '__main__':
    init_db()
    init_wind_index_db()
    
    last_pubdate = get_last_pubdate_from_db()
    print(f"🚀 執行 SMG 同步任務，資料庫前次發布時間：'{last_pubdate}'")

    try:
        root = fetch_smg_weather_xml()
        if root is not None:
            pub_time_str, _ = parse_xml_timestamp(root)

            # XML 時間未更新：返回狀態碼 2，通知 GitHub Actions 跳過後續 Git push
            if pub_time_str == last_pubdate:
                print(f"ℹ️ SMG XML 尚未更新 (當前發布時間: {pub_time_str})，跳過繪圖與發布。")
                sys.exit(2)
            else:
                print(f"🔔 檢測到新發布數據: {pub_time_str}，開始管線處理...")
                process_pipeline(root)
                sys.exit(0)
        else:
            print("❌ 無法獲取 SMG XML 內容。")
            sys.exit(1)
    except Exception as e:
        print(f"⚠️ 處理過程異常: {e}")
        sys.exit(1)
