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
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon, FancyBboxPatch, Circle
from matplotlib import transforms
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from PIL import Image
import geopandas as gpd
from shapely.geometry import shape, box
from shapely.ops import unary_union
import osmnx as ox

warnings.filterwarnings("ignore")

# 建立輸出目錄
os.makedirs('static/images', exist_ok=True)
os.makedirs('cache', exist_ok=True)

ox.settings.use_cache = True
ox.settings.cache_folder = './cache'
ox.settings.log_console = False

DB_PATH = 'history.db'
JSON_PATH = 'weather_data.json'
CSV_PATH = 'static/weather_data.csv'
WIND_INDEX_DB = 'windindex.db'
WIND_INDEX_JSON = 'windindexdata.json'

MACAU_TZ = ZoneInfo('Asia/Macau')

# 全域時間變數：統一由 XML SysPubdate 初始化
update_time_str = ""
date_part = ""
time_part = ""
pub_timestamp = 0.0

plt.rcParams['font.sans-serif'] = [
    'Noto Sans CJK TC', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei',
    'PingFang SC', 'Microsoft YaHei', 'SimHei', 'sans-serif'
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
# 嚴格 4:3 (1600x1200) 座標範圍與地圖底圖
# ==========================================
LAT_CENTER = 22.165
LON_CENTER = 113.563
LAT_SPAN = 0.120
LON_SPAN = (4.0 / 3.0) * LAT_SPAN / math.cos(math.radians(LAT_CENTER))

MAP_BBOX = {
    'min_lat': LAT_CENTER - LAT_SPAN / 2,
    'max_lat': LAT_CENTER + LAT_SPAN / 2,
    'min_lon': LON_CENTER - LON_SPAN / 2,
    'max_lon': LON_CENTER + LON_SPAN / 2
}

def load_mapdata_json(json_file='mapdata.json'):
    if not os.path.exists(json_file):
        return None, None
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        land_geoms = []
        bridge_geoms = []
        for feat in data.get('features', []):
            lbl = feat.get('properties', {}).get('label', '')
            geom = shape(feat['geometry'])
            if '橋' in lbl or '口岸' in lbl:
                bridge_geoms.append(geom)
            else:
                land_geoms.append(geom)

        land_union = unary_union(land_geoms) if land_geoms else None
        bridge_union = unary_union(bridge_geoms) if bridge_geoms else None
        
        land_gdf = gpd.GeoDataFrame(geometry=[land_union], crs="EPSG:4326") if land_union else None
        bridge_gdf = gpd.GeoDataFrame(geometry=[bridge_union], crs="EPSG:4326") if bridge_union else None
        return land_gdf, bridge_gdf
    except Exception as e:
        print(f"⚠️ 解析 mapdata.json 失敗: {e}")
        return None, None

unified_land_gdf, bridges_gdf = load_mapdata_json('mapdata.json')

try:
    if unified_land_gdf is None or unified_land_gdf.empty:
        zhuhai_raw = ox.geocode_to_gdf("Zhuhai, China")
        macau_raw = ox.geocode_to_gdf("Macau, China")
        all_geoms = []
        if macau_raw is not None and not macau_raw.empty:
            all_geoms.append(macau_raw.geometry.unary_union)
        if zhuhai_raw is not None and not zhuhai_raw.empty:
            all_geoms.append(zhuhai_raw.geometry.unary_union)
        if all_geoms:
            unified_land_gdf = gpd.GeoDataFrame(geometry=[unary_union(all_geoms)], crs="EPSG:4326")
except Exception as e:
    print(f"OSM 補充下載略過: {e}")

def draw_base_map(ax):
    bg_water = '#475569'
    land_color = '#E2E8F0'
    bridge_color = '#BAC7CC'
    
    ax.set_facecolor(bg_water)
    if unified_land_gdf is not None and not unified_land_gdf.empty:
        unified_land_gdf.plot(ax=ax, color=land_color, edgecolor='#FFFFFF', linewidth=1.6, zorder=2)
    if bridges_gdf is not None and not bridges_gdf.empty:
        bridges_gdf.plot(ax=ax, color=bridge_color, edgecolor='#FFFFFF', linewidth=1.5, zorder=3)

    ax.set_xlim(MAP_BBOX['min_lon'], MAP_BBOX['max_lon'])
    ax.set_ylim(MAP_BBOX['min_lat'], MAP_BBOX['max_lat'])
    ax.set_aspect(1.0 / math.cos(math.radians(LAT_CENTER)))
    ax.set_axis_off()

# ==========================================
# 風格與色彩映射
# ==========================================
def get_temp_style(temp):
    if temp is None or np.isnan(temp): return '#94A3B8', '#FFFFFF'
    t = float(temp)
    if t >= 35.0: return '#DC2626', '#FFFFFF'
    elif t >= 33.0: return '#EF4444', '#FFFFFF'
    elif t >= 28.0: return '#F97316', '#FFFFFF'
    elif t >= 22.1: return '#EAB308', '#FFFFFF'
    elif t >= 17.1: return '#84CC16', '#FFFFFF'
    elif t >= 12.1: return '#10B981', '#FFFFFF'
    elif t >= 7.1:  return '#06B6D4', '#FFFFFF'
    else: return '#8B5CF6', '#FFFFFF'

def get_temp_diff_style(diff):
    if diff is None or np.isnan(diff): return '#94A3B8', '#FFFFFF'
    d = float(diff)
    if d >= 0: return '#EF4444', '#FFFFFF'
    return '#3B82F6', '#FFFFFF'

def get_humidity_style(h):
    if h is None or np.isnan(h): return '#94A3B8', '#FFFFFF'
    h = float(h)
    if h >= 95: return '#8B5CF6', '#FFFFFF'
    elif h >= 85: return '#3B82F6', '#FFFFFF'
    elif h >= 71: return '#10B981', '#FFFFFF'
    elif h >= 40: return '#EAB308', '#FFFFFF'
    else: return '#EF4444', '#FFFFFF'

def get_wind_style(w):
    if w is None or np.isnan(w): return '#94A3B8', '#FFFFFF'
    w = float(w)
    if w >= 118: return '#DC2626', '#FFFFFF'
    elif w >= 88: return '#EA580C', '#FFFFFF'
    elif w >= 63: return '#D97706', '#FFFFFF'
    elif w >= 41: return '#16A34A', '#FFFFFF'
    return '#0D9488', '#FFFFFF'

def get_pressure_style(p):
    if p is None or np.isnan(p): return '#94A3B8', '#FFFFFF'
    p = float(p)
    if p >= 1030.0: return '#1E3A8A', '#FFFFFF'
    elif p >= 1010.0: return '#2563EB', '#FFFFFF'
    elif p >= 1000.0: return '#16A34A', '#FFFFFF'
    elif p >= 990.0: return '#EAB308', '#FFFFFF'
    elif p >= 980.0: return '#F97316', '#FFFFFF'
    elif p >= 970.0: return '#EF4444', '#FFFFFF'
    elif p >= 950.0: return '#DC2626', '#FFFFFF'
    else: return '#7C3AED', '#FFFFFF'

def get_rain1h_style(v):
    if v is None or np.isnan(v): return "#94A3B8", "#FFFFFF"
    v = float(v)
    if v >= 100.0: return "#7C3AED", "#FFFFFF"
    if v >= 50.0:  return "#DC2626", "#FFFFFF"
    if v >= 20.0:  return "#EA580C", "#FFFFFF"
    if v >= 10.0:  return "#16A34A", "#FFFFFF"
    if v >= 0.1:   return "#2563EB", "#FFFFFF"
    return "#94A3B8", "#FFFFFF"

def get_wbgt_style(wbgt):
    if wbgt is None or np.isnan(wbgt): return "#94A3B8", "#FFFFFF"
    w = float(wbgt)
    if w >= 32: return '#991B1B', '#FFFFFF'
    elif w >= 31: return '#DC2626', '#FFFFFF'
    elif w >= 30: return '#EA580C', '#FFFFFF'
    elif w >= 28: return '#F59E0B', '#FFFFFF'
    elif w >= 25: return '#EAB308', '#FFFFFF'
    return '#10B981', '#FFFFFF'

# ==========================================
# 資料庫管理與 CSV 存儲
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS weather_history
                 (time_str TEXT, timestamp REAL, station TEXT, 
                  temp REAL, humidity REAL, wind_speed REAL, gust_speed REAL, 
                  wind_dir TEXT, rain REAL, pressure REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS uv_history
                 (time_str TEXT, timestamp REAL, station TEXT, uvi REAL)''')
    conn.commit()
    conn.close()

def init_wind_index_db():
    conn = sqlite3.connect(WIND_INDEX_DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS wind_index_history
                 (time_str TEXT, timestamp REAL UNIQUE,
                  land_u REAL, land_ug REAL,
                  bridge_v REAL, bridge_vg REAL,
                  wind_index REAL, grade TEXT)''')
    conn.commit()
    conn.close()

def save_to_db(stations, pub_time_str, pub_timestamp_val):
    conn = sqlite3.connect(DB_PATH, timeout=10)
    c = conn.cursor()
    cutoff_ts = pub_timestamp_val - 93600
    c.execute("DELETE FROM weather_history WHERE timestamp < ?", (cutoff_ts,))
    for name, data in stations.items():
        row = (pub_time_str, pub_timestamp_val, name, 
               data.get('temp_now'), data.get('humidity'), 
               data.get('wind_speed'), data.get('gust_speed'),
               data.get('wind_dir'), data.get('rain_h'), data.get('pressure'))
        c.execute('''INSERT INTO weather_history 
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
    cutoff_ts = pub_timestamp_val - 93600
    c.execute("DELETE FROM uv_history WHERE timestamp < ?", (cutoff_ts,))
    c.execute('''INSERT INTO uv_history (time_str, timestamp, station, uvi) 
                 VALUES (?, ?, ?, ?)''',
              (pub_time_str, pub_timestamp_val, uv_data.get('station', '大潭山'), uv_data.get('uvi')))
    conn.commit()
    conn.close()

# ==========================================
# 風力指數計算與儲存
# ==========================================
def compute_and_save_wind_index(stations_data, pub_time_str, pub_timestamp_val):
    land_ws, land_gust = [], []
    bridge_ws, bridge_gust = [], []

    for name, d in stations_data.items():
        ws = d.get('wind_speed')
        g = d.get('gust_speed')
        is_bridge = name in BRIDGE_STATIONS

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

    conn = sqlite3.connect(WIND_INDEX_DB, timeout=10)
    c = conn.cursor()
    c.execute("DELETE FROM wind_index_history WHERE timestamp < ?", (pub_timestamp_val - 7 * 86400,))
    c.execute('''INSERT OR REPLACE INTO wind_index_history 
                 (time_str, timestamp, land_u, land_ug, bridge_v, bridge_vg, wind_index, grade)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (pub_time_str, pub_timestamp_val, round(U, 2), round(Ug, 2), round(V, 2), round(Vg, 2), idx_rounded, grade))
    conn.commit()
    conn.close()

    json_stations = {}
    for name, d in stations_data.items():
        json_stations[name] = {
            "wind_speed": d.get('wind_speed'),
            "gust_speed": d.get('gust_speed'),
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
# 數據獲取與時間同步 (統一依賴 SysPubdate)
# ==========================================
def parse_xml_timestamp(root):
    pub_node = root.find('.//SysPubdate') or root.find('.//SysPubDate')
    if pub_node is not None and pub_node.text:
        raw_time = pub_node.text.strip()
        try:
            dt = datetime.strptime(raw_time, "%Y-%m-%d %H:%M").replace(tzinfo=MACAU_TZ)
            return raw_time, dt.timestamp()
        except ValueError:
            pass
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
                # 若無剛好 24 小時前紀錄，退而求其次比對資料庫最早之一筆
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

def fetch_smg_weather(current_uvi=0.0):
    global update_time_str, date_part, time_part, pub_timestamp
    url = "https://xml.smg.gov.mo/c_actualweather.xml"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        res = requests.get(url, headers=headers, timeout=12)
        res.encoding = 'utf-8'
        root = ET.fromstring(res.content)
    except Exception as e:
        print(f"❌ 請求 SMG XML 失敗: {e}")
        return None

    # 統一使用 XML SysPubdate
    update_time_str, pub_timestamp = parse_xml_timestamp(root)
    dt_obj = datetime.strptime(update_time_str, '%Y-%m-%d %H:%M')
    date_part = dt_obj.strftime('%m/%d')
    time_part = dt_obj.strftime('%H:%M')

    print(f"[{time_part}] 成功解析 SMG 氣象數據，發布時間：{update_time_str}")

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
        "uv_index": current_uvi,
        "stations": stations
    }
    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_json_payload, f, ensure_ascii=False, indent=2)

    return stations

def fetch_smg_uvi():
    global update_time_str, date_part, time_part, pub_timestamp
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
    if not update_time_str:
        update_time_str = uv_time_str
        pub_timestamp = uv_pub_timestamp
        dt_obj = datetime.strptime(update_time_str, '%Y-%m-%d %H:%M')
        date_part = dt_obj.strftime('%m/%d')
        time_part = dt_obj.strftime('%H:%M')

    uv_val = None
    val_node = root.find('.//UVI') or root.find('.//Value')
    if val_node is not None and val_node.text:
        try: uv_val = float(val_node.text.strip())
        except ValueError: pass

    st_node = root.find('.//StationName')
    station = st_node.text.strip() if st_node is not None and st_node.text else "大潭山"
    uv_data = {'station': station, 'uvi': uv_val, 'time_str': uv_time_str}
    save_uv_to_db(uv_data, uv_time_str, uv_pub_timestamp)
    return uv_data

# ==========================================
# 繪製各項分佈底圖
# ==========================================
def render_distribution_maps(stations, current_uvi=0.0):
    for st_name, d in stations.items():
        if d.get('apparent') is None:
            t = d.get('temp_now')
            h = d.get('humidity')
            w = d.get('wind_speed', 0.0)
            d['apparent'] = calculate_apparent_temperature(t, h, w) if (t is not None and h is not None) else None
            d['wbgt'] = estimate_wbgt_kmh(t, h, w, current_uvi) if (t is not None and h is not None) else None

    temp_legend = [
        ("極端酷熱 (≥35.0℃)", "#DC2626"),
        ("酷熱 (33.0-34.9℃)", "#EF4444"),
        ("炎熱 (28.0-32.9℃)", "#F97316"),
        ("溫暖 (22.1-27.9℃)", "#EAB308"),
        ("和暖 (17.1-22.0℃)", "#84CC16"),
        ("清涼 (12.1-17.0℃)", "#10B981"),
        ("寒冷 (7.1-12.0℃)", "#06B6D4"),
        ("嚴寒 (≤7.0℃)", "#8B5CF6")
    ]

    humidity_legend = [
        ("非常潮濕 (95-100%)", "#8B5CF6"),
        ("潮濕 (85-94%)", "#3B82F6"),
        ("正常 (71-84%)", "#10B981"),
        ("乾燥 (40-70%)", "#EAB308"),
        ("非常乾燥 (0-39%)", "#EF4444")
    ]

    wind_legend = [
        ("颶風 (≥118 km/h)", "#DC2626"),
        ("暴風 (88-117 km/h)", "#EA580C"),
        ("烈風 (63-87 km/h)", "#D97706"),
        ("強風 (41-62 km/h)", "#16A34A"),
        ("清勁或以下 (0-40 km/h)", "#0D9488")
    ]

    pressure_legend = [
        ("≥ 1030.0 hPa", "#1E3A8A"),
        ("1010.0-1029.9 hPa", "#2563EB"),
        ("1000.0-1009.9 hPa", "#16A34A"),
        ("990.0-999.9 hPa", "#EAB308"),
        ("980.0-989.9 hPa", "#F97316"),
        ("970.0-979.9 hPa", "#EF4444"),
        ("950.0-969.9 hPa", "#DC2626"),
        ("≤ 949.9 hPa", "#7C3AED")
    ]

    tasks = [
        ('temp_now', '分區氣溫', get_temp_style, temp_legend),
        ('temp_diff_24h', '過去24小時氣溫差別', get_temp_diff_style, None),
        ('apparent', '體感溫度', get_temp_style, temp_legend),
        ('wbgt', '暑熱指數 WBGT', get_wbgt_style, None),
        ('temp_max', '今日最高氣溫', get_temp_style, temp_legend),
        ('temp_min', '今日最低氣溫', get_temp_style, temp_legend),
        ('humidity', '相對濕度', get_humidity_style, humidity_legend),
        ('rain_h', '過去一小時降水', get_rain1h_style, None),
        ('rain_d', '今日累積降水', get_rain1h_style, None),
        ('wind_speed', '過去10分鐘平均風速及風向', get_wind_style, wind_legend),
        ('gust_speed', '最高陣風分佈', get_wind_style, wind_legend),
        ('wind_dir', '實時風向與風速分佈', get_wind_style, wind_legend),
        ('pressure', '平均海平面氣壓', get_pressure_style, pressure_legend),
    ]

    logo_path = 'MSSO(H).png' if os.path.exists('MSSO(H).png') else ('static/images/MSSO(H).png' if os.path.exists('static/images/MSSO(H).png') else None)

    for key, title_text, style_fn, legend_items in tasks:
        fig = plt.figure(figsize=(16.0, 12.0), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        draw_base_map(ax)

        card_w, card_h = 0.285, 0.076
        card_x, card_y = 0.045, 0.885
        header_card = FancyBboxPatch((card_x, card_y), card_w, card_h, transform=ax.transAxes,
                                     boxstyle="round,pad=0.006,rounding_size=0.016",
                                     facecolor="#FFFFFF", edgecolor="#94A3B8", linewidth=1.5, zorder=15)
        ax.add_patch(header_card)
        
        accent_bar = FancyBboxPatch((card_x, card_y), 0.009, card_h, transform=ax.transAxes,
                                    boxstyle="round,pad=0,rounding_size=0.005",
                                    facecolor="#F59E0B", edgecolor="none", zorder=16)
        ax.add_patch(accent_bar)
        ax.text(card_x + 0.022, card_y + card_h / 2, title_text, transform=ax.transAxes,
                fontsize=24, fontweight='bold', color='#0F172A', va='center', ha='left', zorder=17)

        time_w, time_h = 0.245, 0.054
        time_y = card_y - 0.068
        time_card = FancyBboxPatch((card_x, time_y), time_w, time_h, transform=ax.transAxes,
                                   boxstyle="round,pad=0.006,rounding_size=0.014",
                                   facecolor="#0F172A", edgecolor="#334155", linewidth=1.2, zorder=15)
        ax.add_patch(time_card)
        
        dt_tag_w = 0.108
        date_badge = FancyBboxPatch((card_x, time_y), dt_tag_w, time_h, transform=ax.transAxes,
                                    boxstyle="round,pad=0.005,rounding_size=0.012",
                                    facecolor="#F59E0B", edgecolor="none", zorder=16)
        ax.add_patch(date_badge)
        ax.text(card_x + dt_tag_w / 2, time_y + time_h / 2, date_part, transform=ax.transAxes,
                fontsize=19, fontweight='bold', color='#FFFFFF', va='center', ha='center', zorder=17)
        ax.text(card_x + dt_tag_w + (time_w - dt_tag_w) / 2, time_y + time_h / 2, time_part, transform=ax.transAxes,
                fontsize=19, fontweight='bold', color='#F8FAFC', va='center', ha='center', zorder=17)

        for name, data in stations.items():
            val = data.get(key)
            lon, lat = data['lon'], data['lat']
            is_bridge = name in BRIDGE_STATIONS

            if key in ['temp_now', 'temp_diff_24h', 'temp_max', 'temp_min', 'humidity', 'rain_h', 'rain_d', 'apparent', 'wbgt', 'pressure'] and is_bridge:
                continue
            if key == 'pressure' and name not in PRESSURE_STATIONS:
                continue
            if val is None:
                continue

            disp_name = data.get('disp', name)

            if key in ['wind_speed', 'gust_speed', 'wind_dir']:
                dir_map = {'N':0, 'NNE':22.5, 'NE':45, 'ENE':67.5, 'E':90, 'ESE':112.5, 'SE':135, 'SSE':157.5,
                           'S':180, 'SSW':202.5, 'SW':225, 'WSW':247.5, 'W':270, 'WNW':292.5, 'NW':315, 'NNW':337.5}
                try: deg = float(data.get('wind_dir', 0))
                except: deg = dir_map.get(str(data.get('wind_dir')).upper(), 0)

                spd_val = data.get('wind_speed' if key != 'gust_speed' else 'gust_speed', 0.0) or 0.0
                bg_c, _ = get_wind_style(spd_val)
                val_display = int(round(spd_val))

                arrow_poly = np.array([[0, 2.3], [-1.1, -1.3], [0, -0.4], [1.1, -1.3]])
                t = transforms.Affine2D().rotate_deg(deg).scale(0.0034).translate(lon, lat) + ax.transData
                ax.add_patch(Polygon(arrow_poly, closed=True, facecolor=bg_c, edgecolor='#FFFFFF', linewidth=1.8, transform=t, zorder=10))

                ax.text(lon, lat, f"{val_display}", fontsize=14, fontweight='bold', color='#FFFFFF',
                        ha='center', va='center', zorder=12)

                ax.text(lon + 0.0080, lat, disp_name, fontsize=13.5, fontweight='bold', color='#FFFFFF',
                        ha='left', va='center',
                        bbox=dict(boxstyle="square,pad=0.28", facecolor='#0E3A53', edgecolor='none'),
                        zorder=11)

            elif key == 'temp_diff_24h':
                d_val = float(val)
                symbol = "▲" if d_val >= 0 else "▼"
                bg_color = "#EF4444" if d_val >= 0 else "#3B82F6"
                val_text = f"{symbol}{abs(d_val):.1f}"

                ax.text(lon - 0.0042, lat, val_text, fontsize=15, fontweight='bold', color='#FFFFFF',
                        ha='right', va='center',
                        bbox=dict(boxstyle="round,pad=0.40,rounding_size=0.6", facecolor=bg_color, edgecolor='#FFFFFF', linewidth=2.0),
                        zorder=12)

                ax.text(lon - 0.0018, lat, disp_name, fontsize=14, fontweight='bold', color='#FFFFFF',
                        ha='left', va='center',
                        bbox=dict(boxstyle="square,pad=0.28", facecolor='#0E3A53', edgecolor='none'),
                        zorder=11)

            else:
                bg_color, font_color = style_fn(val)
                if key in ['temp_now', 'apparent', 'wbgt', 'temp_max', 'temp_min', 'pressure']:
                    val_str = f"{val:.1f}"
                else:
                    val_str = f"{int(round(val))}"

                ax.text(lon - 0.0042, lat, val_str, fontsize=15.5, fontweight='bold', color=font_color,
                        ha='right', va='center',
                        bbox=dict(boxstyle="round,pad=0.42,rounding_size=0.8", facecolor=bg_color, edgecolor='#FFFFFF', linewidth=2.0),
                        zorder=12)

                ax.text(lon - 0.0018, lat, disp_name, fontsize=14, fontweight='bold', color='#FFFFFF',
                        ha='left', va='center',
                        bbox=dict(boxstyle="square,pad=0.28", facecolor='#0E3A53', edgecolor='none'),
                        zorder=11)

        if legend_items:
            n_items = len(legend_items)
            leg_w = 0.198
            item_h = 0.050
            leg_h = item_h * n_items + 0.035
            leg_x = 0.775
            leg_y = 0.50 - leg_h / 2

            leg_card = FancyBboxPatch((leg_x, leg_y), leg_w, leg_h, transform=ax.transAxes,
                                      boxstyle="round,pad=0.008,rounding_size=0.016",
                                      facecolor="#0F172A", edgecolor="#334155", linewidth=1.2, alpha=0.88, zorder=14)
            ax.add_patch(leg_card)

            start_y = leg_y + leg_h - 0.036
            for i, (label, col) in enumerate(legend_items):
                curr_y = start_y - i * item_h
                circle = Circle((leg_x + 0.020, curr_y), 0.0085, transform=ax.transAxes,
                                facecolor=col, edgecolor='#FFFFFF', linewidth=1.0, zorder=16)
                ax.add_patch(circle)
                ax.text(leg_x + 0.038, curr_y, label, transform=ax.transAxes,
                        fontsize=13.5, color='#F8FAFC', va='center', ha='left', fontweight='500', zorder=16)

        if logo_path and os.path.exists(logo_path):
            try:
                logo_img = Image.open(logo_path)
                imagebox = OffsetImage(logo_img, zoom=0.22)
                ab = AnnotationBbox(imagebox, (0.885, 0.095), frameon=False, xycoords='axes fraction', zorder=18)
                ax.add_artist(ab)
            except Exception as e:
                print(f"Logo 載入失敗: {e}")
        else:
            ax.text(0.885, 0.080, "MSSO", transform=ax.transAxes,
                    fontsize=28, fontweight='heavy', color='#FFFFFF', ha='center', va='center', zorder=18)
            ax.text(0.885, 0.045, "WEATHER MACAU", transform=ax.transAxes,
                    fontsize=12, fontweight='bold', color='#94A3B8', ha='center', va='center', zorder=18)

        plt.savefig(f"static/images/{key}.png", dpi=100, facecolor='#475569')
        plt.close(fig)

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
    """將溫差數值正值畫為紅色，負值畫為藍色"""
    x_nums = mdates.date2num(dts)
    y_vals = np.array(values, dtype=float)

    segments = []
    colors = []

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
            # 跨越 0 軸時計算插值交點
            t = (0.0 - y1) / (y2 - y1)
            x_cross = x1 + t * (x2 - x1)
            segments.append([(x1, y1), (x_cross, 0.0)])
            colors.append('#EF4444' if y1 >= 0 else '#3B82F6')
            segments.append([(x_cross, 0.0), (x2, y2)])
            colors.append('#EF4444' if y2 >= 0 else '#3B82F6')

    if segments:
        lc = LineCollection(segments, colors=colors, linewidths=1.8, zorder=4)
        ax.add_collection(lc)

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

    # 計算各點相對於 24 小時前或資料庫基線的溫差
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
                # 歷史累積不足 24 小時，改與最早一筆有效資料比對溫差
                temp_diff_series.append((idx, round(t_curr - base_temp, 1)))
            else:
                temp_diff_series.append((idx, np.nan))

    temp_diff_dict = dict(temp_diff_series)
    df['temp_diff_24h'] = df.index.map(temp_diff_dict)

    GRID_BORDER_COLOR = '#8DA59A'
    TEXT_COLOR = '#334155'
    header_update_text = f"更新時間 Update Time:{update_time_str.replace('-', '/')}"

    dir_deg_map = {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
        'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5, 'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
    }

    METRIC_DEFS = {
        'temp': {
            'label': '溫度 Temperature(℃)', 'color': '#A23B5A', 'step': 1.0, 'is_bar': False,
            'lines': [(35.5, '非常酷熱 35.5', '#EF4444'), (32.5, '酷熱 32.5', '#F97316'), (27.5, '炎熱 27.5', '#F59E0B'), (17.5, '和暖 17.5', '#10B981'), (12.5, '清涼 12.5', '#0EA5E9'), (7.5, '寒冷 7.5', '#3B82F6')]
        },
        'temp_diff_24h': {
            'label': '24小時溫差 24H Temp Change(℃)', 'color': '#EF4444', 'step': 1.0, 'is_bar': False,
            'lines': [(0.0, '0℃ 氣溫持平', '#64748B')]
        },
        'apparent': {
            'label': '體感溫度 Apparent Temp(℃)', 'color': '#EA580C', 'step': 1.0, 'is_bar': False,
            'lines': [(35.5, '非常酷熱 35.5', '#EF4444'), (32.5, '酷熱 32.5', '#F97316'), (27.5, '炎熱 27.5', '#F59E0B'), (17.5, '和暖 17.5', '#10B981'), (12.5, '清涼 12.5', '#0EA5E9'), (7.5, '寒冷 7.5', '#3B82F6')]
        },
        'wbgt': {
            'label': '暑熱指數 WBGT(℃)', 'color': '#D97706', 'step': 1.0, 'is_bar': False,
            'lines': [(32.0, '極度危險 32', '#991B1B'), (31.0, '非常危險 31', '#DC2626'), (30.0, '危險 30', '#EA580C'), (28.0, '警戒 28', '#F59E0B'), (25.0, '注意 25', '#EAB308')]
        },
        'humidity': {
            'label': '濕度 Humidity(%)', 'color': '#2A7B9B', 'step': 5.0, 'is_bar': False,
            'lines': [(95.0, '非常潮濕 95%', '#3B82F6'), (88.0, '潮濕 88%', '#0EA5E9'), (50.0, '正常 50%', '#10B981'), (30.0, '乾燥 30%', '#F59E0B')]
        },
        'wind_speed': {
            'label': '風速 Wind Speed(km/h)', 'color': '#8B3A62', 'step': 5.0, 'is_bar': False,
            'lines': [(118.0, '颶風 118', '#EF4444'), (88.0, '暴風 88', '#F97316'), (63.0, '烈風 63', '#F59E0B'), (41.0, '強風 41', '#3B82F6'), (31.0, '清勁 31', '#10B981')]
        },
        'rain': {
            'label': '降水量 Rainfall(mm)', 'color': '#1D4ED8', 'step': 2.0, 'is_bar': True,
            'lines': [(10.0, '10mm', '#149F37'), (20.0, '20mm', '#878E14'), (50.0, '50mm', '#921818')]
        },
        'pressure': {
            'label': '海平面氣壓 Pressure(hPa)', 'color': '#1E3A8A', 'step': 1.0, 'is_bar': False,
            'lines': [(1013.25, '標準大氣壓 1013.25', '#64748B')]
        }
    }

    global_max_dt = df['datetime'].max()
    x_end = global_max_dt
    x_start = x_end - pd.Timedelta(hours=24)

    UNIFIED_Y_LIMITS = {}
    for col, cfg in METRIC_DEFS.items():
        sub_df = df.copy()
        if col == 'wind_speed':
            all_v = pd.concat([sub_df['wind_speed'].dropna(), sub_df['gust_speed'].dropna()])
        else:
            all_v = sub_df[col].dropna()

        if all_v.empty:
            UNIFIED_Y_LIMITS[col] = (-2, 2, 1.0) if col == 'temp_diff_24h' else (0, 10, cfg['step'])
            continue

        y_min, y_max = all_v.min(), all_v.max()

        if col == 'wind_speed':
            v_range = y_max - y_min
            step = 10.0 if v_range > 80 else (5.0 if v_range > 40 else (4.0 if v_range > 20 else 2.0))
            cfg['step'] = step
            plot_ymin = -1.0 * step
            plot_ymax = int(np.ceil(y_max / step) * step + step)
            if plot_ymax < 20: plot_ymax = 20
        elif col == 'temp_diff_24h':
            step = 1.0
            plot_ymin = np.floor(min(-1.0, y_min)) - 1
            plot_ymax = np.ceil(max(1.0, y_max)) + 1
        elif col in ['temp', 'apparent', 'wbgt', 'pressure']:
            step = cfg['step']
            plot_ymin = np.floor(y_min) - 1
            plot_ymax = np.ceil(y_max) + 1
        else:
            step = cfg['step']
            plot_ymin = max(0, np.floor(y_min / step) * step - step)
            plot_ymax = np.ceil(y_max / step) * step + step

        if plot_ymin >= plot_ymax:
            plot_ymin -= 1
            plot_ymax += 1

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

            plot_df = insert_nans_on_gap(st_df, col, max_gap_minutes=25)

            if cfg['is_bar']:
                ax.bar(plot_df['datetime'], plot_df[col], width=pd.Timedelta(minutes=8), color=cfg['color'], alpha=0.85, edgecolor='none')
            elif col == 'temp_diff_24h':
                # 溫差正值為紅、負值為藍
                plot_two_color_diff_line(ax, plot_df['datetime'], plot_df[col])
                ax.fill_between(plot_df['datetime'], plot_df[col], 0, where=(plot_df[col] >= 0), color='#EF4444', alpha=0.25, interpolate=True)
                ax.fill_between(plot_df['datetime'], plot_df[col], 0, where=(plot_df[col] < 0), color='#3B82F6', alpha=0.25, interpolate=True)
            else:
                ax.plot(plot_df['datetime'], plot_df[col], color=cfg['color'], linewidth=1.5, antialiased=True, label=col.capitalize() if col != 'wind_speed' else 'Wind Speed')

            if col == 'wind_speed':
                gust_valid = st_df.dropna(subset=['gust_speed'])
                if not gust_valid.empty:
                    ax.scatter(gust_valid['datetime'], gust_valid['gust_speed'], color='#C0567A', s=16, alpha=0.7, edgecolors='none', label='Gust', zorder=5)
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

            ax.set_yticklabels([y_fmt(y) for y in y_ticks], fontsize=14, color=cfg['color'], family='monospace')

            if 'lines' in cfg:
                for threshold_val, line_label, line_color in cfg['lines']:
                    if plot_ymin < threshold_val < plot_ymax:
                        ax.axhline(y=threshold_val, color=line_color, linestyle='--', linewidth=0.9, alpha=0.6, zorder=2)
                        ax.text(x_end, threshold_val, f" {line_label}", color=line_color, 
                                va='center', ha='left', fontsize=9.5, alpha=0.85, clip_on=True)

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
                        except: deg = dir_deg_map.get(str(w_dir).upper(), None)

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

            ax.grid(True, which='major', linestyle=':', linewidth=1.6, color=GRID_BORDER_COLOR, alpha=0.95)
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

        for b_st in BRIDGE_STATIONS:
            b_df = df[df['station'] == b_st].dropna(subset=[b_type])
            if not b_df.empty:
                plot_b_df = insert_nans_on_gap(b_df, b_type, max_gap_minutes=25)
                ax.plot(plot_b_df['datetime'], plot_b_df[b_type], 
                        label=b_st, color=bridge_colors.get(b_st, '#000000'), 
                        linewidth=1.8, antialiased=True)

        ax.set_xlim(x_start, x_end)
        plot_ymin, plot_ymax, step = UNIFIED_Y_LIMITS['wind_speed']
        y_ticks = np.arange(0, plot_ymax + step * 0.1, step)
        ax.set_ylim(0, plot_ymax)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(int(y)) for y in y_ticks], fontsize=14, color='#8B3A62', family='monospace')

        for threshold_val, line_label, line_color in METRIC_DEFS['wind_speed']['lines']:
            if 0 < threshold_val < plot_ymax:
                ax.axhline(y=threshold_val, color=line_color, linestyle='--', linewidth=0.9, alpha=0.6, zorder=2)
                ax.text(x_end, threshold_val, f" {line_label}", color=line_color, va='center', ha='left', fontsize=9.5, alpha=0.85)

        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[15, 30, 45]))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_hour_ticks))
        plt.setp(ax.get_xticklabels(), fontsize=13, color=TEXT_COLOR, family='monospace')

        ax.grid(True, which='major', linestyle=':', linewidth=1.6, color=GRID_BORDER_COLOR, alpha=0.95)
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

    plot_df = insert_nans_on_gap(valid_df, 'uvi', max_gap_minutes=25)
    ax.plot(plot_df['datetime'], plot_df['uvi'], color=UV_COLOR, linewidth=1.8, marker='o', markersize=3.5, antialiased=True)

    max_dt = valid_df['datetime'].max()
    x_end = max_dt
    x_start = x_end - pd.Timedelta(hours=24)
    ax.set_xlim(x_start, x_end)

    max_uv = valid_df['uvi'].max()
    plot_ymax = max(11, int(np.ceil(max_uv)) + 1)
    y_ticks = np.arange(0, plot_ymax + 1, 2)
    ax.set_ylim(0, plot_ymax)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(int(y)) for y in y_ticks], fontsize=14, color=UV_COLOR, family='monospace')

    uv_levels = [(11.0, '極高 11+', '#8B5CF6'), (8.0, '甚高 8-10', '#EF4444'), (6.0, '高 6-7', '#F97316'), (3.0, '中 3-5', '#EAB308')]
    for uv_val, uv_lbl, uv_c in uv_levels:
        if 0 < uv_val < plot_ymax:
            ax.axhline(y=uv_val, color=uv_c, linestyle='--', linewidth=0.9, alpha=0.5)
            ax.text(x_end, uv_val, f" {uv_lbl}", color=uv_c, va='center', ha='left', fontsize=9.5, alpha=0.85)

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[15, 30, 45]))

    def format_hour_ticks(x, pos=None):
        dt = mdates.num2date(x)
        hour_str = f"{dt.strftime('%H')}H"
        return f"{hour_str}\n{dt.strftime('%d %b')}" if dt.hour == 0 else hour_str

    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_hour_ticks))
    plt.setp(ax.get_xticklabels(), fontsize=13, color=TEXT_COLOR, family='monospace')

    ax.grid(True, which='major', linestyle=':', linewidth=1.6, color=GRID_BORDER_COLOR, alpha=0.95)
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
# 繪製風力指數歷史折線圖 (動態 Y 軸調整)
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

    x_end = df['datetime'].max()
    x_start = x_end - pd.Timedelta(hours=24)
    ax.set_xlim(x_start, x_end)

    valid_vals = df['wind_index'].dropna()
    max_val = valid_vals.max() if not valid_vals.empty else 10.0

    # 依據資料動態調整 Y 軸，不強求固定顯示 80/所有分類
    if max_val <= 20:
        step = 5
        plot_ymax = max(15, int(np.ceil(max_val / 5.0) * 5 + 5))
    elif max_val <= 50:
        step = 10
        plot_ymax = int(np.ceil(max_val / 10.0) * 10 + 10)
    else:
        step = 10
        plot_ymax = int(np.ceil(max_val / 10.0) * 10 + 10)

    ax.set_ylim(0, plot_ymax)

    grades = [
        (0, 13, "輕微 (≤13)", "#E7F2D4", "#227722"),
        (13, 24, "清勁 (14-24)", "#E1ECF9", "#2255BB"),
        (24, 44, "強勁 (25-44)", "#FCE8D8", "#DD2222"),
        (44, 74, "強烈 (45-74)", "#F9D8F2", "#8822BB"),
        (74, 150, "極強 (≥75)", "#E6D9BC", "#774422")
    ]

    # 僅渲染涵蓋在動態 plot_ymax 內的等級帶
    for low, high, g_label, bg_color, line_color in grades:
        if low >= plot_ymax:
            continue
        eff_high = min(high, plot_ymax)
        ax.axhspan(low, eff_high, facecolor=bg_color, alpha=0.35, zorder=1)
        if 0 < low < plot_ymax:
            ax.axhline(low, color=line_color, linestyle='--', linewidth=1.1, alpha=0.7, zorder=2)
            ax.text(x_end, low, f" {low}", color=line_color, va='center', ha='left', fontsize=9.5, fontweight='bold')
        mid_y = (low + eff_high) / 2.0
        if mid_y < plot_ymax:
            ax.text(x_start + pd.Timedelta(minutes=20), mid_y, g_label, 
                    color=line_color, fontsize=11, fontweight='bold', alpha=0.85, va='center', zorder=2)

    ax.plot(plot_df['datetime'], plot_df['wind_index'], color='#1E293B', linewidth=2.2, label='風力指數 Wind Index', zorder=4)
    ax.scatter(df['datetime'], df['wind_index'], color='#0F172A', s=20, alpha=0.85, zorder=5)

    y_ticks = np.arange(0, plot_ymax + 1, step)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(int(y)) for y in y_ticks], fontsize=13, color='#1E293B', family='monospace')

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

    latest_val = df['wind_index'].iloc[-1]
    latest_grade = df['grade'].iloc[-1]
    title_left = f"澳門風力指數趨勢 (最新: {latest_val:.1f} - {latest_grade})"
    header_update_text = f"更新時間 Update Time:{update_time_str.replace('-', '/')}"

    ax.text(0.0, 1.02, title_left, transform=ax.transAxes, fontsize=15, va='bottom', ha='left', color=TEXT_COLOR, fontweight='bold')
    ax.text(1.0, 1.02, header_update_text, transform=ax.transAxes, fontsize=15, va='bottom', ha='right', color=TEXT_COLOR)

    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor=GRID_BORDER_COLOR, fontsize=11)

    plt.tight_layout()
    output_img = "static/images/line_wind_index.png"
    plt.savefig(output_img, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ 風力指數折線圖動態渲染完成：{output_img}")

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
            latest_time = df['datetime'].max()
            start_24h_window = latest_time - timedelta(hours=24)
            df_24h = df[df['datetime'] >= start_24h_window].copy()

            for station, info in STATION_COORDS.items():
                if station in BRIDGE_STATIONS:
                    continue
                st_data = df_24h[df_24h['station'] == station].copy()
                
                total_24h_rain = 0.0
                current_h_rain = 0.0

                if not st_data.empty:
                    st_data['minute'] = st_data['datetime'].dt.minute
                    st_data['date_hour'] = st_data['datetime'].dt.strftime('%Y-%m-%d %H')
                    sample_candidates = st_data[(st_data['minute'] >= 30) & (st_data['minute'] <= 40)].copy()
                    if not sample_candidates.empty:
                        sample_candidates['diff_35'] = (sample_candidates['minute'] - 35).abs()
                        sampled_hourly = sample_candidates.sort_values('diff_35').groupby('date_hour').first().reset_index()
                        total_24h_rain = sampled_hourly['rain'].fillna(0.0).sum()
                    else:
                        sampled_hourly = st_data.groupby(st_data['datetime'].dt.strftime('%Y-%m-%d %H')).last().reset_index()
                        total_24h_rain = sampled_hourly['rain'].fillna(0.0).sum()

                    latest_row = st_data.iloc[-1]
                    current_h_rain = latest_row['rain'] if pd.notnull(latest_row['rain']) else 0.0

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
# 程式進入點
# ==========================================
if __name__ == '__main__':
    init_db()
    init_wind_index_db()
    
    # 1. 優先獲取 UVI 資料
    uv_data = fetch_smg_uvi()
    current_uvi = uv_data.get('uvi', 0.0) if uv_data else 0.0

    # 2. 獲取氣象數據（統一解析 SysPubdate 設定全局時間）
    stations_data = fetch_smg_weather(current_uvi=current_uvi)

    if stations_data:
        compute_and_save_wind_index(stations_data, update_time_str, pub_timestamp)
        generate_24h_rainfall_summary(stations_data)
        render_distribution_maps(stations_data, current_uvi)
        render_trend_charts()
        render_uv_trend_chart()
        render_wind_index_chart()
        print(f"🎉 全部圖表生成完成！統一 XML 發布時間：{update_time_str}")
    else:
        generate_24h_rainfall_summary(None)
        print("❌ 無法獲取 SMG XML 數據。")
