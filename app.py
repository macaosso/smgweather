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
from matplotlib.patches import Polygon, FancyBboxPatch
from matplotlib import transforms
import matplotlib.dates as mdates
import geopandas as gpd
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
MACAU_TZ = ZoneInfo('Asia/Macau')

update_time_str = datetime.now(MACAU_TZ).strftime('%Y-%m-%d %H:%M')

plt.rcParams['font.sans-serif'] = [
    'Microsoft YaHei', 'SimHei', 'PingFang SC', 
    'Noto Sans CJK SC', 'Noto Sans CJK TC', 'WenQuanYi Micro Hei', 
    'Arial Unicode MS', 'sans-serif'
]
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 氣象站點設定
# ==========================================
STATION_COORDS = {
    "紀念孫中山公園": {"lat": 22.216088, "lon": 113.544967, "area": "澳門半島", "code": "DP"},
    "黑沙環": {"lat": 22.207104, "lon": 113.560261, "area": "澳門半島", "code": "EM"},
    "大炮台": {"lat": 22.196739, "lon": 113.542226, "area": "澳門半島", "code": "FM"},
    "外港碼頭": {"lat": 22.196783, "lon": 113.558233, "area": "澳門半島", "code": "PE"},
    "媽閣": {"lat": 22.185970, "lon": 113.530564, "area": "澳門半島", "code": "MM"},
    "大潭山": {"lat": 22.158936, "lon": 113.568968, "area": "氹仔", "code": "TG"},
    "東亞運大馬路": {"lat": 22.155145, "lon": 113.542488, "area": "氹仔", "code": "JA"},
    "澳門大學": {"lat": 22.131157, "lon": 113.546909, "area": "橫琴澳大校區", "code": "UM"},
    "九澳": {"lat": 22.134651, "lon": 113.580201, "area": "路環", "code": "KV"},
    "路環": {"lat": 22.117241, "lon": 113.551101, "area": "路環", "code": "DC"},
    "友誼大橋北": {"lat": 22.190346, "lon": 113.563335, "area": "跨海大橋", "code": "PN"},
    "友誼大橋南": {"lat": 22.180513, "lon": 113.563346, "area": "跨海大橋", "code": "PS"},
    "澳門大橋北": {"lat": 22.188002, "lon": 113.573184, "area": "跨海大橋", "code": "MN"},
    "澳門大橋南": {"lat": 22.173380, "lon": 113.573100, "area": "跨海大橋", "code": "MS"},
    "嘉樂庇總督大橋": {"lat": 22.179965, "lon": 113.545396, "area": "跨海大橋", "code": "PG"},
    "西灣大橋": {"lat": 22.175653, "lon": 113.534625, "area": "跨海大橋", "code": "PV"},
    "蓮花大橋": {"lat": 22.139996, "lon": 113.549312, "area": "跨海大橋", "code": "PL"}
}

BRIDGE_STATIONS = [
    "友誼大橋北", "友誼大橋南", "澳門大橋北", "澳門大橋南", 
    "嘉樂庇總督大橋", "西灣大橋", "蓮花大橋"
]

PRESSURE_STATIONS = ["外港碼頭", "媽閣", "大潭山", "澳門大學", "九澳", "東亞運大馬路"]

# ==========================================
# 氣象指數數學公式 (WBGT & 體感溫度)
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
# 底圖資料下載與繪製
# ==========================================
print("正在初始化下載地圖底圖與 OSM 大橋...")
bbox = (113.50, 22.05, 113.65, 22.25)
zhuhai, macau, water, bridges = None, None, None, None

try:
    zhuhai = ox.geocode_to_gdf("Zhuhai, China")
    macau = ox.geocode_to_gdf("Macau, China")
    water = ox.features_from_bbox(bbox=bbox, tags={"natural": ["water", "bay", "strait"], "waterway": ["riverbank", "river", "dock"]})
    if water is not None and not water.empty:
        water = water[water.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        water = water[water.geometry.area > 0.000005]
    bridge_tags = {"bridge": "yes", "highway": ["motorway", "trunk", "primary", "secondary"]}
    bridges = ox.features_from_bbox(bbox=bbox, tags=bridge_tags)
    if bridges is not None and not bridges.empty:
        bridges = bridges[bridges.geometry.type.isin(['LineString', 'MultiLineString'])]
        if macau is not None and not macau.empty:
            macau_geom = macau.geometry.unary_union
            bridges = bridges[bridges.intersects(macau_geom)]
except Exception as e:
    print(f"底圖下載失敗: {e}")

if bridges is None:
    bridges = gpd.GeoDataFrame()

def draw_base_map(ax, bg_color='#C4E1F5'):
    fig = ax.figure
    fig.patch.set_facecolor('white')
    ax.set_facecolor(bg_color)
    if zhuhai is not None:
        zhuhai.plot(ax=ax, color='#F5ECD7', edgecolor='none', zorder=1)
    if macau is not None:
        macau.plot(ax=ax, color='#B0B0B0', edgecolor='white', linewidth=1.2, zorder=2)
    if water is not None and not water.empty:
        water.plot(ax=ax, color=bg_color, edgecolor='none', zorder=3)
    if bridges is not None and not bridges.empty:
        bridges.plot(ax=ax, color='#B0B0B0', edgecolor='white', linewidth=1.2, zorder=3.5)
    ax.set_xlim(113.51, 113.62)
    ax.set_ylim(22.08, 22.242)
    ax.set_axis_off()

# ==========================================
# 顏色與指標等級標準映射配置
# ==========================================
PALETTE = {
    'b-rose': '#F43F5E', 'b-red': '#EF4444', 'b-orange': '#F97316',
    'b-amber': '#F59E0B', 'b-yellow': '#EAB308', 'b-green': '#10B981',
    'b-sky': '#0EA5E9', 'b-blue': '#3B82F6', 'b-indigo': '#6366F1',
    'b-purple': '#8B5CF6', 'gray': '#78797A'
}

def get_contrast_text_color(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6: return '#FFFFFF'
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return '#1E293B' if luminance > 0.65 else '#FFFFFF'

def get_temp_style(temp):
    if temp is None or np.isnan(temp): return PALETTE['gray'], '#FFFFFF'
    if temp >= 37.5: c = PALETTE['b-rose']
    elif temp >= 35.5: c = PALETTE['b-red']
    elif temp >= 32.5: c = PALETTE['b-orange']
    elif temp >= 27.5: c = PALETTE['b-amber']
    elif temp >= 22.5: c = PALETTE['b-yellow']
    elif temp >= 17.5: c = PALETTE['b-green']
    elif temp >= 12.5: c = PALETTE['b-sky']
    elif temp >= 7.5: c = PALETTE['b-blue']
    elif temp >= 4.5: c = PALETTE['b-indigo']
    else: c = PALETTE['b-purple']
    return c, get_contrast_text_color(c)

def get_temp_diff_style(diff):
    if diff is None or np.isnan(diff): return "#78797A", "#FFFFFF"
    d = float(diff)
    if d > 0.2:
        return "#EF4444", "#FFFFFF"   # + 紅色
    elif d < -0.2:
        return "#3B82F6", "#FFFFFF"   # - 藍色
    else:
        return "#78797A", "#FFFFFF"   # 0 灰色

def get_wbgt_style(wbgt):
    if wbgt is None or np.isnan(wbgt): return "#555555", "#FFFFFF"
    w = float(wbgt)
    if w >= 32: c = '#991B1B'
    elif w >= 31: c = '#DC2626'
    elif w >= 30: c = '#EA580C'
    elif w >= 28: c = '#F59E0B'
    elif w >= 25: c = '#EAB308'
    else: c = '#10B981'
    return c, get_contrast_text_color(c)

def get_humidity_style(h):
    if h is None or np.isnan(h): return PALETTE['gray'], '#FFFFFF'
    if h >= 95: c = PALETTE['b-blue']
    elif h >= 88: c = PALETTE['b-sky']
    elif h >= 50: c = PALETTE['b-green']
    elif h >= 30: c = PALETTE['b-amber']
    else: c = PALETTE['b-orange']
    return c, get_contrast_text_color(c)

def get_rain1h_style(val):
    if val is None or np.isnan(val): return "#78797a", "#FFFFFF"
    v = float(val)
    if v >= 100.0: return "#6a0c7a", "#FFFFFF"
    if v >= 80.0:  return "#020000", "#FFFFFF"
    if v >= 50.0:  return "#921818", "#FFFFFF"
    if v >= 20.0:  return "#878e14", "#FFFFFF"
    if v >= 10.0:  return "#149f37", "#FFFFFF"
    if v >= 0.1:   return "#227192", "#FFFFFF"
    return "#636363", "#FFFFFF"

def get_raintotal_style(val):
    if val is None or np.isnan(val): return "#78797a", "#FFFFFF"
    v = float(val)
    if v >= 350.0: return "#000000", "#FFFFFF"
    if v >= 200.0: return "#921818", "#FFFFFF"
    if v >= 100.0: return "#e86119", "#FFFFFF"
    if v >= 50.0:  return "#878e14", "#FFFFFF"
    if v >= 0.1:   return "#227192", "#FFFFFF"
    return "#636363", "#FFFFFF"

# 依需求 3 更新風力分級標準:
# 0-30 清勁以下 gray, 31-40 清勁 green, 41-62 強風 blue, 63-87 烈風 amber, 88-117 暴風 orange, 118+ 颶風 red
def get_wind_style(wind):
    if wind is None or np.isnan(wind): return "#78797A", "#FFFFFF"
    w = float(wind)
    if w >= 118.0: return "#EF4444", "#FFFFFF"  # 颶風 red
    if w >= 88.0:  return "#F97316", "#FFFFFF"  # 暴風 orange
    if w >= 63.0:  return "#F59E0B", "#FFFFFF"  # 烈風 amber
    if w >= 41.0:  return "#3B82F6", "#FFFFFF"  # 強風 blue
    if w >= 31.0:  return "#10B981", "#FFFFFF"  # 清勁 green
    return "#78797A", "#FFFFFF"                  # 清勁以下 gray

def get_pressure_style(p):
    if p is None or np.isnan(p): return "#78797a", "#FFFFFF"
    p = float(p)
    if p >= 1025.0: c = '#1E3A8A'
    elif p >= 1020.0: c = '#2563EB'
    elif p >= 1015.0: c = '#3B82F6'
    elif p >= 1010.0: c = '#10B981'
    elif p >= 1005.0: c = '#F59E0B'
    elif p >= 1000.0: c = '#EA580C'
    else: c = '#DC2626'
    return c, get_contrast_text_color(c)

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
    
    c.execute("PRAGMA table_info(weather_history)")
    cols = [r[1] for r in c.fetchall()]
    if 'gust_speed' not in cols:
        c.execute("ALTER TABLE weather_history ADD COLUMN gust_speed REAL")
    if 'wind_dir' not in cols:
        c.execute("ALTER TABLE weather_history ADD COLUMN wind_dir TEXT")
    if 'pressure' not in cols:
        c.execute("ALTER TABLE weather_history ADD COLUMN pressure REAL")

    conn.commit()
    conn.close()

def save_to_db(stations, pub_time_str, pub_timestamp):
    conn = sqlite3.connect(DB_PATH, timeout=10)
    c = conn.cursor()
    cutoff_ts = pub_timestamp - 93600
    c.execute("DELETE FROM weather_history WHERE timestamp < ?", (cutoff_ts,))
    
    rows_to_save = []
    for name, data in stations.items():
        row = (pub_time_str, pub_timestamp, name, 
               data.get('temp_now'), data.get('humidity'), 
               data.get('wind_speed'), data.get('gust_speed'),
               data.get('wind_dir'), data.get('rain_h'), data.get('pressure'))
        rows_to_save.append(row)
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

def save_uv_to_db(uv_data, pub_time_str, pub_timestamp):
    if not uv_data or uv_data.get('uvi') is None: return
    conn = sqlite3.connect(DB_PATH, timeout=10)
    c = conn.cursor()
    cutoff_ts = pub_timestamp - 93600
    c.execute("DELETE FROM uv_history WHERE timestamp < ?", (cutoff_ts,))
    c.execute('''INSERT INTO uv_history (time_str, timestamp, station, uvi) 
                 VALUES (?, ?, ?, ?)''',
              (pub_time_str, pub_timestamp, uv_data.get('station', '大潭山'), uv_data.get('uvi')))
    conn.commit()
    conn.close()

# ==========================================
# 數據獲取與解析 (XML)
# ==========================================
def parse_xml_timestamp(root):
    pub_node = root.find('.//SysPubdate')
    if pub_node is None:
        pub_node = root.find('.//SysPubDate')
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
    """計算各站 24 小時溫差: Current Temp - Temp 24H prior"""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        target_ts_24h = current_timestamp - 86400
        # 尋找 24 小時前 (±35 分鐘內) 最接近的一筆數據
        for name, data in stations.items():
            cur_temp = data.get('temp_now')
            if cur_temp is None or np.isnan(cur_temp):
                data['temp_diff_24h'] = None
                continue
            query = """
                SELECT temp, timestamp FROM weather_history
                WHERE station = ? AND temp IS NOT NULL
                  AND timestamp BETWEEN ? AND ?
                ORDER BY ABS(timestamp - ?) ASC LIMIT 1
            """
            cursor = conn.cursor()
            cursor.execute(query, (name, target_ts_24h - 2100, target_ts_24h + 2100, target_ts_24h))
            row = cursor.fetchone()
            if row and row[0] is not None:
                past_temp = float(row[0])
                data['temp_diff_24h'] = round(cur_temp - past_temp, 1)
            else:
                data['temp_diff_24h'] = None
        conn.close()
    except Exception as e:
        print(f"⚠️ 計算 24 小時溫差異常: {e}")
        for name, data in stations.items():
            if 'temp_diff_24h' not in data:
                data['temp_diff_24h'] = None

def fetch_smg_weather():
    global update_time_str
    print(f"[{datetime.now(MACAU_TZ).strftime('%H:%M:%S')}] 獲取 SMG 實時 XML 氣象數據...")
    url = "https://xml.smg.gov.mo/c_actualweather.xml"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        res = requests.get(url, headers=headers, timeout=12)
        res.encoding = 'utf-8'
        root = ET.fromstring(res.content)
    except Exception as e:
        print(f"❌ 請求 SMG XML 失敗: {e}")
        return None

    update_time_str, pub_timestamp = parse_xml_timestamp(root)
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

        # 海平面氣壓
        press_node = None
        for tag_cand in ['MeanSeaLevelPressure', 'meansealevelpressure', 'Pressure', 'SeaLevelPressure', 'AirPressure']:
            press_node = st.find(tag_cand)
            if press_node is not None:
                break
        item['pressure'] = parse_float(press_node)

        # 雨量
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

        # 風速、陣風與風向
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
    save_to_db(stations, update_time_str, pub_timestamp)

    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump({"update_time": update_time_str, "stations": stations}, f, ensure_ascii=False, indent=2)

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

    uv_time_str, pub_timestamp = parse_xml_timestamp(root)
    uv_val = None
    val_node = root.find('.//UVI') or root.find('.//Value')
    if val_node is not None and val_node.text:
        try: uv_val = float(val_node.text.strip())
        except ValueError: pass

    st_node = root.find('.//StationName')
    station = st_node.text.strip() if st_node is not None and st_node.text else "大潭山"
    uv_data = {'station': station, 'uvi': uv_val, 'time_str': uv_time_str}
    save_uv_to_db(uv_data, uv_time_str, pub_timestamp)
    return uv_data

# ==========================================
# 繪製各項分佈底圖
# ==========================================
def render_distribution_maps(stations, current_uvi=0.0):
    for st_name, d in stations.items():
        t = d.get('temp_now')
        h = d.get('humidity')
        w = d.get('wind_speed', 0.0)
        d['apparent'] = calculate_apparent_temperature(t, h, w) if (t is not None and h is not None) else None
        d['wbgt'] = estimate_wbgt_kmh(t, h, w, current_uvi) if (t is not None and h is not None) else None

    temp_legend = [
        ("≥37.5° 極為酷熱", PALETTE['b-rose']), ("≥35.5° 非常酷熱", PALETTE['b-red']),
        ("≥32.5° 酷熱", PALETTE['b-orange']), ("≥27.5° 炎熱", PALETTE['b-amber']),
        ("≥22.5° 溫暖", PALETTE['b-yellow']), ("≥17.5° 和暖", PALETTE['b-green']),
        ("≥12.5° 清涼", PALETTE['b-sky']), ("≥7.5° 寒冷", PALETTE['b-blue']),
        ("≥4.5° 非常寒冷", PALETTE['b-indigo']), ("<4.5° 極為寒冷", PALETTE['b-purple'])
    ]

    temp_diff_legend = [
        (">0°C 氣溫升高 (+)", "#EF4444"),
        ("<0°C 氣溫下降 (-)", "#3B82F6"),
        ("0°C 氣溫持平", "#78797A")
    ]

    wind_legend = [
        ("≥118 颶風", "#EF4444"), ("88-117 暴風", "#F97316"),
        ("63-87 烈風", "#F59E0B"), ("41-62 強風", "#3B82F6"),
        ("31-40 清勁", "#10B981"), ("0-30 清勁以下", "#78797A")
    ]

    pressure_legend = [
        ("≥1025 hPa", '#1E3A8A'), ("1020-1024 hPa", '#2563EB'),
        ("1015-1019 hPa", '#3B82F6'), ("1010-1014 hPa", '#10B981'),
        ("1005-1009 hPa", '#F59E0B'), ("1000-1004 hPa", '#EA580C'),
        ("<1000 hPa", '#DC2626')
    ]

    tasks = [
        ('temp_now', '實時氣溫分佈', '°C', lambda v: get_temp_style(v), temp_legend),
        ('temp_diff_24h', '24小時氣溫變幅', '°C', get_temp_diff_style, temp_diff_legend),
        ('apparent', '實時體感溫度分佈', '°C', lambda v: get_temp_style(v), temp_legend),
        ('wbgt', '實時暑熱指數 WBGT', '°C', lambda v: get_wbgt_style(v), [
            ("≥32° 極度危險", '#991B1B'), ("≥31° 非常危險", '#DC2626'),
            ("≥30° 危險", '#EA580C'), ("≥28° 警戒", '#F59E0B'),
            ("≥25° 注意", '#EAB308'), ("<25° 正常", '#10B981')
        ]),
        ('temp_max', '今日最高氣溫', '°C', lambda v: get_temp_style(v), temp_legend),
        ('temp_min', '今日最低氣溫', '°C', lambda v: get_temp_style(v), temp_legend),
        ('humidity', '實時相對濕度', '%', lambda v: get_humidity_style(v), [
            ("≥95% 非常潮濕", PALETTE['b-blue']), ("88-94% 潮濕", PALETTE['b-sky']),
            ("50-87% 正常", PALETTE['b-green']), ("30-49% 乾燥", PALETTE['b-amber']),
            ("<30% 非常乾燥", PALETTE['b-orange'])
        ]),
        ('rain_h', '過去一小時降水量', 'mm', get_rain1h_style, [
            ("≥100 mm", "#6a0c7a"), ("80-99.9 mm", "#020000"), ("50-79.9 mm", "#921818"),
            ("20-49.9 mm", "#878e14"), ("10-19.9 mm", "#149f37"), ("0.1-9.9 mm", "#227192"), ("0 mm", "#636363")
        ]),
        ('rain_d', '今日累積降水量', 'mm', get_raintotal_style, [
            ("≥350 mm", "#000000"), ("200-349.9 mm", "#921818"), ("100-199.9 mm", "#e86119"),
            ("50-99.9 mm", "#878e14"), ("0.1-49.9 mm", "#227192"), ("0 mm", "#636363")
        ]),
        ('wind_speed', '平均風速分佈', 'km/h', get_wind_style, wind_legend),
        ('gust_speed', '最高陣風分佈', 'km/h', get_wind_style, wind_legend),
        ('pressure', '海平面氣壓分佈', 'hPa', get_pressure_style, pressure_legend),
        ('wind_dir', '實時風向與風速分佈', '', None, None)
    ]

    for key, title, unit, style_fn, legend_items in tasks:
        fig, ax = plt.subplots(figsize=(11, 14), dpi=140)
        draw_base_map(ax, bg_color='#C4E1F5')

        title_box = FancyBboxPatch(
            (0.03, 0.925), 0.58, 0.062,
            transform=ax.transAxes,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            facecolor='#FFFFFF', edgecolor='#94A3B8',
            linewidth=1.2, alpha=0.96, zorder=12
        )
        ax.add_patch(title_box)

        ax.text(0.045, 0.962, f"澳門氣象分佈  •  {title}", transform=ax.transAxes,
                fontsize=18, fontweight='bold', color='#0F172A', va='center', zorder=13)
        ax.text(0.045, 0.938, f"資料來源：SMG XML  |  數據發布時間：{update_time_str}", transform=ax.transAxes,
                fontsize=11.5, color='#475569', va='center', zorder=13)

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

            if key == 'wind_dir':
                dir_map = {'N':0, 'NNE':22.5, 'NE':45, 'ENE':67.5, 'E':90, 'ESE':112.5, 'SE':135, 'SSE':157.5,
                           'S':180, 'SSW':202.5, 'SW':225, 'WSW':247.5, 'W':270, 'WNW':292.5, 'NW':315, 'NNW':337.5}
                try: deg = float(val)
                except: deg = dir_map.get(str(val).upper(), 0)

                speed = data.get('wind_speed', 0)
                bg_c, _ = get_wind_style(speed)

                ax.plot(lon, lat, marker='o', markersize=8, color=bg_c, markeredgecolor='white', markeredgewidth=1.8, zorder=7)
                arrow_poly = np.array([[0, 2.2], [-0.9, -1.1], [0, -0.4], [0.9, -1.1]])
                t = transforms.Affine2D().rotate_deg(deg).scale(0.0024).translate(lon, lat) + ax.transData
                ax.add_patch(Polygon(arrow_poly, closed=True, facecolor='#1E293B', edgecolor='#FFFFFF', linewidth=1.3, transform=t, zorder=8))

                short_name = name.replace("大橋", "").replace("友誼", "友誼橋").replace("澳門", "澳門橋")
                ax.text(lon, lat - 0.0048, short_name, fontsize=10, ha='center', va='top', 
                        color='#0F172A', fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.25", facecolor='white', edgecolor='none', alpha=0.88), zorder=6)
            else:
                bg_color, font_color = style_fn(val)
                val_text = f"{val:+g}" if key == 'temp_diff_24h' else f"{val:g}"

                ax.annotate(
                    f"{val_text}{unit}",
                    xy=(lon, lat),
                    ha='center', va='center',
                    fontsize=13,
                    fontweight='bold',
                    color=font_color,
                    bbox=dict(
                        boxstyle="round,pad=0.45,rounding_size=0.4",
                        facecolor=bg_color,
                        edgecolor="#FFFFFF",
                        linewidth=2.0,
                        alpha=0.98
                    ),
                    zorder=8
                )

                clean_lbl = name.replace("紀念孫中山公園", "孫中山公園").replace("東亞運大馬路", "東亞運")
                ax.annotate(
                    clean_lbl,
                    xy=(lon, lat),
                    xytext=(0, 18),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=11,
                    fontweight='bold',
                    color='#0F172A',
                    bbox=dict(
                        boxstyle="round,pad=0.25,rounding_size=0.2",
                        facecolor='#FFFFFF',
                        edgecolor='#CBD5E1',
                        linewidth=1.0,
                        alpha=0.95
                    ),
                    zorder=7
                )

        if legend_items:
            patches = [mpatches.Patch(facecolor=c, label=lbl, edgecolor='#94A3B8', linewidth=0.8) for lbl, c in legend_items]
            leg = ax.legend(
                handles=patches, title="分級標準", loc='upper right',
                bbox_to_anchor=(0.97, 0.99),
                title_fontsize=12, fontsize=10.5,
                frameon=True, facecolor='white', edgecolor='#CBD5E1', framealpha=0.96
            )
            leg.set_zorder(12)
            leg.get_title().set_fontweight('bold')

        footer_info = f"澳門地球物理氣象局 (SMG) 實時觀測網絡  |  產圖時間: {datetime.now(MACAU_TZ).strftime('%Y-%m-%d %H:%M')}"
        ax.text(0.035, 0.02, footer_info, transform=ax.transAxes, fontsize=10.5, color='#475569',
                bbox=dict(boxstyle="round,pad=0.4", facecolor='white', edgecolor='#CBD5E1', linewidth=1.0, alpha=0.95), zorder=12)

        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        plt.savefig(f"static/images/{key}.png", dpi=140, bbox_inches='tight')
        plt.close(fig)

    print("✅ 全要素地圖分佈圖 (包含 temp_diff_24h.png 與新風力分級) 渲染完成！")

# ==========================================
# 輔助函數：智能插入 NaN 實現中斷斷線
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

    # 4. 計算每個觀測點與 24 小時前的溫差序列 (Temp Diff 24H)
    temp_diff_series = []
    for idx, row in df.iterrows():
        st = row['station']
        t_curr = row['temp']
        ts_curr = row['timestamp']
        if pd.isna(t_curr):
            temp_diff_series.append(np.nan)
            continue
        ts_24h = ts_curr - 86400
        # 尋找前 24H 同一站點最近的氣溫
        candidates = df[(df['station'] == st) & (df['timestamp'] >= ts_24h - 2100) & (df['timestamp'] <= ts_24h + 2100) & df['temp'].notna()]
        if not candidates.empty:
            closest_idx = (candidates['timestamp'] - ts_24h).abs().idxmin()
            past_t = candidates.loc[closest_idx, 'temp']
            temp_diff_series.append(round(t_curr - past_t, 1))
        else:
            temp_diff_series.append(np.nan)
    df['temp_diff_24h'] = temp_diff_series

    GRID_BORDER_COLOR = '#8DA59A'
    TEXT_COLOR = '#334155'
    header_update_text = f"更新時間 Update Time:{update_time_str.replace('-', '/')}"

    dir_deg_map = {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
        'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5, 'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
    }

    # 各要素定義 (含需求 3 之最新風力分級標記線)
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
            'lines': [
                (118.0, '颶風 118', '#EF4444'),
                (88.0,  '暴風 88',  '#F97316'),
                (63.0,  '烈風 63',  '#F59E0B'),
                (41.0,  '強風 41',  '#3B82F6'),
                (31.0,  '清勁 31',  '#10B981')
            ]
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

    # 全局 X 軸時間範圍：以最新數據時間點為 end，前推 24 小時
    global_max_dt = df['datetime'].max()
    x_end = global_max_dt
    x_start = x_end - pd.Timedelta(hours=24)

    # 計算各氣象要素的統一 Y 軸範圍 (包含動態 Step)
    UNIFIED_Y_LIMITS = {}
    for col, cfg in METRIC_DEFS.items():
        sub_df = df.copy()
        if col == 'wind_speed':
            all_v = pd.concat([sub_df['wind_speed'].dropna(), sub_df['gust_speed'].dropna()])
        else:
            all_v = sub_df[col].dropna()

        if all_v.empty:
            UNIFIED_Y_LIMITS[col] = (0, 10, cfg['step'])
            continue

        y_min, y_max = all_v.min(), all_v.max()

        if col == 'wind_speed':
            # 1. 動態計算間距：依據風速最大值自動適應，不再固定為 4
            v_range = y_max - y_min
            if v_range > 80: step = 10.0
            elif v_range > 40: step = 5.0
            elif v_range > 20: step = 4.0
            else: step = 2.0
            cfg['step'] = step

            # 1. Y 軸僅為風向風羽預留 1 格空間 (即 -1 * step)
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

    # 渲染各測站的獨立要素時序圖
    for station_name in STATION_COORDS.keys():
        st_df = df[df['station'] == station_name].copy()
        if st_df.empty: 
            continue

        is_bridge = station_name in BRIDGE_STATIONS
        code = STATION_COORDS[station_name].get("code", "")
        disp_name = station_name.replace("大炮台", "大炮台").replace("外港碼頭", "外港")
        header_station_text = f"{disp_name}({code})" if code else disp_name

        if is_bridge:
            metrics = ['wind_speed']
        else:
            metrics = ['temp', 'temp_diff_24h', 'apparent', 'wbgt', 'humidity', 'wind_speed', 'rain']
            if station_name in PRESSURE_STATIONS:
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
                # 4. 24 小時溫差圖：正溫差紅線，負溫差藍線，並繪製雙色漸變填充
                ax.plot(plot_df['datetime'], plot_df[col], color='#EF4444', linewidth=1.6, label='24H Temp Diff')
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

            # 標記線
            if 'lines' in cfg:
                for threshold_val, line_label, line_color in cfg['lines']:
                    if plot_ymin < threshold_val < plot_ymax:
                        ax.axhline(y=threshold_val, color=line_color, linestyle='--', linewidth=0.9, alpha=0.6, zorder=2)
                        ax.text(x_end, threshold_val, f" {line_label}", color=line_color, 
                                va='center', ha='left', fontsize=9.5, alpha=0.85, clip_on=True)

            # 2. & 3. 風向僅在「整點」顯示，並改用標準氣象「風羽圖」(Wind Barb)
            if col == 'wind_speed':
                arrow_df = st_df.dropna(subset=['wind_dir']).copy()
                arrow_df['minute'] = arrow_df['datetime'].dt.minute
                # 篩選整點前後 10 分鐘，每個小時只保留最接近 00 分的單一記錄
                whole_hour_candidates = arrow_df[(arrow_df['minute'] <= 10) | (arrow_df['minute'] >= 50)].copy()
                
                if not whole_hour_candidates.empty:
                    whole_hour_candidates['diff_hour'] = np.minimum(whole_hour_candidates['minute'], (60 - whole_hour_candidates['minute']).abs())
                    whole_hour_candidates['hour_str'] = whole_hour_candidates['datetime'].dt.strftime('%Y-%m-%d %H')
                    hourly_barbs = whole_hour_candidates.sort_values('diff_hour').groupby('hour_str').first().reset_index()

                    barb_x = []
                    barb_y = []
                    barb_u = []
                    barb_v = []
                    barb_colors = []

                    barb_base_y = plot_ymin + 0.5 * step  # 位於底部專用預留區間

                    for _, r in hourly_barbs.iterrows():
                        d_t = r['datetime']
                        if pd.isnull(d_t) or d_t < x_start or d_t > x_end:
                            continue
                        w_dir = r.get('wind_dir')
                        w_spd = r.get('wind_speed', 0.0) or 0.0
                        try: deg = float(w_dir)
                        except: deg = dir_deg_map.get(str(w_dir).upper(), None)

                        if deg is not None:
                            # 氣象風向轉 u, v 分量 (風羽指向風來向)
                            # 1 節 (knot) ≈ 1.852 km/h
                            spd_kt = max(2.0, float(w_spd) / 1.852)
                            rad = np.radians(deg)
                            u = -spd_kt * np.sin(rad)
                            v = -spd_kt * np.cos(rad)

                            barb_x.append(mdates.date2num(d_t))
                            barb_y.append(barb_base_y)
                            barb_u.append(u)
                            barb_v.append(v)
                            c, _ = get_wind_style(w_spd)
                            barb_colors.append(c)

                    if barb_x:
                        # 繪製專業風羽
                        ax.barbs(
                            barb_x, barb_y, barb_u, barb_v,
                            length=5.5,
                            pivot='middle',
                            color='#2563EB',
                            barbcolor='#1E40AF',
                            flagcolor='#1E40AF',
                            linewidth=1.2,
                            zorder=6
                        )

            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[15, 30, 45]))

            def format_hour_ticks(x, pos=None):
                dt = mdates.num2date(x)
                hour_str = f"{dt.strftime('%H')}H"
                if dt.hour == 0:
                    return f"{hour_str}\n{dt.strftime('%d %b')}"
                return hour_str

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

    # 跨海大橋風力趨勢圖
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

    print("✅ 時序趨勢圖 (包含整點風羽 Wind Barb、24H 溫差折線與大橋圖) 全部渲染完成！")

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
        if dt.hour == 0:
            return f"{hour_str}\n{dt.strftime('%d %b')}"
        return hour_str

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
# 24 小時降雨量統計表 (保證永久輸出 JSON 文件)
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

    # 如果歷史紀錄不足或分析為空，回退使用 stations_data 或預設值確保必定有 JSON 輸出
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

    print("\n" + "=" * 65)
    print("📊 澳門過去 24 小時累積雨量統計表 (每小時 35 分定點抽樣加總)")
    print("=" * 65)
    print(summary_df.to_string(index=False))
    print("=" * 65 + "\n")

    summary_df.to_json(output_path, orient="records", force_ascii=False, indent=2)
    print(f"✅ 成功生成並輸出 {output_path}")

# ==========================================
# 程式進入點
# ==========================================
if __name__ == '__main__':
    init_db()
    stations_data = fetch_smg_weather()
    uv_data = fetch_smg_uvi()
    current_uvi = uv_data.get('uvi', 0.0) if uv_data else 0.0

    if stations_data:
        # 優先保證雨量 JSON 統計表輸出
        generate_24h_rainfall_summary(stations_data)
        render_distribution_maps(stations_data, current_uvi)
        render_trend_charts()
        render_uv_trend_chart()
        print(f"🎉 全部功能升級完畢！當前 XML 觀測時間：{update_time_str}")
    else:
        # 即便網路異常，依然保證輸出當前最新可用的雨量 JSON 檔案
        generate_24h_rainfall_summary(None)
        print("❌ 無法獲取 SMG XML 數據，已確保預備狀態檔案輸出。")
