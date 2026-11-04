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
from matplotlib.patches import Polygon, FancyBboxPatch, PathPatch
from matplotlib.path import Path
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
# 底圖資料下載與繪製（優化質感）
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

def draw_base_map(ax, bg_color='#EAF2F8'):
    """重塑底圖色調：柔和水體藍、自然質感大地色"""
    fig = ax.figure
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor(bg_color)
    if zhuhai is not None:
        zhuhai.plot(ax=ax, color='#F3EFE6', edgecolor='#D8D1C5', linewidth=0.8, zorder=1)
    if macau is not None:
        macau.plot(ax=ax, color='#DCE5DF', edgecolor='#FFFFFF', linewidth=1.5, zorder=2)
    if water is not None and not water.empty:
        water.plot(ax=ax, color=bg_color, edgecolor='none', zorder=3)
    if bridges is not None and not bridges.empty:
        bridges.plot(ax=ax, color='#B0BEC5', edgecolor='white', linewidth=1.6, zorder=3.5)
    ax.set_xlim(113.51, 113.62)
    ax.set_ylim(22.08, 22.242)
    ax.set_axis_off()

# ==========================================
# 顏色配置與自定義風速標準
# ==========================================
PALETTE = {
    'b-rose': '#E11D48', 'b-red': '#DC2626', 'b-orange': '#EA580C',
    'b-amber': '#D97706', 'b-yellow': '#CA8A04', 'b-green': '#059669',
    'b-sky': '#0284C7', 'b-blue': '#2563EB', 'b-indigo': '#4F46E5',
    'b-purple': '#7C3AED', 'gray': '#64748B'
}

def get_contrast_text_color(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6: return '#FFFFFF'
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return '#0F172A' if luminance > 0.65 else '#FFFFFF'

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

def get_wbgt_style(wbgt):
    if wbgt is None or np.isnan(wbgt): return "#64748B", "#FFFFFF"
    w = float(wbgt)
    if w >= 32: c = '#991B1B'
    elif w >= 31: c = '#DC2626'
    elif w >= 30: c = '#EA580C'
    elif w >= 28: c = '#D97706'
    elif w >= 25: c = '#CA8A04'
    else: c = '#059669'
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
    if val is None or np.isnan(val): return "#64748B", "#FFFFFF"
    v = float(val)
    if v >= 100.0: return "#581C87", "#FFFFFF"
    if v >= 80.0:  return "#09090B", "#FFFFFF"
    if v >= 50.0:  return "#991B1B", "#FFFFFF"
    if v >= 20.0:  return "#CA8A04", "#FFFFFF"
    if v >= 10.0:  return "#15803D", "#FFFFFF"
    if v >= 0.1:   return "#0369A1", "#FFFFFF"
    return "#64748B", "#FFFFFF"

def get_raintotal_style(val):
    if val is None or np.isnan(val): return "#64748B", "#FFFFFF"
    v = float(val)
    if v >= 350.0: return "#09090B", "#FFFFFF"
    if v >= 200.0: return "#991B1B", "#FFFFFF"
    if v >= 100.0: return "#EA580C", "#FFFFFF"
    if v >= 50.0:  return "#CA8A04", "#FFFFFF"
    if v >= 0.1:   return "#0369A1", "#FFFFFF"
    return "#64748B", "#FFFFFF"

# 4. 指定風速色調映射
def get_wind_style(wind):
    if wind is None or np.isnan(wind): return "#64748B", "#FFFFFF"
    w = float(wind)
    if w >= 118.0: return "#DC2626", "#FFFFFF" # 颶風 red
    if w >= 88.0:  return "#F97316", "#FFFFFF" # 暴風 orange
    if w >= 63.0:  return "#F59E0B", "#FFFFFF" # 烈風 amber
    if w >= 41.0:  return "#0284C7", "#FFFFFF" # 強風 blue
    if w >= 31.0:  return "#10B981", "#FFFFFF" # 清勁 green
    return "#64748B", "#FFFFFF"                # 清勁以下 gray

def get_pressure_style(p):
    if p is None or np.isnan(p): return "#64748B", "#FFFFFF"
    p = float(p)
    if p >= 1025.0: c = '#1E3A8A'
    elif p >= 1020.0: c = '#2563EB'
    elif p >= 1015.0: c = '#0284C7'
    elif p >= 1010.0: c = '#059669'
    elif p >= 1005.0: c = '#D97706'
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
    conn.commit()
    conn.close()

def save_to_db(stations, pub_time_str, pub_timestamp):
    conn = sqlite3.connect(DB_PATH, timeout=10)
    c = conn.cursor()
    cutoff_ts = pub_timestamp - 93600
    c.execute("DELETE FROM weather_history WHERE timestamp < ?", (cutoff_ts,))
    
    for name, data in stations.items():
        row = (pub_time_str, pub_timestamp, name, 
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
# 繪製各項分佈底圖（全新現代設計）
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

    pressure_legend = [
        ("≥1025 hPa", '#1E3A8A'), ("1020-1024 hPa", '#2563EB'),
        ("1015-1019 hPa", '#0284C7'), ("1010-1014 hPa", '#059669'),
        ("1005-1009 hPa", '#D97706'), ("1000-1004 hPa", '#EA580C'),
        ("<1000 hPa", '#DC2626')
    ]

    wind_legend = [
        ("≥118 颶風", "#DC2626"),
        ("88-117 暴風", "#F97316"),
        ("63-87 烈風", "#F59E0B"),
        ("41-62 強風", "#0284C7"),
        ("31-40 清勁", "#10B981"),
        ("0-30 清勁以下", "#64748B")
    ]

    tasks = [
        ('temp_now', '實時氣溫分佈', '°C', lambda v: get_temp_style(v), temp_legend),
        ('apparent', '實時體感溫度分佈', '°C', lambda v: get_temp_style(v), temp_legend),
        ('wbgt', '實時暑熱指數 WBGT', '°C', lambda v: get_wbgt_style(v), [
            ("≥32° 極度危險", '#991B1B'), ("≥31° 非常危險", '#DC2626'),
            ("≥30° 危險", '#EA580C'), ("≥28° 警戒", '#D97706'),
            ("≥25° 注意", '#CA8A04'), ("<25° 正常", '#059669')
        ]),
        ('temp_max', '今日最高氣溫', '°C', lambda v: get_temp_style(v), temp_legend),
        ('temp_min', '今日最低氣溫', '°C', lambda v: get_temp_style(v), temp_legend),
        ('humidity', '實時相對濕度', '%', lambda v: get_humidity_style(v), [
            ("≥95% 非常潮濕", PALETTE['b-blue']), ("88-94% 潮濕", PALETTE['b-sky']),
            ("50-87% 正常", PALETTE['b-green']), ("30-49% 乾燥", PALETTE['b-amber']),
            ("<30% 非常乾燥", PALETTE['b-orange'])
        ]),
        ('rain_h', '過去一小時降水量', 'mm', get_rain1h_style, [
            ("≥100 mm", "#581C87"), ("80-99.9 mm", "#09090B"), ("50-79.9 mm", "#991B1B"),
            ("20-49.9 mm", "#CA8A04"), ("10-19.9 mm", "#15803D"), ("0.1-9.9 mm", "#0369A1"), ("0 mm", "#64748B")
        ]),
        ('rain_d', '今日累積降水量', 'mm', get_raintotal_style, [
            ("≥350 mm", "#09090B"), ("200-349.9 mm", "#991B1B"), ("100-199.9 mm", "#EA580C"),
            ("50-99.9 mm", "#CA8A04"), ("0.1-49.9 mm", "#0369A1"), ("0 mm", "#64748B")
        ]),
        ('wind_speed', '平均風速分佈', 'km/h', get_wind_style, wind_legend),
        ('gust_speed', '最高陣風分佈', 'km/h', get_wind_style, wind_legend),
        ('pressure', '海平面氣壓分佈', 'hPa', get_pressure_style, pressure_legend),
        ('wind_dir', '實時風向與風速分佈', '', None, None)
    ]

    for key, title, unit, style_fn, legend_items in tasks:
        fig, ax = plt.subplots(figsize=(11, 14), dpi=140)
        draw_base_map(ax, bg_color='#EAF2F8')

        # 頂部浮動標題欄 (現代質感設計)
        title_box = FancyBboxPatch(
            (0.03, 0.925), 0.58, 0.062,
            transform=ax.transAxes,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            facecolor='#FFFFFF', edgecolor='#CBD5E1',
            linewidth=1.2, alpha=0.98, zorder=12
        )
        ax.add_patch(title_box)

        ax.text(0.045, 0.962, f"澳門氣象分佈  •  {title}", transform=ax.transAxes,
                fontsize=17.5, fontweight='bold', color='#0F172A', va='center', zorder=13)
        ax.text(0.045, 0.938, f"資料來源：SMG XML  |  數據發布：{update_time_str}", transform=ax.transAxes,
                fontsize=11.5, color='#64748B', va='center', zorder=13)

        for name, data in stations.items():
            val = data.get(key)
            lon, lat = data['lon'], data['lat']
            is_bridge = name in BRIDGE_STATIONS

            if key in ['temp_now', 'temp_max', 'temp_min', 'humidity', 'rain_h', 'rain_d', 'apparent', 'wbgt', 'pressure'] and is_bridge:
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
                        bbox=dict(boxstyle="round,pad=0.25", facecolor='white', edgecolor='#CBD5E1', alpha=0.9), zorder=6)
            else:
                bg_color, font_color = style_fn(val)
                val_text = f"{val:g}"

                # 數據浮動藥丸卡片
                ax.annotate(
                    f"{val_text}{unit}",
                    xy=(lon, lat),
                    ha='center', va='center',
                    fontsize=12.5,
                    fontweight='bold',
                    color=font_color,
                    bbox=dict(
                        boxstyle="round,pad=0.42,rounding_size=0.4",
                        facecolor=bg_color,
                        edgecolor="#FFFFFF",
                        linewidth=1.8,
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
                    fontsize=10.5,
                    fontweight='bold',
                    color='#334155',
                    bbox=dict(
                        boxstyle="round,pad=0.25,rounding_size=0.2",
                        facecolor='#FFFFFF',
                        edgecolor='#CBD5E1',
                        linewidth=0.8,
                        alpha=0.95
                    ),
                    zorder=7
                )

        if legend_items:
            patches = [mpatches.Patch(facecolor=c, label=lbl, edgecolor='#CBD5E1', linewidth=0.8) for lbl, c in legend_items]
            leg = ax.legend(
                handles=patches, title="分級標準", loc='upper right',
                bbox_to_anchor=(0.97, 0.99),
                title_fontsize=11.5, fontsize=10,
                frameon=True, facecolor='white', edgecolor='#CBD5E1', framealpha=0.98
            )
            leg.set_zorder(12)
            leg.get_title().set_fontweight('bold')

        footer_info = f"澳門氣象局 (SMG) 實時數據網絡  |  渲染時間: {datetime.now(MACAU_TZ).strftime('%Y-%m-%d %H:%M')}"
        ax.text(0.035, 0.02, footer_info, transform=ax.transAxes, fontsize=10.5, color='#64748B',
                bbox=dict(boxstyle="round,pad=0.38", facecolor='white', edgecolor='#CBD5E1', linewidth=0.8, alpha=0.96), zorder=12)

        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        plt.savefig(f"static/images/{key}.png", dpi=140, bbox_inches='tight')
        plt.close(fig)

    print("✅ 全要素現代風地圖分佈圖渲染完成！")

# ==========================================
# 輔助：智慧斷線與氣象風羽標繪製
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

def draw_meteorological_wind_barb(ax, x_num, y_pos, deg, speed_kmh, scale_hours=0.45, scale_y=1.0):
    """
    3. 繪製標準氣象風標 (Wind Barb)
    風速單位：km/h。
    1 三角旗 = 50 km/h，1 長羽 = 20 km/h，1 短羽 = 10 km/h。
    """
    if deg is None or np.isnan(deg):
        return

    # 氣象風向為來源方向，羽毛在來向尾端
    rad = np.radians(deg)
    cos_d, sin_d = np.cos(rad), np.sin(rad)
    
    # 旋轉輔助
    def rotate_point(px, py):
        # 預設向上為北
        rx = px * cos_d + py * sin_d
        ry = -px * sin_d + py * cos_d
        return (rx * scale_hours / 24.0, ry * scale_y)

    # 繪製主桿 (長度為 1 單位)
    p0 = rotate_point(0, -0.2) # 箭尾
    p1 = rotate_point(0, 0.8)  # 箭頭點
    ax.plot([x_num + p0[0], x_num + p1[0]], [y_pos + p0[1], y_pos + p1[1]], color='#1E293B', lw=1.4, zorder=6)

    # 依照風速添加羽標（從尾端向桿身排列）
    remain = speed_kmh
    flags = int(remain // 50)
    remain %= 50
    long_bars = int(remain // 20)
    remain %= 20
    short_bars = int(remain // 10)

    cur_pos = 0.8
    barb_length = 0.35

    # 繪製三角旗
    for _ in range(flags):
        b1 = rotate_point(0, cur_pos)
        b2 = rotate_point(barb_length, cur_pos - 0.08)
        b3 = rotate_point(0, cur_pos - 0.16)
        poly = Polygon([
            [x_num + b1[0], y_pos + b1[1]],
            [x_num + b2[0], y_pos + b2[1]],
            [x_num + b3[0], y_pos + b3[1]]
        ], closed=True, facecolor='#1E293B', edgecolor='#1E293B', zorder=6)
        ax.add_patch(poly)
        cur_pos -= 0.18

    # 繪製長羽
    for _ in range(long_bars):
        b1 = rotate_point(0, cur_pos)
        b2 = rotate_point(barb_length, cur_pos + 0.05)
        ax.plot([x_num + b1[0], x_num + b2[0]], [y_pos + b1[1], y_pos + b2[1]], color='#1E293B', lw=1.3, zorder=6)
        cur_pos -= 0.14

    # 繪製短羽
    for _ in range(short_bars):
        b1 = rotate_point(0, cur_pos)
        b2 = rotate_point(barb_length * 0.5, cur_pos + 0.03)
        ax.plot([x_num + b1[0], x_num + b2[0]], [y_pos + b1[1], y_pos + b2[1]], color='#1E293B', lw=1.3, zorder=6)
        cur_pos -= 0.12

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

    GRID_BORDER_COLOR = '#CBD5E1'
    TEXT_COLOR = '#334155'
    header_update_text = f"更新時間 Update Time:{update_time_str.replace('-', '/')}"

    dir_deg_map = {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
        'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5, 'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
    }

    METRIC_DEFS = {
        'temp': {
            'label': '溫度 Temperature(℃)', 'color': '#A23B5A', 'default_step': 1.0, 'is_bar': False,
            'lines': [(35.5, '非常酷熱 35.5', '#EF4444'), (32.5, '酷熱 32.5', '#F97316'), (27.5, '炎熱 27.5', '#F59E0B'), (17.5, '和暖 17.5', '#10B981'), (12.5, '清涼 12.5', '#0EA5E9'), (7.5, '寒冷 7.5', '#3B82F6')]
        },
        'apparent': {
            'label': '體感溫度 Apparent Temp(℃)', 'color': '#EA580C', 'default_step': 1.0, 'is_bar': False,
            'lines': [(35.5, '非常酷熱 35.5', '#EF4444'), (32.5, '酷熱 32.5', '#F97316'), (27.5, '炎熱 27.5', '#F59E0B'), (17.5, '和暖 17.5', '#10B981'), (12.5, '清涼 12.5', '#0EA5E9'), (7.5, '寒冷 7.5', '#3B82F6')]
        },
        'wbgt': {
            'label': '暑熱指數 WBGT(℃)', 'color': '#D97706', 'default_step': 1.0, 'is_bar': False,
            'lines': [(32.0, '極度危險 32', '#991B1B'), (31.0, '非常危險 31', '#DC2626'), (30.0, '危險 30', '#EA580C'), (28.0, '警戒 28', '#F59E0B'), (25.0, '注意 25', '#EAB308')]
        },
        'humidity': {
            'label': '濕度 Humidity(%)', 'color': '#2A7B9B', 'default_step': 5.0, 'is_bar': False,
            'lines': [(95.0, '非常潮濕 95%', '#3B82F6'), (88.0, '潮濕 88%', '#0EA5E9'), (50.0, '正常 50%', '#10B981'), (30.0, '乾燥 30%', '#F59E0B')]
        },
        'wind_speed': {
            'label': '風速 Wind Speed(km/h)', 'color': '#0284C7', 'default_step': 4.0, 'is_bar': False,
            'lines': [(118.0, '颶風 118', '#DC2626'), (88.0, '暴風 88', '#F97316'), (63.0, '烈風 63', '#F59E0B'), (41.0, '強風 41', '#0284C7'), (31.0, '清勁 31', '#10B981')]
        },
        'rain': {
            'label': '降水量 Rainfall(mm)', 'color': '#1D4ED8', 'default_step': 2.0, 'is_bar': True,
            'lines': [(10.0, '10mm', '#149F37'), (20.0, '20mm', '#878E14'), (50.0, '50mm', '#921818')]
        },
        'pressure': {
            'label': '海平面氣壓 Pressure(hPa)', 'color': '#1E3A8A', 'default_step': 1.0, 'is_bar': False,
            'lines': [(1013.25, '標準大氣壓 1013.25', '#64748B')]
        }
    }

    # 全局 X 軸時間範圍：前推 24 小時
    global_max_dt = df['datetime'].max()
    x_end = global_max_dt
    x_start = x_end - pd.Timedelta(hours=24)

    # 2. 計算風速與各要素的動態 Y 軸（只留 1 個單位給風標，動態步長）
    UNIFIED_Y_LIMITS = {}
    for col, cfg in METRIC_DEFS.items():
        sub_df = df.copy()
        if col == 'wind_speed':
            all_v = pd.concat([sub_df['wind_speed'].dropna(), sub_df['gust_speed'].dropna()])
        else:
            all_v = sub_df[col].dropna()

        if all_v.empty:
            UNIFIED_Y_LIMITS[col] = (0, 10, cfg['default_step'])
            continue

        y_min, y_max = all_v.min(), all_v.max()

        if col == 'wind_speed':
            # 依風速跨度自動選擇合適步長 (2, 4, 5, 10)
            span = max(y_max, 10)
            if span > 100: step = 15.0
            elif span > 60: step = 10.0
            elif span > 30: step = 5.0
            else: step = 4.0

            # 2. 需求修復：只需預留 1 個間距空間給風標
            plot_ymin = -step * 1.0
            plot_ymax = np.ceil(y_max / step) * step + step
            UNIFIED_Y_LIMITS[col] = (plot_ymin, plot_ymax, step)
        elif col in ['temp', 'apparent', 'wbgt', 'pressure']:
            step = 1.0
            plot_ymin = np.floor(y_min) - 1
            plot_ymax = np.ceil(y_max) + 1
            UNIFIED_Y_LIMITS[col] = (plot_ymin, plot_ymax, step)
        else:
            step = cfg['default_step']
            plot_ymin = max(0, np.floor(y_min / step) * step - step)
            plot_ymax = np.ceil(y_max / step) * step + step
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
            metrics = ['temp', 'apparent', 'wbgt', 'humidity', 'wind_speed', 'rain']
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
            else:
                ax.plot(plot_df['datetime'], plot_df[col], color=cfg['color'], linewidth=1.6, antialiased=True, label=col.capitalize() if col != 'wind_speed' else 'Wind Speed')

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
                if col == 'wind_speed' and v < 0: return "" # 隱藏預留給風標的負數刻度
                return f"{int(v)}" if float(v).is_integer() else f"{v:.1f}"

            ax.set_yticklabels([y_fmt(y) for y in y_ticks], fontsize=13.5, color=cfg['color'], family='monospace')

            # 警戒標註線
            if 'lines' in cfg:
                for threshold_val, line_label, line_color in cfg['lines']:
                    if 0 <= threshold_val < plot_ymax:
                        ax.axhline(y=threshold_val, color=line_color, linestyle='--', linewidth=0.9, alpha=0.6, zorder=2)
                        ax.text(x_end, threshold_val, f" {line_label}", color=line_color, 
                                va='center', ha='left', fontsize=9.5, alpha=0.85, clip_on=True)

            # 3. 風向改為整點繪製標準氣象風標 (Wind Barb)
            if col == 'wind_speed':
                arrow_df = st_df.dropna(subset=['wind_dir'])
                barb_y = plot_ymin + 0.5 * step
                
                # 篩選整點整刻數據 (避免密集)
                hourly_candidates = arrow_df.copy()
                hourly_candidates['minute'] = hourly_candidates['datetime'].dt.minute
                hourly_candidates['date_hour'] = hourly_candidates['datetime'].dt.strftime('%Y-%m-%d %H')
                
                # 尋找最接近整點 (0 分) 的點
                near_hour = hourly_candidates[(hourly_candidates['minute'] <= 6) | (hourly_candidates['minute'] >= 54)].copy()
                if not near_hour.empty:
                    near_hour['diff_0'] = near_hour['minute'].apply(lambda m: min(m, 60 - m))
                    sampled_barbs = near_hour.sort_values('diff_0').groupby('date_hour').first().reset_index()
                    
                    for _, r in sampled_barbs.iterrows():
                        d_t = r['datetime']
                        if d_t < x_start or d_t > x_end:
                            continue
                        w_dir = r.get('wind_dir')
                        w_spd = r.get('wind_speed', 0.0)
                        try: deg = float(w_dir)
                        except: deg = dir_deg_map.get(str(w_dir).upper(), None)

                        if deg is not None:
                            t_num = mdates.date2num(d_t)
                            draw_meteorological_wind_barb(ax, t_num, barb_y, deg, w_spd, scale_hours=0.45, scale_y=step * 0.45)

            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[15, 30, 45]))

            def format_hour_ticks(x, pos=None):
                dt = mdates.num2date(x)
                hour_str = f"{dt.strftime('%H')}H"
                if dt.hour == 0:
                    return f"{hour_str}\n{dt.strftime('%d %b')}"
                return hour_str

            ax.xaxis.set_major_formatter(plt.FuncFormatter(format_hour_ticks))
            plt.setp(ax.get_xticklabels(), fontsize=12.5, color=TEXT_COLOR, family='monospace')

            ax.grid(True, which='major', linestyle=':', linewidth=1.3, color=GRID_BORDER_COLOR, alpha=0.95)
            ax.tick_params(axis='both', which='major', length=5, width=1.0, color=GRID_BORDER_COLOR, direction='out')
            ax.tick_params(axis='x', which='minor', length=3, width=0.8, color=GRID_BORDER_COLOR, direction='out')

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(GRID_BORDER_COLOR)
                spine.set_linewidth(1.2)

            ax.set_ylabel(cfg['label'], fontsize=14, color=cfg['color'], labelpad=12)
            ax.set_xlabel('澳門時間(時) Local Time (Hour)', fontsize=14, color=TEXT_COLOR, labelpad=14)

            ax.text(0.0, 1.02, header_station_text, transform=ax.transAxes, fontsize=15, va='bottom', ha='left', color=TEXT_COLOR)
            ax.text(1.0, 1.02, header_update_text, transform=ax.transAxes, fontsize=15, va='bottom', ha='right', color=TEXT_COLOR)

            plt.tight_layout()
            plt.savefig(f"static/images/line_{station_name}_{col}.png", dpi=120, bbox_inches='tight')
            plt.close(fig)

    # 跨海大橋專用折線圖
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
        _, plot_ymax, step = UNIFIED_Y_LIMITS['wind_speed']
        y_ticks = np.arange(0, plot_ymax + step * 0.1, step)
        ax.set_ylim(0, plot_ymax)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(int(y)) for y in y_ticks], fontsize=13.5, color='#0284C7', family='monospace')

        for threshold_val, line_label, line_color in METRIC_DEFS['wind_speed']['lines']:
            if 0 < threshold_val < plot_ymax:
                ax.axhline(y=threshold_val, color=line_color, linestyle='--', linewidth=0.9, alpha=0.6, zorder=2)
                ax.text(x_end, threshold_val, f" {line_label}", color=line_color, va='center', ha='left', fontsize=9.5, alpha=0.85)

        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[15, 30, 45]))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_hour_ticks))
        plt.setp(ax.get_xticklabels(), fontsize=12.5, color=TEXT_COLOR, family='monospace')

        ax.grid(True, which='major', linestyle=':', linewidth=1.3, color=GRID_BORDER_COLOR, alpha=0.95)
        ax.set_ylabel(type_label, fontsize=14, color='#0284C7', labelpad=12)
        ax.set_xlabel('澳門時間(時) Local Time (Hour)', fontsize=14, color=TEXT_COLOR, labelpad=14)

        ax.legend(loc='upper left', ncol=4, frameon=True, facecolor='white', edgecolor=GRID_BORDER_COLOR, fontsize=11)
        ax.text(0.0, 1.02, title_prefix, transform=ax.transAxes, fontsize=15, va='bottom', ha='left', color=TEXT_COLOR, fontweight='bold')
        ax.text(1.0, 1.02, header_update_text, transform=ax.transAxes, fontsize=15, va='bottom', ha='right', color=TEXT_COLOR)

        plt.tight_layout()
        plt.savefig(f"static/images/{filename}", dpi=120, bbox_inches='tight')
        plt.close(fig)

    print("✅ 時序趨勢圖 (包含標準風羽標與動態 Y 軸) 全部渲染完成！")

# ==========================================
# 5. 新增：24 小時溫差折線圖
# ==========================================
def render_24h_temp_range_charts():
    """
    計算並繪製各測站過去 24 小時的氣溫日較差 / 溫差趨勢圖 (最大值 - 最小值及實時動態溫差)
    生成：
      - static/images/line_{station_name}_temp_range.png (單站動態)
      - static/images/line_macau_24h_temp_range.png (全澳溫差一覽)
    """
    conn = sqlite3.connect(DB_PATH, timeout=10)
    df = pd.read_sql_query("SELECT * FROM weather_history ORDER BY timestamp ASC", conn)
    conn.close()

    if df.empty: return

    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Macau').dt.tz_localize(None)
    global_max_dt = df['datetime'].max()
    x_end = global_max_dt
    x_start = x_end - pd.Timedelta(hours=24)
    df_24h = df[df['datetime'] >= x_start].copy()

    GRID_BORDER_COLOR = '#CBD5E1'
    TEXT_COLOR = '#334155'
    header_update_text = f"更新時間 Update Time:{update_time_str.replace('-', '/')}"

    # 1. 繪製全澳代表性測站溫差折線圖
    fig, ax = plt.subplots(figsize=(15, 6.5), dpi=120)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    station_palette = {
        "大潭山": "#E11D48", "大炮台": "#2563EB", 
        "紀念孫中山公園": "#059669", "路環": "#D97706",
        "九澳": "#7C3AED", "黑沙環": "#0284C7"
    }

    for st_name, color in station_palette.items():
        st_df = df_24h[df_24h['station'] == st_name].dropna(subset=['temp']).copy()
        if st_df.empty: continue
        # 繪製各站相對於 24 小時基底溫度的動態
        plot_df = insert_nans_on_gap(st_df, 'temp', max_gap_minutes=25)
        ax.plot(plot_df['datetime'], plot_df['temp'], label=st_name, color=color, linewidth=1.6, antialiased=True)

    ax.set_xlim(x_start, x_end)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[15, 30, 45]))

    def format_hour_ticks(x, pos=None):
        dt = mdates.num2date(x)
        hour_str = f"{dt.strftime('%H')}H"
        if dt.hour == 0:
            return f"{hour_str}\n{dt.strftime('%d %b')}"
        return hour_str

    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_hour_ticks))
    plt.setp(ax.get_xticklabels(), fontsize=12.5, color=TEXT_COLOR, family='monospace')

    ax.grid(True, which='major', linestyle=':', linewidth=1.3, color=GRID_BORDER_COLOR, alpha=0.95)
    ax.set_ylabel('溫度 Temperature(℃)', fontsize=14, color='#A23B5A', labelpad=12)
    ax.set_xlabel('澳門時間(時) Local Time (Hour)', fontsize=14, color=TEXT_COLOR, labelpad=14)
    ax.legend(loc='upper left', ncol=6, frameon=True, facecolor='white', edgecolor=GRID_BORDER_COLOR, fontsize=11)

    ax.text(0.0, 1.02, "澳門各區 24 小時氣溫變化與溫差對比", transform=ax.transAxes, fontsize=15, va='bottom', ha='left', color=TEXT_COLOR, fontweight='bold')
    ax.text(1.0, 1.02, header_update_text, transform=ax.transAxes, fontsize=15, va='bottom', ha='right', color=TEXT_COLOR)

    plt.tight_layout()
    plt.savefig("static/images/line_macau_24h_temp_range.png", dpi=120, bbox_inches='tight')
    plt.close(fig)

    # 2. 獨立生成每個測站過去 24 小時累積最高/最低溫差區間圖
    for st_name in STATION_COORDS.keys():
        if st_name in BRIDGE_STATIONS: continue
        st_df = df_24h[df_24h['station'] == st_name].dropna(subset=['temp']).copy()
        if st_df.empty: continue

        fig, ax = plt.subplots(figsize=(15, 6.5), dpi=120)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        t_min = st_df['temp'].min()
        t_max = st_df['temp'].max()
        t_range = round(t_max - t_min, 1)

        plot_df = insert_nans_on_gap(st_df, 'temp', max_gap_minutes=25)
        ax.plot(plot_df['datetime'], plot_df['temp'], color='#E11D48', linewidth=1.8, label=f'實時氣溫 (24H 溫差: {t_range}℃)')
        ax.axhline(y=t_max, color='#DC2626', linestyle=':', linewidth=1.2, label=f'24H 最高溫: {t_max}℃')
        ax.axhline(y=t_min, color='#0284C7', linestyle=':', linewidth=1.2, label=f'24H 最低溫: {t_min}℃')
        ax.fill_between(plot_df['datetime'], t_min, plot_df['temp'], color='#FDA4AF', alpha=0.25)

        ax.set_xlim(x_start, x_end)
        ax.set_ylim(math.floor(t_min) - 1, math.ceil(t_max) + 1)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[15, 30, 45]))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_hour_ticks))
        plt.setp(ax.get_xticklabels(), fontsize=12.5, color=TEXT_COLOR, family='monospace')

        ax.grid(True, which='major', linestyle=':', linewidth=1.3, color=GRID_BORDER_COLOR, alpha=0.95)
        ax.set_ylabel('溫度 Temperature(℃)', fontsize=14, color='#A23B5A', labelpad=12)
        ax.set_xlabel('澳門時間(時) Local Time (Hour)', fontsize=14, color=TEXT_COLOR, labelpad=14)

        code = STATION_COORDS[st_name].get("code", "")
        header_st = f"{st_name}({code})" if code else st_name
        ax.text(0.0, 1.02, f"{header_st} - 24小時氣溫與溫差走勢", transform=ax.transAxes, fontsize=15, va='bottom', ha='left', color=TEXT_COLOR)
        ax.text(1.0, 1.02, header_update_text, transform=ax.transAxes, fontsize=15, va='bottom', ha='right', color=TEXT_COLOR)
        ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor=GRID_BORDER_COLOR, fontsize=11)

        plt.tight_layout()
        plt.savefig(f"static/images/line_{st_name}_temp_range.png", dpi=120, bbox_inches='tight')
        plt.close(fig)

    print("✅ 24 小時溫差趨勢圖 (全澳綜合 + 各站明細) 生成完成！")

# ==========================================
# 繪製 24 小時紫外線折線圖
# ==========================================
def render_uv_trend_chart():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    df = pd.read_sql_query("SELECT * FROM uv_history ORDER BY timestamp ASC", conn)
    conn.close()

    if df.empty: return

    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Macau').dt.tz_localize(None)

    GRID_BORDER_COLOR = '#CBD5E1'
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
    ax.set_yticklabels([str(int(y)) for y in y_ticks], fontsize=13.5, color=UV_COLOR, family='monospace')

    uv_levels = [(11.0, '極高 11+', '#7C3AED'), (8.0, '甚高 8-10', '#DC2626'), (6.0, '高 6-7', '#F97316'), (3.0, '中 3-5', '#CA8A04')]
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
    plt.setp(ax.get_xticklabels(), fontsize=12.5, color=TEXT_COLOR, family='monospace')

    ax.grid(True, which='major', linestyle=':', linewidth=1.3, color=GRID_BORDER_COLOR, alpha=0.95)
    ax.tick_params(axis='both', which='major', length=5, width=1.0, color=GRID_BORDER_COLOR, direction='out')
    ax.tick_params(axis='x', which='minor', length=3, width=0.8, color=GRID_BORDER_COLOR, direction='out')

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(GRID_BORDER_COLOR)
        spine.set_linewidth(1.2)

    ax.set_ylabel('紫外線指數 UV Index', fontsize=14, color=UV_COLOR, labelpad=12)
    ax.set_xlabel('澳門時間(時) Local Time (Hour)', fontsize=14, color=TEXT_COLOR, labelpad=14)

    header_update_text = f"更新時間 Update Time:{update_time_str.replace('-', '/')}"
    ax.text(0.0, 1.02, "大潭山(TG) 紫外線指數 UV Index", transform=ax.transAxes, fontsize=15, va='bottom', ha='left', color=TEXT_COLOR)
    ax.text(1.0, 1.02, header_update_text, transform=ax.transAxes, fontsize=15, va='bottom', ha='right', color=TEXT_COLOR)

    plt.tight_layout()
    plt.savefig("static/images/line_uv_index.png", dpi=120, bbox_inches='tight')
    plt.close(fig)
    print("✅ 紫外線折線圖生成完成！")

# ==========================================
# 24 小時降雨量統計表
# ==========================================
def generate_24h_rainfall_summary():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    df = pd.read_sql_query("SELECT * FROM weather_history ORDER BY timestamp ASC", conn)
    conn.close()

    if df.empty: return

    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Macau')
    latest_time = df['datetime'].max()
    start_24h_window = latest_time - timedelta(hours=24)
    df_24h = df[df['datetime'] >= start_24h_window].copy()

    records = []
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

    summary_df = pd.DataFrame(records)
    summary_df.sort_values(by="Past_24H_Total_Rain (mm)", ascending=False, inplace=True)

    print("\n" + "=" * 65)
    print(f"📊 澳門過去 24 小時累積雨量統計表 (每小時 35 分定點抽樣加總)")
    print("=" * 65)
    print(summary_df.to_string(index=False))
    print("=" * 65 + "\n")

    summary_df.to_json("static/images/rainfall_24h_summary.json", orient="records", force_ascii=False, indent=2)

# ==========================================
# 程式進入點
# ==========================================
if __name__ == '__main__':
    init_db()
    stations_data = fetch_smg_weather()
    uv_data = fetch_smg_uvi()
    current_uvi = uv_data.get('uvi', 0.0) if uv_data else 0.0

    if stations_data:
        render_distribution_maps(stations_data, current_uvi)
        render_trend_charts()
        render_24h_temp_range_charts() # 5. 執行溫差圖生成
        render_uv_trend_chart()
        generate_24h_rainfall_summary()
        print(f"🎉 全部功能升級完畢！當前 XML 觀測時間：{update_time_str}")
    else:
        print("❌ 無法獲取 SMG XML 數據，請檢查網絡連接。")
