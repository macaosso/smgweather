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
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.dates as mdates
from PIL import Image
import geopandas as gpd
from shapely.geometry import box
import osmnx as ox

warnings.filterwarnings("ignore")

# ==========================================
# 建立輸出目錄與快取設定
# ==========================================
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
    'Noto Sans CJK TC', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei',
    'PingFang SC', 'Microsoft YaHei', 'SimHei', 'sans-serif'
]
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 測站坐標與基本配置
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
# 2. 各項要素色階映射函數 (依照最新色階)
# ==========================================
def get_temp_color(v):
    if v is None: return "#9E9E9E"
    if v >= 37.5: return "#8B0000"      # 極為酷熱 (Dark Red)
    if v >= 35.5: return "#D32F2F"      # 非常酷熱 (Furnace Red)
    if v >= 32.5: return "#E64A19"      # 酷熱 (Redstone)
    if v >= 27.5: return "#FF8F00"      # 炎熱 (Orange)
    if v >= 22.5: return "#FFA726"      # 溫暖 (Coral Orange)
    if v >= 17.5: return "#FFCA28"      # 和暖 (Amber)
    if v >= 12.5: return "#7CB342"      # 清涼 (Pickle Green)
    if v >= 7.5:  return "#1E88E5"      # 寒冷 (Denim Blue)
    if v >= 3.5:  return "#1565C0"      # 非常寒冷 (Midnight Blue)
    return "#4A148C"                    # 極為寒冷 (Plum Purple)

def get_temp_diff_color(v):
    if v is None: return "#9E9E9E"
    if v > 0: return "#D32F2F"          # + furnace
    if v == 0: return "#9E9E9E"         # 0 gray
    return "#1565C0"                    # - midnight blue

def get_wbgt_color(v):
    if v is None: return "#9E9E9E"
    if v >= 31.5: return "#8B0000"      # 極危
    if v >= 30.5: return "#F57C00"      # 危險 (orange)
    if v >= 29.5: return "#FFB300"      # 警戒 (amber)
    if v >= 27.5: return "#7CB342"      # 注意 (pickle green)
    return "#4FC3F7"                    # 正常 (light blue)

def get_wind_color(v):
    if v is None: return "#9E9E9E"
    if v >= 117.5: return "#D32F2F"     # 颶風 (red)
    if v >= 87.5:  return "#FB8C00"     # 暴風 (orange)
    if v >= 62.5:  return "#FFC107"     # 烈風 (amber)
    if v >= 40.5:  return "#1E88E5"     # 強風 (blue)
    if v >= 30.5:  return "#43A047"     # 清勁 (green)
    return "#9E9E9E"                    # 清勁以下 (gray)

def get_rain1h_color(v):
    if v is None: return "#9E9E9E"
    if v >= 100.0: return "#7B1FA2"     # 大暴雨 (purple)
    if v >= 50.0:  return "#E53935"     # 暴雨 (red)
    if v >= 20.0:  return "#1E88E5"     # 大雨 (blue)
    if v >= 10.0:  return "#43A047"     # 驟雨 (green)
    if v >= 0.1:   return "#81C784"     # 雨 (light green)
    return "#BDBDBD"                    # 0 gray

def get_rain24h_color(v):
    if v is None: return "#9E9E9E"
    if v >= 350.0: return "#212121"     # black
    if v >= 200.0: return "#D32F2F"     # red
    if v >= 100.0: return "#FB8C00"     # orange
    if v >= 50.0:  return "#FFC107"     # amber
    if v >= 0.1:   return "#4FC3F7"     # light blue
    return "#BDBDBD"                    # 0 gray

def get_pressure_color(v):
    if v is None: return "#9E9E9E"
    if v >= 1030.0: return "#8B0000"
    if v >= 1025.0: return "#C62828"
    if v >= 1020.0: return "#E65100"
    if v >= 1016.0: return "#F57C00"
    if v >= 1012.0: return "#FBC02D"
    if v >= 1008.0: return "#7CB342"
    if v >= 1000.0: return "#00897B"
    if v >= 990.0:  return "#0288D1"
    if v >= 980.0:  return "#1565C0"
    if v >= 970.0:  return "#283593"
    if v >= 960.0:  return "#6A1B9A"
    return "#311B92"

def get_humidity_color(h):
    if h is None: return "#9E9E9E"
    if h >= 95: return "#7B1FA2"
    if h >= 85: return "#1565C0"
    if h >= 70: return "#43A047"
    if h >= 40: return "#FB8C00"
    return "#D32F2F"

# ==========================================
# 3. 初始化底圖 (OSM 下載並裁剪)
# ==========================================
print("正在初始化下載地圖底圖與 OSM 大橋...")
bbox = (113.48, 22.08, 113.68, 22.25)
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
    print(f"底圖下載注意/失敗: {e}")

if bridges is None:
    bridges = gpd.GeoDataFrame()

# ==========================================
# 4. 16:9 底圖與繪圖核心函數
# ==========================================
def draw_base_map(ax):
    # 底圖配色：溫和現代淺色調
    bg_water = '#DCEEFB'     # 水體淺藍
    land_zhuhai = '#F7F4EB'  # 珠海陸地淡黃白
    land_macau = '#FFFFFF'   # 澳門陸地純白
    border_color = '#D1D5DB' # 柔和邊界線
    bridge_color = '#B0BEC5' # 橋樑線條

    ax.set_facecolor(bg_water)
    
    if zhuhai is not None:
        zhuhai.plot(ax=ax, color=land_zhuhai, edgecolor=border_color, linewidth=0.8, zorder=1)
    if macau is not None:
        macau.plot(ax=ax, color=land_macau, edgecolor=border_color, linewidth=1.0, zorder=2)
    if water is not None and not water.empty:
        water.plot(ax=ax, color=bg_water, edgecolor='none', zorder=3)
    if bridges is not None and not bridges.empty:
        bridges.plot(ax=ax, color=bridge_color, edgecolor='white', linewidth=1.8, zorder=4)

    # 精準 16:9 坐標跨度
    ax.set_xlim(113.485, 113.685)
    ax.set_ylim(22.105, 22.235)
    ax.set_aspect(1.0 / 0.927) # 緯度正形補償，保證不被壓扁
    ax.set_axis_off()

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
# 數據獲取與解析
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
                data['temp_diff_24h'] = round(cur_temp - float(row[0]), 1)
            else:
                data['temp_diff_24h'] = None
        conn.close()
    except Exception as e:
        print(f"⚠️ 計算 24 小時溫差異常: {e}")

def fetch_smg_weather(current_uvi=0.0):
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
# 繪製各要素分佈底圖 (16:9 全新底圖邏輯)
# ==========================================
def render_distribution_maps(stations, current_uvi=0.0):
    for st_name, d in stations.items():
        if d.get('apparent') is None:
            t = d.get('temp_now')
            h = d.get('humidity')
            w = d.get('wind_speed', 0.0)
            d['apparent'] = calculate_apparent_temperature(t, h, w) if (t is not None and h is not None) else None
            d['wbgt'] = estimate_wbgt_kmh(t, h, w, current_uvi) if (t is not None and h is not None) else None

    tasks = [
        ('temp_now', '分區氣溫 (°C)', get_temp_color),
        ('temp_diff_24h', '過去24小時氣溫差別 (°C)', get_temp_diff_color),
        ('apparent', '體感溫度 (°C)', get_temp_color),
        ('wbgt', '暑熱指數 WBGT (°C)', get_wbgt_color),
        ('temp_max', '今日最高氣溫 (°C)', get_temp_color),
        ('temp_min', '今日最低氣溫 (°C)', get_temp_color),
        ('humidity', '相對濕度 (%)', get_humidity_color),
        ('rain_h', '過去一小時降水 (mm)', get_rain1h_color),
        ('rain_d', '今日累積降水 (mm)', get_rain24h_color),
        ('wind_speed', '過去10分鐘平均風速 (km/h)', get_wind_color),
        ('gust_speed', '最高陣風分佈 (km/h)', get_wind_color),
        ('wind_dir', '實時風向與風速分佈 (km/h)', get_wind_color),
        ('pressure', '平均海平面氣壓 (hPa)', get_pressure_color),
    ]

    logo_path = 'MSSO(H).png' if os.path.exists('MSSO(H).png') else ('static/images/MSSO(H).png' if os.path.exists('static/images/MSSO(H).png') else None)

    for key, title_text, color_fn in tasks:
        fig = plt.figure(figsize=(16, 9), dpi=150)
        ax = fig.add_axes([0, 0, 1, 1])
        
        # 繪製 16:9 淺色清新現代底圖
        draw_base_map(ax)

        # 左上角標題與更新時間資訊卡
        info_box = dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='#CFD8DC', alpha=0.92, lw=1.2)
        time_text = f"●  更新時間  :  {update_time_str}\n●  觀測項目  :  {title_text}"
        ax.text(113.490, 22.228, time_text,
                fontsize=13, fontweight='bold', color='#263238',
                va='top', ha='left', zorder=10, bbox=info_box)

        # 站點數值與膠囊標籤
        for name, data in stations.items():
            val = data.get(key)
            lon, lat = data['lon'], data['lat']
            disp_name = data.get('disp', name)
            is_bridge = name in BRIDGE_STATIONS

            if key in ['temp_now', 'temp_diff_24h', 'temp_max', 'temp_min', 'humidity', 'rain_h', 'rain_d', 'apparent', 'wbgt', 'pressure'] and is_bridge:
                continue
            if key == 'pressure' and name not in PRESSURE_STATIONS:
                continue

            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                color = color_fn(val)

                if key in ['wind_speed', 'gust_speed', 'wind_dir']:
                    dir_map = {'N':0, 'NNE':22.5, 'NE':45, 'ENE':67.5, 'E':90, 'ESE':112.5, 'SE':135, 'SSE':157.5,
                               'S':180, 'SSW':202.5, 'SW':225, 'WSW':247.5, 'W':270, 'WNW':292.5, 'NW':315, 'NNW':337.5}
                    try: deg = float(data.get('wind_dir', 0))
                    except: deg = dir_map.get(str(data.get('wind_dir')).upper(), 0)
                    
                    spd_val = data.get('wind_speed' if key != 'gust_speed' else 'gust_speed', 0.0) or 0.0
                    arrow_poly = np.array([[0, 2.3], [-1.1, -1.3], [0, -0.4], [1.1, -1.3]])
                    t = (matplotlib.transforms.Affine2D().rotate_deg(deg).scale(0.0030).translate(lon, lat) + ax.transData)
                    ax.add_patch(Polygon(arrow_poly, closed=True, facecolor=color, edgecolor='#37474F', linewidth=1.2, transform=t, zorder=8))
                    
                    ax.text(lon, lat, f"{int(round(spd_val))}", fontsize=9.0, fontweight='heavy', color='#212121',
                            ha='center', va='center', zorder=9)
                    ax.text(lon, lat - 0.0055, disp_name, fontsize=8.5, fontweight='bold', color='#37474F',
                            ha='center', va='top', zorder=8,
                            path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
                elif key == 'temp_diff_24h':
                    d_val = float(val)
                    prefix = "+" if d_val > 0 else ""
                    badge_text = f"{prefix}{d_val:.1f}"
                    ax.text(lon, lat, badge_text, fontsize=9.5, fontweight='heavy', color='#FFFFFF',
                            ha='center', va='center', zorder=8,
                            bbox=dict(boxstyle='round,pad=0.35,rounding_size=0.5', facecolor=color, edgecolor='#37474F', linewidth=1.2))
                    ax.text(lon, lat - 0.0055, disp_name, fontsize=8.5, fontweight='bold', color='#37474F',
                            ha='center', va='top', zorder=8,
                            path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
                else:
                    if key in ['temp_now', 'apparent', 'wbgt', 'temp_max', 'temp_min', 'pressure']:
                        badge_text = f"{val:.1f}"
                    else:
                        badge_text = f"{int(round(val))}"

                    ax.text(lon, lat, badge_text, fontsize=9.5, fontweight='heavy', color='#212121',
                            ha='center', va='center', zorder=8,
                            bbox=dict(boxstyle='round,pad=0.35,rounding_size=0.5', facecolor=color, edgecolor='#37474F', linewidth=1.2))
                    ax.text(lon, lat - 0.0055, disp_name, fontsize=8.5, fontweight='bold', color='#37474F',
                            ha='center', va='top', zorder=8,
                            path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
            else:
                # 無數據時渲染點位與站名
                ax.scatter(lon, lat, color='#90A4AE', s=25, edgecolor='#37474F', zorder=7)
                ax.text(lon, lat - 0.0045, disp_name, fontsize=8, fontweight='bold', color='#546E7A',
                        ha='center', va='top', zorder=8,
                        path_effects=[pe.withStroke(linewidth=2, foreground='white')])

        # 右下角 MSSO Logo (縮小尺寸 zoom=0.12)
        if logo_path and os.path.exists(logo_path):
            try:
                logo_img = Image.open(logo_path)
                imagebox = OffsetImage(logo_img, zoom=0.12)
                ab = AnnotationBbox(imagebox, (0.935, 0.065), frameon=False, xycoords='axes fraction', zorder=12)
                ax.add_artist(ab)
            except Exception as e:
                print(f"Logo 載入失敗: {e}")
        else:
            ax.text(0.935, 0.055, "MSSO", transform=ax.transAxes,
                    fontsize=20, fontweight='heavy', color='#37474F', ha='center', va='center', zorder=12)

        out_file = f"static/images/{key}.png"
        plt.savefig(out_file, format='png', dpi=150)
        plt.close(fig)

    print("✅ 16:9 全要素氣象地圖渲染完畢！")

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
# 繪製各測站歷史時序折線圖 (保持原樣不變)
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
    for idx, row in df.iterrows():
        st = row['station']
        t_curr = row['temp']
        ts_curr = row['timestamp']
        if pd.isna(t_curr):
            temp_diff_series.append(np.nan)
            continue
        ts_24h = ts_curr - 86400
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
            UNIFIED_Y_LIMITS[col] = (0, 10, cfg['step'])
            continue

        y_min, y_max = all_v.min(), all_v.max()

        if col == 'wind_speed':
            v_range = y_max - y_min
            if v_range > 80: step = 10.0
            elif v_range > 40: step = 5.0
            elif v_range > 20: step = 4.0
            else: step = 2.0
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

                    barb_x, barb_y, barb_u, barb_v, barb_colors = [], [], [], [], []
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
                            u = -spd_kt * np.sin(rad)
                            v = -spd_kt * np.cos(rad)

                            barb_x.append(mdates.date2num(d_t))
                            barb_y.append(barb_base_y)
                            barb_u.append(u)
                            barb_v.append(v)
                            c = get_wind_color(w_spd)
                            barb_colors.append(c)

                    if barb_x:
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

    print("✅ 時序趨勢圖全部渲染完成！")

# ==========================================
# 繪製 24 小時紫外線折線圖 (保持原樣不變)
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
# 24 小時降雨量統計表 (保持原樣不變)
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
    
    # 1. 獲取 UVI 資料
    uv_data = fetch_smg_uvi()
    current_uvi = uv_data.get('uvi', 0.0) if uv_data else 0.0

    # 2. 獲取 SMG 氣象數據
    stations_data = fetch_smg_weather(current_uvi=current_uvi)

    if stations_data:
        generate_24h_rainfall_summary(stations_data)
        render_distribution_maps(stations_data, current_uvi)
        render_trend_charts()
        render_uv_trend_chart()
        print(f"🎉 全部功能更新完畢！當前 XML 觀測時間：{update_time_str}")
    else:
        generate_24h_rainfall_summary(None)
        print("❌ 無法獲取 SMG XML 數據，已確保預備狀態檔案輸出。")
