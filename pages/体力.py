# app.py
import csv
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from io import StringIO

import streamlit as st

JST = ZoneInfo("Asia/Tokyo")

st.set_page_config(page_title="体力回復予測（単体）", layout="centered")
st.title("体力回復予測（単体）")

# --- CSV データをソース内に埋め込む ---
CSV_DATA = """event_name,max_stamina,recovery_minutes
鬼市,30,7.5
商会拠点(野外),30,7.5
始皇帝,30,12
大富豪,30,15
五雄争覇,30,16
海神島,30,16
絶代双驕,20,20
怒涛斬魁,20,20
海上争覇,10,60
敦煌石窟,30,12
封神演義,30,12
木蘭戦記,30,15
山海伏獣,30,16
山水遊行,10,60
菊下楼経営（料理人対決）,30,15
甄嬛伝（1st）,30,20
甄嬛伝（2nd）,20,40
開封即配,30,15
懐仙歌,30,15
山河の饗宴,30,15
降魔護世,20,30
黒風砦,10,90
質屋経営,60,10
蘭若寺,30,10
蜀山,30,240
呉剛柱を切る,50,15
登龍門,30,30
大暴れ孫悟空,30,30
女媧補天,30,30
灯篭通り,30,30
深海至宝,30,30
百鬼夜行,10,120
飲み物屋,120,5
美人大会,500,1
楽坊,50,10
瑞獣降福,150,3.3333
旧正月料理,100,6
"""

# --- ヘルパー関数 ---
def load_events_from_string(csv_text):
    events = []
    f = StringIO(csv_text)
    reader = csv.DictReader(f)
    for row in reader:
        name = row.get("event_name") or row.get("name")
        if not name:
            continue
        max_stamina = row.get("max_stamina")
        recovery = row.get("recovery_minutes")
        events.append({
            "name": name,
            "max_stamina": int(float(max_stamina)) if max_stamina else None,
            "recovery_minutes": float(recovery) if recovery else 0.0,
        })
    return events

def calc_recovery_local(event_info, current):
    max_stamina = event_info["max_stamina"]
    t = event_info["recovery_minutes"]
    if max_stamina is None:
        raise ValueError("This event has no max stamina")
    if current > max_stamina:
        raise ValueError("Current stamina exceeds max")
    need = max_stamina - current
    now = datetime.now(JST)
    if need <= 0:
        time_str = f"{now.month}/{now.day} {now.strftime('%H:%M')}"
        return {"need": 0, "min": time_str, "max": time_str}
    min_minutes = (need - 1) * t
    max_minutes = need * t
    min_time = now + timedelta(minutes=min_minutes)
    max_time = now + timedelta(minutes=max_minutes)
    min_str = f"{min_time.month}/{min_time.day} {min_time.strftime('%H:%M')}"
    max_str = f"{max_time.month}/{max_time.day} {max_time.strftime('%H:%M')}"
    return {"need": need, "min": min_str, "max": max_str}

def format_range(min_str, max_str):
    if not min_str or not max_str:
        return ""
    min_parts = str(min_str).split(" ")
    max_parts = str(max_str).split(" ")
    min_date, min_time = (min_parts[0], min_parts[1]) if len(min_parts) >= 2 else (min_parts[0], "")
    max_date, max_time = (max_parts[0], max_parts[1]) if len(max_parts) >= 2 else (max_parts[0], "")
    if min_date == max_date:
        return f"{min_date} {min_time}〜{max_time}"
    return f"{min_str} 〜 {max_str}"

# --- データ読み込み（ソース内 CSV を使う） ---
events = load_events_from_string(CSV_DATA)

if not events:
    st.warning("イベントデータが読み込めませんでした。")
    st.stop()

names = [e["name"] for e in events]

# --- UI（単一表示） ---
selected = st.selectbox("イベントを選択", ["（選択しない）"] + names, index=0)

# スライダーの max は選択イベントの max_stamina を使う（選択なしは 100）
selected_max = next((e["max_stamina"] for e in events if e["name"] == selected), 100)
if selected_max is None:
    selected_max = 100

current = st.slider("現在体力", min_value=0, max_value=selected_max, value=0)

if st.button("計算する"):
    if selected == "（選択しない）":
        st.info("イベントを選んでください。")
    else:
        ev = next((e for e in events if e["name"] == selected), None)
        if not ev:
            st.error("選択されたイベント情報が見つかりません。")
        else:
            try:
                res = calc_recovery_local(ev, current)
                st.markdown(f"**必要回復数：{res['need']}**")
                st.markdown(f"**全回復予想：{format_range(res['min'], res['max'])}**")
            except Exception as ex:
                st.error(f"計算エラー: {ex}")
