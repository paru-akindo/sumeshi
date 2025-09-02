import streamlit as st
import pandas as pd
import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

st.title("いつ段位アップ？")

# ── 昇段に必要な修練値 ──
required_training = {
    "錬気": {1:41000,2:288200,3:497700},
    "築基": {1:747600,2:1058000,3:1414000},
    "結丹": {1:1874000,2:2364000,3:2980000},
    "元嬰": {1:3667000,2:4466000,3:5341000,4:6346000,5:7491000,6:8730000},
    "化神": {1:10120000,2:11680000,3:13350000,4:15200000,5:17180000,6:19340000},
    "錬虚": {1:16190000,2:18300000,3:26970000,4:63300000,5:68430000,6:73540000},
    "合体": {1:79070000,2:84680000,3:0,4:0,5:0,6:0,7:0,8:0,9:0},
    "大乗": {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
}

# ── 周天あたり修練速度 ──
training_speeds = {
    "錬気": {1:100,2:200,3:300},
    "築基": {1:400,2:500,3:600},
    "結丹": {1:700,2:800,3:900},
    "元嬰": {1:1000,2:1100,3:1200,4:1300,5:1400,6:1500},
    "化神": {1:1600,2:1700,3:1800,4:1900,5:2000,6:2100},
    "錬虚": {1:2200,2:2300,3:2400,4:2500,5:2600,6:2700},
    "合体": {1:2800,2:2900,3:3000,4:3100,5:3200,6:3300,7:3400,8:3500,9:3600},
    "大乗": {1:3700,2:3800,3:3900,4:4000,5:4100,6:4200,7:4300,8:4400,9:4500}
}

# ── セッション初期化 ──
if 'current_stage' not in st.session_state:
    st.session_state.current_stage = list(required_training.keys())[0]
if 'current_rank' not in st.session_state:
    st.session_state.current_rank = list(required_training[st.session_state.current_stage].keys())[0]
if 'current_value_w10k' not in st.session_state:
    st.session_state.current_value_w10k = 0
if 'target_value_w10k' not in st.session_state:
    raw = required_training[st.session_state.current_stage][st.session_state.current_rank]
    st.session_state.target_value_w10k = raw // 10000
if 'item_count' not in st.session_state:
    st.session_state.item_count = 0

def update_target():
    raw = required_training[st.session_state.current_stage][st.session_state.current_rank]
    st.session_state.target_value_w10k = raw // 10000

with st.expander("入力項目", expanded=True):
    st.selectbox("現在の境地",
                 list(required_training.keys()),
                 key="current_stage", on_change=update_target)
    st.selectbox("現在の段位",
                 list(required_training[st.session_state.current_stage].keys()),
                 key="current_rank", on_change=update_target)
    st.number_input("現在の修練値（万）",
                    min_value=0,
                    value=st.session_state.current_value_w10k,
                    step=1, key="current_value_w10k")
    st.number_input("目標修練値（万）",
                    min_value=0,
                    value=st.session_state.target_value_w10k,
                    step=1, key="target_value_w10k")
    st.number_input("アイテム個数 (1個→仙草3つ)",
                    min_value=0,
                    value=st.session_state.item_count,
                    step=1, key="item_count")

# ── 定数 ──
cycle_time    = 8
herb_interval = 15 * 60
herb_cycles   = 40
buff_options  = {"30%":0.30, "20%":0.20, "10%":0.10, "3%":0.03}

def simulate_time(remaining, base_speed, buff):
    """
    item_pts を先に remaining から引いた上でシミュレーションする
    戻り値: t, manual_pts, time_herb_pts
    """
    factor = base_speed * ((1 + buff) / cycle_time + herb_cycles / herb_interval)
    t = max(int(remaining / factor), 0)
    while True:
        manual_pts    = (t / cycle_time) * base_speed * (1 + buff)
        time_herb_pts = (t // herb_interval) * herb_cycles * base_speed
        if manual_pts + time_herb_pts >= remaining:
            # 最小化ループ
            while t > 0:
                t2 = t - 1
                m2 = (t2 / cycle_time) * base_speed * (1 + buff)
                th2= (t2 // herb_interval) * herb_cycles * base_speed
                if m2 + th2 < remaining:
                    break
                t = t2
            return t, manual_pts, time_herb_pts
        t += 1

if st.button("シミュレーション開始"):
    current   = st.session_state.current_value_w10k * 10000
    target    = st.session_state.target_value_w10k * 10000
    remaining = max(0, target - current)
    items     = st.session_state.item_count
    base_speed= training_speeds[st.session_state.current_stage][st.session_state.current_rank]
    now_jst   = datetime.now(ZoneInfo("Asia/Tokyo"))

    # 未使用シナリオ
    rows_unused = []
    for label, buff in buff_options.items():
        t0, m0, th0 = simulate_time(remaining, base_speed, buff)
        rows_unused.append({
            "バフ":             label,
            "到達時刻":         (now_jst + timedelta(seconds=t0)).strftime("%Y-%m-%d %H:%M"),
            "所要時間":         f"{t0//3600}h {(t0%3600)//60}m",
            "自動修練(万)":    f"{int(m0)//10000}",
            "時間仙草修練(万)": f"{int(th0)//10000}",
            "アイテム修練(万)": "0"
        })
    st.markdown("## 未使用")
    st.table(pd.DataFrame(rows_unused))

    # アイテム使用シナリオ
    if items > 0:
        per_item_pts = 3 * herb_cycles * base_speed
        needed_items= math.ceil(remaining / per_item_pts)
        used_items  = min(items, needed_items)

        # アイテムによる先行獲得ポイント
        item_pts = used_items * 3 * herb_cycles * base_speed
        remaining2= max(0, remaining - item_pts)

        rows_used = []
        for label, buff in buff_options.items():
            t1, m1, th1 = simulate_time(remaining2, base_speed, buff)
            rows_used.append({
                "バフ":             label,
                "到達時刻":         (now_jst + timedelta(seconds=t1)).strftime("%Y-%m-%d %H:%M"),
                "所要時間":         f"{t1//3600}h {(t1%3600)//60}m",
                "自動修練(万)":    f"{int(m1)//10000}",
                "時間仙草修練(万)": f"{int(th1)//10000}",
                "アイテム修練(万)": f"{int(item_pts)//10000}"
            })
        st.markdown(f"## {used_items}個使用")
        st.table(pd.DataFrame(rows_used))
