import streamlit as st
import math
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

st.title("いつ段位アップ？")

# ── 昇段に必要な修練値 ──
required_training = {
    "良民": {3: 41000, 2: 288200, 1: 497700},
    "文人": {3: 747600, 2: 1058000, 1: 1414000},
    "才女": {3: 1874000, 2: 2364000, 1: 2980000},
    "学士": {6: 3667000, 5: 4466000, 4: 5341000, 3: 6346000, 2: 7491000, 1: 8730000},
    "翰林": {6: 10120000, 5: 11680000, 4: 13350000, 3: 15200000, 2: 17180000, 1: 19340000},
    "博雅": {6: 16190000, 5: 18300000, 4: 26900000, 3: 63300000, 2: 68430000, 1: 73540000},
    "名仕": {9: 79070000, 8: 84680000, 7: 0, 6: 0, 5: 0, 4: 0, 3: 0, 2: 0, 1: 0},
    "聖人": {9: 0, 8: 0, 7: 0, 6: 0, 5: 0, 4: 0, 3: 0, 2: 0, 1: 0}
}

# ── 周天あたり修練速度（基礎） ──
training_speeds = {
    "良民": {3: 100, 2: 200, 1: 300},
    "文人": {3: 400, 2: 500, 1: 600},
    "才女": {3: 700, 2: 800, 1: 900},
    "学士": {6: 1000, 5: 1100, 4: 1200, 3: 1300, 2: 1400, 1: 1500},
    "翰林": {6: 1600, 5: 1700, 4: 1800, 3: 1900, 2: 2000, 1: 2100},
    "博雅": {6: 2200, 5: 2300, 4: 2400, 3: 2500, 2: 2600, 1: 2700},
    "名仕": {9: 2800, 8: 2900, 7: 3000, 6: 3100, 5: 3200, 4: 3300, 3: 3400, 2: 3500, 1: 3600},
    "聖人": {9: 3700, 8: 3800, 7: 3900, 6: 4000, 5: 4100, 4: 4200, 3: 4300, 2: 4400, 1: 4500}
}

# ── 定数 ──
CYCLE_TIME = 8            # 1周天に必要な秒数
HERB_INTERVAL = 15 * 60   # 仙草が手に入る間隔（秒）
HERB_CYCLES = 40          # 仙草1個で補助される周天数
BUFF_OPTIONS = {"30%": 0.30, "20%": 0.20, "10%": 0.10, "3%": 0.03}

# ── セッション初期化 ──
if 'stage' not in st.session_state:
    st.session_state.stage = list(required_training.keys())[0]

# 初期段位は「その境地での最小速度の段位」を自動決定
def default_rank_for(stage):
    speeds = training_speeds.get(stage, {})
    if not speeds:
        return list(required_training.get(stage, {}).keys())[0]
    # 最小速度の段位を返す
    return min(speeds.keys(), key=lambda r: speeds[r])

if 'rank' not in st.session_state:
    st.session_state.rank = default_rank_for(st.session_state.stage)

if 'current_w10k' not in st.session_state:
    st.session_state.current_w10k = 0
if 'target_w10k' not in st.session_state:
    raw = required_training.get(st.session_state.stage, {}).get(st.session_state.rank, 0)
    st.session_state.target_w10k = raw // 10000
if 'item_count' not in st.session_state:
    st.session_state.item_count = 0

def update_target():
    raw = required_training.get(st.session_state.stage, {}).get(st.session_state.rank, 0)
    st.session_state.target_w10k = raw // 10000

# ── 入力パネル ──
with st.expander("入力項目", expanded=True):
    st.selectbox("現在の境地", list(required_training.keys()), key="stage", on_change=lambda: st.session_state.update({'rank': default_rank_for(st.session_state.stage)}) or update_target())

    # 境地に応じた段位リストを「修練速度の昇順」で作成し、先頭（最小速度）を初期表示にする
    speeds_for_stage = training_speeds.get(st.session_state.stage, {})
    if speeds_for_stage:
        ranks_for_stage = sorted(speeds_for_stage.keys(), key=lambda r: speeds_for_stage[r])  # 昇順（小→大）
    else:
        ranks_for_stage = sorted(required_training.get(st.session_state.stage, {}).keys(), reverse=True)

    # 初期表示は先頭（最小速度）
    default_index = 0
    st.selectbox("現在の段位", ranks_for_stage, key="rank", index=default_index, on_change=update_target)

    st.number_input("現在の修練値（万）", min_value=0, value=st.session_state.current_w10k, step=1, key="current_w10k")
    st.number_input("目標修練値（万）", min_value=0, value=st.session_state.target_w10k, step=1, key="target_w10k")
    st.number_input("アイテム個数（1個＝仙草3つ分）", min_value=0, value=st.session_state.item_count, step=1, key="item_count")

# ── シミュレーション関数 ──
def simulate_time(remaining, base_speed, buff):
    estimation_factor = base_speed * ((1 + buff) / CYCLE_TIME + HERB_CYCLES / HERB_INTERVAL)
    t = max(int(remaining / estimation_factor), 0)
    while True:
        manual_points = (t / CYCLE_TIME) * base_speed * (1 + buff)
        herb_points   = (t // HERB_INTERVAL) * HERB_CYCLES * base_speed
        total = manual_points + herb_points
        if total >= remaining:
            while t > 0:
                t2 = t - 1
                m2 = (t2 / CYCLE_TIME) * base_speed * (1 + buff)
                h2 = (t2 // HERB_INTERVAL) * HERB_CYCLES * base_speed
                if m2 + h2 < remaining:
                    break
                t = t2
            return t, manual_points, herb_points
        t += 1

# ── 実行 ──
if st.button("シミュレーション開始"):
    current = int(st.session_state.current_w10k * 10000)
    target  = int(st.session_state.target_w10k * 10000)
    remaining = max(0, target - current)
    items = int(st.session_state.item_count)
    base_speed = training_speeds.get(st.session_state.stage, {}).get(st.session_state.rank, 0)
    now_jst = datetime.now(ZoneInfo("Asia/Tokyo"))

    if remaining <= 0:
        st.success("既に目標に到達しています。")
    else:
        # アイテム未使用
        rows_no_item = []
        for label, buff in BUFF_OPTIONS.items():
            t, m, h = simulate_time(remaining, base_speed, buff)
            finish = now_jst + timedelta(seconds=t)
            rows_no_item.append({
                "バフ": label,
                "予想到達時刻": finish.strftime("%Y-%m-%d %H:%M"),
                "所要時間": f"{t//3600}時間 {(t%3600)//60}分",
                "自動修練(万)": f"{int(m)//10000}",
                "仙草修練(万)": f"{int(h)//10000}",
                "合計修練(万)": f"{int((m+h))//10000}"
            })
        st.markdown("### アイテム未使用")
        st.table(pd.DataFrame(rows_no_item))

        # アイテム使用
        if items > 0:
            per_item_pts = 3 * HERB_CYCLES * base_speed
            item_pts = items * per_item_pts
            remaining_after_items = max(0, remaining - item_pts)

            rows_with_item = []
            for label, buff in BUFF_OPTIONS.items():
                t, m, h = simulate_time(remaining_after_items, base_speed, buff)
                finish = now_jst + timedelta(seconds=t)
                rows_with_item.append({
                    "バフ": label,
                    "予想到達時刻": finish.strftime("%Y-%m-%d %H:%M"),
                    "所要時間": f"{t//3600}時間 {(t%3600)//60}分",
                    "自動修練(万)": f"{int(m)//10000}",
                    "仙草修練(万)": f"{int(h)//10000}",
                    "アイテム修練(万)": f"{int(item_pts)//10000}",
                    "合計修練(万)": f"{int((m+h+item_pts))//10000}"
                })
            st.markdown(f"### アイテム {items} 個使用（1個＝仙草3つ分）")
            st.table(pd.DataFrame(rows_with_item))
