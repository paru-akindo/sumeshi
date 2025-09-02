import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

st.title("いつ段位アップ？")

# ── 昇段に必要な修練値マスタ（生ポイント） ──
required_training = {
    "錬気": {1: 41000,    2: 288200,   3: 497700},
    "築基": {1: 747600,   2: 1058000,  3: 1414000},
    "結丹": {1: 1874000,  2: 2364000,  3: 2980000},
    "元嬰": {1: 3667000,  2: 4466000,  3: 5341000,
           4: 6346000,  5: 7491000,  6: 8730000},
    "化神": {1: 10120000, 2: 11680000, 3: 13350000,
           4: 15200000, 5: 17180000, 6: 19340000},
    "錬虚": {1: 16190000, 2: 18300000, 3: 26970000,
           4: 63300000, 5: 68430000, 6: 73540000},
    "合体": {1: 79070000, 2: 84680000, 3: 0,
           4: 0,        5: 0,        6: 0,
           7: 0,        8: 0,        9: 0},
    "大乗": {1: 0,        2: 0,        3: 0,
           4: 0,        5: 0,        6: 0,
           7: 0,        8: 0,        9: 0}
}

# ── 周天あたりの修練速度（ポイント） ──
training_speeds = {
    "錬気": {1: 100,  2: 200,  3: 300},
    "築基": {1: 400,  2: 500,  3: 600},
    "結丹": {1: 700,  2: 800,  3: 900},
    "元嬰": {1: 1000, 2: 1100, 3: 1200,
           4: 1300, 5: 1400, 6: 1500},
    "化神": {1: 1600, 2: 1700, 3: 1800,
           4: 1900, 5: 2000, 6: 2100},
    "錬虚": {1: 2200, 2: 2300, 3: 2400,
           4: 2500, 5: 2600, 6: 2700},
    "合体": {1: 2800, 2: 2900, 3: 3000,
           4: 3100, 5: 3200, 6: 3300,
           7: 3400, 8: 3500, 9: 3600},
    "大乗": {1: 3700, 2: 3800, 3: 3900,
           4: 4000, 5: 4100, 6: 4200,
           7: 4300, 8: 4400, 9: 4500}
}

# ── セッション状態の初期化 ──
if 'current_stage' not in st.session_state:
    st.session_state.current_stage = list(required_training.keys())[0]
if 'current_rank' not in st.session_state:
    st.session_state.current_rank = list(required_training[st.session_state.current_stage].keys())[0]
if 'current_value_w10k' not in st.session_state:
    st.session_state.current_value_w10k = 0
if 'target_value_w10k' not in st.session_state:
    # 初期は現在境地・段位の必要ポイントを万単位に変換
    raw = required_training[st.session_state.current_stage][st.session_state.current_rank]
    st.session_state.target_value_w10k = raw // 10000

# ── 境地・段位が変わったら目標値(万)を更新 ──
def update_target():
    raw = required_training[
        st.session_state.current_stage
    ][
        st.session_state.current_rank
    ]
    st.session_state.target_value_w10k = raw // 10000

with st.expander("入力項目", expanded=True):
    # 境地選択
    st.selectbox(
        "現在の境地",
        list(required_training.keys()),
        key="current_stage",
        on_change=update_target
    )
    # 段位選択（境地ごとに可変）
    st.selectbox(
        "現在の段位",
        list(required_training[st.session_state.current_stage].keys()),
        key="current_rank",
        on_change=update_target
    )

    # 現在の修練値（万単位）
    st.number_input(
        "現在の修練値（万）",
        min_value=0,
        value=st.session_state.current_value_w10k,
        step=1,
        key="current_value_w10k"
    )

    # 目標修練値（万単位、自動更新）
    st.number_input(
        "目標修練値（万）",
        min_value=0,
        value=st.session_state.target_value_w10k,
        step=1,
        key="target_value_w10k"
    )

# ── シミュレーション用定数 ──
cycle_time    = 8            # 1周天に要する秒数
herb_interval = 15 * 60      # 仙草入手間隔（秒）
herb_cycles   = 40           # 仙草1個あたりの補助周天数
buff_options  = {"30%": 0.30, "20%": 0.20, "10%": 0.10, "3%": 0.03}

# ── シミュレーション関数 ──
def simulate_time(remaining, base_speed, buff):
    factor = base_speed * ((1 + buff) / cycle_time + herb_cycles / herb_interval)
    t_est = int(remaining / factor)
    t = max(t_est, 0)
    while True:
        manual_pts = (t / cycle_time) * base_speed * (1 + buff)
        herb_pts   = (t // herb_interval) * herb_cycles * base_speed
        if manual_pts + herb_pts >= remaining:
            # 1秒前チェック
            while t > 0:
                t_minus = t - 1
                m2 = (t_minus / cycle_time) * base_speed * (1 + buff)
                h2 = (t_minus // herb_interval) * herb_cycles * base_speed
                if m2 + h2 < remaining:
                    break
                t = t_minus
            return t, manual_pts, herb_pts
        t += 1

# ── シミュレーション実行 ──
if st.button("シミュレーション開始"):
    # 万単位→ポイント単位に変換
    current = st.session_state.current_value_w10k * 10000
    target  = st.session_state.target_value_w10k * 10000
    remaining = target - current

    if remaining <= 0:
        st.success("既に目標修練値に達しています。")
    else:
        base_speed = training_speeds[
            st.session_state.current_stage
        ][
            st.session_state.current_rank
        ]
        now_jst = datetime.now(ZoneInfo("Asia/Tokyo"))

        results = []
        for label, buff in buff_options.items():
            t_need, manual_pts, herb_pts = simulate_time(remaining, base_speed, buff)
            finish = now_jst + timedelta(seconds=t_need)
            hours   = t_need // 3600
            mins    = (t_need % 3600) // 60

            results.append({
                "バフ":       label,
                "予想到達時刻": finish.strftime("%Y-%m-%d %H:%M"),
                "所要時間":   f"{hours}h {mins}m",
                "手動修練(万)":   f"{int(manual_pts)//10000}",
                "仙草修練(万)":   f"{int(herb_pts)//10000}",
                "合計修練(万)":   f"{int((manual_pts+herb_pts))//10000}",
            })

        df = pd.DataFrame(results)
        st.markdown("### シミュレーション結果")
        st.table(df)
