import streamlit as st
import math
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

st.title("いつ段位アップ？")

# 各境地・段位ごとの修練速度（1周天あたりのポイント）
training_speeds = {
    "錬気": {1: 100, 2: 200, 3: 300},
    "築基": {1: 400, 2: 500, 3: 600},
    "結丹": {1: 700, 2: 800, 3: 900},
    "元嬰": {1: 1000, 2: 1100, 3: 1200, 4: 1300, 5: 1400, 6: 1500},
    "化神": {1: 1600, 2: 1700, 3: 1800, 4: 1900, 5: 2000, 6: 2100},
    "錬虚": {1: 2200, 2: 2300, 3: 2400, 4: 2500, 5: 2600, 6: 2700},
    "合体": {1: 2800, 2: 2900, 3: 3000, 4: 3100, 5: 3200, 6: 3300, 7: 3400, 8: 3500, 9: 3600},
    "大乗": {1: 3700, 2: 3800, 3: 3900, 4: 4000, 5: 4100, 6: 4200, 7: 4300, 8: 4400, 9: 4500}
}

# メインエリアに入力パネルを配置（スマホでも見やすい）
with st.expander("入力項目", expanded=True):
    # 現在の境地・段位の入力
    current_stage = st.selectbox("現在の境地", list(training_speeds.keys()))
    current_rank = st.selectbox("現在の段位", list(training_speeds[current_stage].keys()))
    
    # 修練値は「万」単位で入力（内部では 10000 倍して扱う）
    current_value_input = st.number_input("現在の修練値（万）", min_value=0, value=0, step=1)
    target_value_input  = st.number_input("目標修練値（万）", min_value=1, value=1000, step=1)

# 定数
cycle_time   = 8            # 1周天に必要な秒数
herb_interval = 15 * 60     # 仙草が手に入る間隔（15分＝900秒）
herb_cycles  = 40           # 仙草1個で補助される周天数

# 霊峰バフのパターン（仙草には影響しない）
buff_options = {"30%": 0.30, "20%": 0.20, "10%": 0.10, "3%": 0.03}

# シミュレーション関数：必要な秒数 t を求め、手動・仙草の修練値を返す
def simulate_time(remaining, base_speed, buff, cycle_time, herb_interval, herb_cycles):
    # 連続としての推定値をまず算出
    estimation_factor = base_speed * ((1 + buff) / cycle_time + herb_cycles / herb_interval)
    t_est = int(remaining / estimation_factor)
    t = max(t_est, 0)
    while True:
        manual_points = (t / cycle_time) * base_speed * (1 + buff)
        herb_points   = (t // herb_interval) * herb_cycles * base_speed
        total_points  = manual_points + herb_points
        if total_points >= remaining:
            # 1秒前では条件を満たさなくなる、最小の t を求める
            while t > 0:
                t_minus = t - 1
                manual_points_minus = (t_minus / cycle_time) * base_speed * (1 + buff)
                herb_points_minus   = (t_minus // herb_interval) * herb_cycles * base_speed
                if manual_points_minus + herb_points_minus < remaining:
                    break
                t = t_minus
            return t, manual_points, herb_points
        t += 1

if st.button("シミュレーション開始"):
    # ユーザー入力の修練値は万単位なので 10000 倍して実際の数値に変換
    current_value = current_value_input * 10000
    target_value  = target_value_input * 10000
    remaining = target_value - current_value

    if remaining <= 0:
        st.success("既に目標修練値に達しています。")
    else:
        # 現在の境地・段位から基本の修練速度を取得
        base_speed = training_speeds[current_stage][current_rank]
        
        # JST (日本標準時) の現在時刻を取得
        now_jst = datetime.now(ZoneInfo("Asia/Tokyo"))
        
        results = []
        # それぞれのバフパターンでシミュレーション結果を求める
        for buff_label, buff in buff_options.items():
            t_required, manual_points, herb_points = simulate_time(remaining, base_speed, buff, cycle_time, herb_interval, herb_cycles)
            finish_time = now_jst + timedelta(seconds=t_required)
            hours   = t_required // 3600
            minutes = (t_required % 3600) // 60
            seconds = t_required % 60
            results.append({
                "バフ": buff_label,
                "予想到達時刻": finish_time.strftime('%Y-%m-%d %H:%M'),
                "所要時間": f"{hours} 時間 {minutes} 分",
                "自動修練(万)": f"{int(manual_points)/10000:.2f} 万",
                "仙草修練(万)": f"{int(herb_points)/10000:.1f} 万",
                "合計修練(万)": f"{int(manual_points + herb_points)/10000:.2f} 万",
            })
        
        df = pd.DataFrame(results)
        st.markdown("### シミュレーション結果")
        st.table(df)
