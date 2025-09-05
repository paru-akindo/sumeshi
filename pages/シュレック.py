# main.py
"""
水準評価モード専用アプリケーション
- 部品レベル入力を 2行×7列 のプルダウンで行う
- 各サイクルで得られる reward_table の報酬名別確率分布を表で表示
"""

import streamlit as st
import pandas as pd
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Tuple, List

# === 定数・テーブル定義 ===

upgrade_info = {
    "受付_A": [
        {"level": 1, "time": 45, "cost": 0},
        {"level": 2, "time": 40, "cost": 1000},
        {"level": 3, "time": 35, "cost": 8250},
        {"level": 4, "time": 30, "cost": 14700},
        {"level": 5, "time": 25, "cost": 19900},
        {"level": 6, "time": 20, "cost": 27600},
        {"level": 7, "time": 15, "cost": 40000},
    ],
    "受付_B": [
        {"level": 1, "time": 30, "cost": 0},
        {"level": 2, "time": 15, "cost": 17250},
        {"level": 3, "time": 7.5, "cost": 44650},
    ],
    "計測_A": [
        {"level": 1, "cost": 0},
        {"level": 2, "cost": 1000},
        {"level": 3, "cost": 8250},
        {"level": 4, "cost": 14700},
        {"level": 5, "cost": 19900},
        {"level": 6, "cost": 27600},
        {"level": 7, "cost": 40000},
    ],
    "計測_B": [
        {"level": 1, "time": 30, "cost": 0},
        {"level": 2, "time": 15, "cost": 17250},
        {"level": 3, "time": 7.5, "cost": 44650},
    ],
    "教室_A": [
        {"level": 1, "time": 60, "cost": 0},
        {"level": 2, "time": 30, "cost": 6900},
        {"level": 3, "time": 20, "cost": 11500},
        {"level": 4, "time": 15, "cost": 23000},
        {"level": 5, "time": 12, "cost": 31850},
        {"level": 6, "time": 10, "cost": 58800},
    ],
    "教室_B": [
        {"level": 1, "time": 60, "cost": 0},
        {"level": 2, "time": 30, "cost": 6900},
        {"level": 3, "time": 20, "cost": 11500},
        {"level": 4, "time": 15, "cost": 23000},
        {"level": 5, "time": 12, "cost": 31850},
        {"level": 6, "time": 10, "cost": 58800},
    ],
}

gym_A_levels = {
    1: ((2, 8), 0.15),
    2: ((7, 13), 0.20),
    3: ((12, 18), 0.25),
    4: ((15, 25), 0.30),
    5: ((22, 38), 0.35),
    6: ((37, 53), 0.40),
    7: ((52, 68), 0.45),
    8: ((65, 85), 0.50),
}

class_A_levels = {
    1: (7, 13),
    2: (10, 20),
    3: (17, 33),
    4: (30, 50),
    5: (47, 74),
    6: (70, 100),
    7: (97, 133),
}

# (合計ポイント low–high, 金貨)
reward_table = [
    (0, 79, 2),
    (80, 139, 5),
    (140, 219, 10),
    (220, 339, 15),
    (340, 459, 20),
    (460, 479, 25),
    (480, 619, 35),
    (620, 819, 40),
    (820, 1039, 50),
]

# 金貨量 → 報酬名
reward_names = {
    2:  "料理初心者",
    5:  "見習い料理人",
    10: "入門料理人",
    15: "初級料理人",
    20: "中級料理人",
    25: "上級料理人",
    35: "超級料理人",
    40: "特級厨師",
    50: "主席厨師",
    60: "厨神",
}

# コード → 部品名
code_to_part = {
    "1a": "受付_A", "1b": "受付_B",
    "2a": "計測_A", "2b": "計測_B",
    "3a": "教室_A1", "4a": "教室_A2", "5a": "教室_A3",
    "6a": "教室_A4", "7a": "教室_A5",
    "3b": "教室_B1", "4b": "教室_B2", "5b": "教室_B3",
    "6b": "教室_B4", "7b": "教室_B5",
}

# === データモデル ===

@dataclass
class EvaluationParams:
    levels: Dict[str, int]
    risk_factor: float = 1.0

@dataclass
class EvaluationResult:
    total_level: int
    cycle_reward: float
    cycle_time: int
    coin_rate: float
    hourly_rate: float

# === PMF ヘルパー ===

def convolve_pmfs(p1: Dict[int, float], p2: Dict[int, float]) -> Dict[int, float]:
    res = defaultdict(float)
    for v1, pr1 in p1.items():
        for v2, pr2 in p2.items():
            res[v1 + v2] += pr1 * pr2
    return dict(res)

def pmf_uniform(a: int, b: int, n: int) -> Dict[int, float]:
    if n <= 0:
        return {0: 1.0}
    base = 1.0 / (b - a + 1)
    pmf = {x: base for x in range(a, b + 1)}
    for _ in range(1, n):
        pmf = convolve_pmfs(pmf, {x: base for x in range(a, b + 1)})
    return pmf

def merge_pmfs(pmfs: List[Dict[int, float]]) -> Dict[int, float]:
    res = {0: 1.0}
    for pmf in pmfs:
        res = convolve_pmfs(res, pmf)
    return res

# === 期待値・レート計算 ===

_expected_cache: Dict[Tuple[int, Tuple[int, ...]], float] = {}

def get_classroom_hist(levels: Dict[str, int]) -> Tuple[int, ...]:
    counts = {i: 0 for i in range(1, 7)}
    for i in range(1, 6):
        lvl = levels.get(f"教室_A{i}", 1)
        counts[lvl] += 1
    return tuple(counts[i] for i in range(1, 7))

def combine_classroom_distribution(hist: Tuple[int, ...]) -> Dict[int, float]:
    pmfs = []
    for lvl, count in enumerate(hist, start=1):
        if count > 0:
            a, b = class_A_levels[lvl]
            pmfs.append(pmf_uniform(a, b, count))
    return merge_pmfs(pmfs) if pmfs else {0: 1.0}

def expected_cycle_reward_compressed(
    gym_level: int,
    class_hist: Tuple[int, ...]
) -> float:
    key = (gym_level, class_hist)
    if key in _expected_cache:
        return _expected_cache[key]
    (gmin, gmax), p_double = gym_A_levels[gym_level]
    gym_pmf = {x: 1.0/(gmax-gmin+1) for x in range(gmin, gmax+1)}
    class_pmf = combine_classroom_distribution(class_hist)
    total_pmf = convolve_pmfs(gym_pmf, class_pmf)
    exp_val = 0.0
    for pts, prob in total_pmf.items():
        coin = next((c for low, high, c in reward_table if low <= pts <= high), 0)
        exp_val += coin * (1 + p_double) * prob
    _expected_cache[key] = exp_val
    return exp_val

def compute_cycle_time(levels: Dict[str, int]) -> int:
    times: List[float] = []
    for part in ["受付_A", "受付_B", "計測_B"]:
        t = next((it["time"] for it in upgrade_info[part]
                  if it["level"] == levels.get(part, 1)), None)
        if t is not None:
            times.append(t)
    for i in range(1, 6):
        t = next((it["time"] for it in upgrade_info["教室_B"]
                  if it["level"] == levels.get(f"教室_B{i}", 1)), None)
        if t is not None:
            times.append(t)
    return int(max(times)) if times else 60

def get_coin_rate(levels: Dict[str, int], risk: float) -> float:
    gym_lv = levels["計測_A"]
    hist = get_classroom_hist(levels)
    reward = expected_cycle_reward_compressed(gym_lv, hist)
    cycle_time = compute_cycle_time(levels)
    return reward * risk / cycle_time

def total_level(levels: Dict[str, int]) -> int:
    keys = (
        ["受付_A", "受付_B", "計測_A", "計測_B"] +
        [f"教室_A{i}" for i in range(1, 6)] +
        [f"教室_B{i}" for i in range(1, 6)]
    )
    return sum(levels.get(k, 1) for k in keys)

# === 報酬分布計算 ===

def reward_distribution(
    gym_level: int,
    class_hist: Tuple[int, ...]
) -> Dict[int, float]:
    (gmin, gmax), p_double = gym_A_levels[gym_level]
    gym_pmf = {x: 1.0/(gmax-gmin+1) for x in range(gmin, gmax+1)}
    class_pmf = combine_classroom_distribution(class_hist)
    total_pmf = convolve_pmfs(gym_pmf, class_pmf)
    dist: Dict[int, float] = defaultdict(float)
    for pts, prob in total_pmf.items():
        coin = next((c for low, high, c in reward_table if low <= pts <= high), 0)
        dist[coin] += prob
    return dict(sorted(dist.items()))

# === 評価関数 ===

def evaluate(params: EvaluationParams) -> EvaluationResult:
    lv = params.levels
    rf = params.risk_factor
    tot_lv = total_level(lv)
    cy_rew = expected_cycle_reward_compressed(lv["計測_A"], get_classroom_hist(lv))
    cy_time = compute_cycle_time(lv)
    coin_rt = cy_rew * rf / cy_time
    hourly_rt = coin_rt * 3600
    return EvaluationResult(
        total_level=tot_lv,
        cycle_reward=cy_rew,
        cycle_time=cy_time,
        coin_rate=coin_rt,
        hourly_rate=hourly_rt,
    )

# === Streamlit UI ===

def main():
    st.title("玉龍幣シミュ")

    # リスク調整係数
    risk = st.sidebar.number_input(
        "リスク調整係数 (-r)", min_value=0.0, max_value=2.0, value=1.0, step=0.01
    )

    # 表示用行・列ラベル
    row_labels  = ["教師", "席"]
    row_codes   = ["a",   "b"]
    col_labels  = ["受付","腕前審査","料理","包丁","製菓","調理","盛付"]
    col_nums    = list(range(1, 8))

    st.markdown("### レベル入力")
    with st.form("level_form"):
        cols = st.columns(7)
        # 列ヘッダー（表示用ラベル）
        for col, lbl in zip(cols, col_labels):
            col.markdown(f"**{lbl}**")

        level_inputs: Dict[str, int] = {}
        # セルごとにプルダウン
        for rcode in row_codes:
            for col, num in zip(cols, col_nums):
                code = f"{num}{rcode}"
                part = code_to_part[code]
                # 内部テーブルキー選択
                if part.startswith("教室_A"):
                    key = "教室_A"
                elif part.startswith("教室_B"):
                    key = "教室_B"
                else:
                    key = part
                max_lv = max(it["level"] for it in upgrade_info[key])
                # ← ラベルを空文字に & collapsed で非表示に
                lvl = col.selectbox(
                    "",                          # ラベルを空に
                    options=list(range(1, max_lv+1)),
                    index=0,
                    key=code,                    # internal key はそのまま
                    label_visibility="collapsed" # ラベル非表示
                )
                level_inputs[code] = lvl

        submitted = st.form_submit_button("評価実行")


    if not submitted:
        return

    # 入力確認：2×7 テーブル表示
    data = [
        [level_inputs[f"{num}{rcode}"] for num in col_nums]
        for rcode in row_codes
    ]
    df_input = pd.DataFrame(data, index=row_labels, columns=col_labels)
    st.markdown("#### 入力内容")
    st.table(df_input)

    # 評価実行
    lvl_dict = {code_to_part[c]: level_inputs[c] for c in level_inputs}
    result = evaluate(EvaluationParams(levels=lvl_dict, risk_factor=risk))

    st.markdown("## 評価結果")
    st.write(f"- 合計レベル: {result.total_level}")
    st.write(f"- 料理人あたり玉龍幣期待値: {result.cycle_reward:.2f}")
    st.write(f"- サイクルタイム: {result.cycle_time} 秒")
    st.write(f"- 1時間あたり玉龍幣期待値: {result.hourly_rate:.2f}")

    # 報酬分布を表で表示（名前をインデックス）
    hist = get_classroom_hist(lvl_dict)
    dist = reward_distribution(lvl_dict["計測_A"], hist)
    names = [reward_names.get(c, str(c)) for c in dist.keys()]
    probs = [round(p*100, 2) for p in dist.values()]
    df_dist = pd.DataFrame({"確率(%)": probs}, index=names)
    st.markdown("## 料理人分布")
    st.table(df_dist)

if __name__ == "__main__":
    main()
