import streamlit as st
from collections import defaultdict

# ============================================================
# イベント定義
# ============================================================
EVENT_DATA = {
    "shousen": {
        "name": "商戦",
        "costs": [8170, 81625, 163250, 653000],
        "feed":  [20, 200, 400, 1600],
        "item":  [1, 1, 2, 2]
    },
    "puzzle": {
        "name": "海上パズル",
        "costs": [550, 1175, 2350, 9400],
        "feed":  [20, 200, 400, 1600],
        "item":  [1, 1, 2, 2]
    },
    "nankai": {
        "name": "南海航路",
        "costs": [1, 4, 7, 28],
        "feed":  [20, 200, 400, 1600],
        "item":  [1, 1, 2, 2]
    },
    "hana": {
        "name": "花咲く春",
        "costs": [210, 850, 1700, 6800],
        "feed":  [20, 200, 400, 1600],
        "item":  [1, 1, 2, 2]
    }
}

EVENT_KEYS = list(EVENT_DATA.keys())
EVENT_JP = {k: v["name"] for k, v in EVENT_DATA.items()}

# ============================================================
# DP 用オプション生成
# ============================================================
OPTIONS = []
STAGE_MAP = {}

for ev, data in EVENT_DATA.items():
    for stage, cost in enumerate(data["costs"], start=1):
        feed = data["feed"][stage - 1]
        item = data["item"][stage - 1]
        OPTIONS.append((ev, cost, feed, item))
        STAGE_MAP[cost] = stage


# ============================================================
# 履歴の日本語化
# ============================================================
def convert_history(history):
    jp = []
    for ev, cost, feed, item in history:
        jp.append({
            "event": EVENT_JP[ev],
            "stage": STAGE_MAP[cost],
            "cost": cost,
            "feed": feed,
            "item": item
        })
    return jp


# ============================================================
# レベル3：完全一致 × 超高速DP（履歴復元つき）
# ============================================================
def optimize_training(points, N):
    # 初期状態
    dp = {
        tuple(points[k] for k in EVENT_KEYS): (0, 0, [])
    }

    # イベント順固定（高速化）
    ordered_events = ["nankai", "puzzle", "hana", "shousen"]

    ordered_options = []
    for ev in ordered_events:
        data = EVENT_DATA[ev]
        for stage, cost in enumerate(data["costs"], start=1):
            ordered_options.append((ev, cost, data["feed"][stage-1], data["item"][stage-1]))

    # DP 実行
    for _ in range(N):
        next_dp = {}

        for key, (f_now, it_now, hist_now) in dp.items():
            rem = dict(zip(EVENT_KEYS, key))

            for ev, cost, f_gain, it_gain in ordered_options:
                if rem[ev] < cost:
                    continue

                new_rem = rem.copy()
                new_rem[ev] -= cost

                new_key = (
                    new_rem["shousen"],
                    new_rem["puzzle"],
                    new_rem["nankai"],
                    new_rem["hana"]
                )

                new_state = (f_now + f_gain, it_now + it_gain, hist_now + [(ev, cost, f_gain, it_gain)])

                if new_key not in next_dp:
                    next_dp[new_key] = new_state
                else:
                    old = next_dp[new_key]
                    if new_state[0] > old[0] or new_state[1] > old[1]:
                        next_dp[new_key] = new_state

        dp = next_dp

    # 結果抽出
    results = list(dp.values())

    # 上位3つ
    top_mix  = sorted(results, key=lambda x: x[0] + x[1] * 100, reverse=True)[:3]
    top_feed = sorted(results, key=lambda x: (x[0], x[1]), reverse=True)[:3]
    top_item = sorted(results, key=lambda x: (x[1], x[0]), reverse=True)[:3]

    # 履歴を日本語化
    top_mix  = [(f, it, convert_history(h)) for f, it, h in top_mix]
    top_feed = [(f, it, convert_history(h)) for f, it, h in top_feed]
    top_item = [(f, it, convert_history(h)) for f, it, h in top_item]

    return top_mix, top_feed, top_item


# ============================================================
# Streamlit UI
# ============================================================
st.title("🐷 豚育成 最適化ツール")

points = {}
for key, ev in EVENT_DATA.items():
    points[key] = st.number_input(
        f"{ev['name']}ポイント",
        min_value=0,
        value=0,
        step=100
    )

count = st.number_input("残り育成回数（最大10）", min_value=1, max_value=10, value=5)

if st.button("最適化する"):
    top_mix, top_feed, top_item = optimize_training(points, count)

    st.subheader("🔥 複合スコア 最大パターン")
    for feed, item, hist in top_mix:
        st.write(f"**餌：{feed} / アイテム：{item}**")
        with st.expander("選択履歴"):
            for h in hist:
                st.write(f"{h['event']} 第{h['stage']}段階（コスト:{h['cost']}） 餌:{h['feed']} アイテム:{h['item']}")

    st.subheader("🍚 餌 最大パターン")
    for feed, item, hist in top_feed:
        st.write(f"**餌：{feed} / アイテム：{item}**")
        with st.expander("選択履歴"):
            for h in hist:
                st.write(f"{h['event']} 第{h['stage']}段階（コスト:{h['cost']}） 餌:{h['feed']} アイテム:{h['item']}")

    st.subheader("🎁 アイテム 最大パターン")
    for feed, item, hist in top_item:
        st.write(f"**餌：{feed} / アイテム：{item}**")
        with st.expander("選択履歴"):
            for h in hist:
                st.write(f"{h['event']} 第{h['stage']}段階（コスト:{h['cost']}） 餌:{h['feed']} アイテム:{h['item']}")
